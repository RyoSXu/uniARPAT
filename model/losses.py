"""Loss functions for uniARPAT.

Extracted from model.py to reduce God Class complexity.
All losses operate on normalized-space predictions and targets.
"""
import torch
import torch.nn.functional as F


def compute_shape_loss(pred_shape, tgt_shape):
    """Pearson + MSE shape loss (M5 shape-scale decoupling)."""
    pred_mean = torch.mean(pred_shape, dim=-1, keepdim=True)
    tgt_mean = torch.mean(tgt_shape, dim=-1, keepdim=True)
    pred_diff = pred_shape - pred_mean
    tgt_diff = tgt_shape - tgt_mean
    var_tgt = torch.mean(tgt_diff ** 2, dim=-1)

    cov = torch.sum(pred_diff * tgt_diff, dim=-1)
    std_p = torch.sqrt(torch.sum(pred_diff ** 2, dim=-1) + 1e-8)
    std_t = torch.sqrt(torch.sum(tgt_diff ** 2, dim=-1) + 1e-8)
    r_pearson = cov / (std_p * std_t + 1e-8)
    loss_pearson = 1.0 - r_pearson

    loss_mse = torch.mean((pred_shape - tgt_shape)**2, dim=-1)
    flat_mask = (var_tgt < 1e-4)
    loss_sample = torch.where(flat_mask, loss_mse, loss_pearson + 0.5 * loss_mse)
    return loss_sample.mean()


def sumnorm_klw_loss(pred_raw, target, cov, use_mask, w_w1, w_huber, huber_delta):
    """SumNorm KL + W1 + Huber loss (C2.1).
    
    Returns per-sample loss tensor [B].
    """
    if use_mask:
        eff = cov.clone()
        eff[eff.sum(dim=-1) == 0] = True  # 全空行回退无掩膜
        pred_raw = pred_raw.masked_fill(~eff, float("-inf"))
    logp = F.log_softmax(pred_raw, dim=-1)
    # q含精确零(掩膜bin): 内层0*inf恒nan, 用where按eff选取丢弃(非传播).
    _inner = target * (target.clamp_min(1e-12).log() - logp)
    _sel = eff if use_mask else torch.ones_like(target, dtype=torch.bool)
    kl = torch.where(_sel, _inner, torch.zeros_like(_inner)).sum(dim=-1)
    p = logp.exp()
    if use_mask:
        cw = eff.float()
        w1 = ((p.cumsum(dim=-1) - target.cumsum(dim=-1)).abs() * cw).sum(dim=-1) / cw.sum(dim=-1).clamp_min(1)
        hub = (F.huber_loss(p, target, reduction="none", delta=huber_delta) * cw).sum(dim=-1) / cw.sum(dim=-1).clamp_min(1)
    else:
        w1 = (p.cumsum(dim=-1) - target.cumsum(dim=-1)).abs().mean(dim=-1)
        hub = F.huber_loss(p, target, reduction="none", delta=huber_delta).mean(dim=-1)
    return kl + w_w1 * w1 + w_huber * hub


def edos_slope_matching_loss(pred_logits, target):
    """Match first differences between normalized eDOS probability shapes."""
    if pred_logits.ndim != 2 or pred_logits.shape != target.shape:
        raise ValueError("eDOS logits and targets must be matching [batch, bins] tensors")
    if pred_logits.shape[-1] < 2:
        raise ValueError("eDOS slope loss requires at least two bins")
    predicted_shape = F.softmax(pred_logits.float(), dim=-1)
    target_shape = target.float()
    predicted_slope = predicted_shape[:, 1:] - predicted_shape[:, :-1]
    target_slope = target_shape[:, 1:] - target_shape[:, :-1]
    return F.mse_loss(predicted_slope, target_slope)


def edos_pair_contrast_loss(logits_a, logits_b, target_a, target_b):
    """Match the signed eDOS difference for paired, sum-normalized spectra."""
    tensors = (logits_a, logits_b, target_a, target_b)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("pair logits and targets must be [pairs, bins] tensors")
    if any(tensor.shape != logits_a.shape for tensor in tensors[1:]):
        raise ValueError("pair logits and targets must have matching shapes")
    if logits_a.shape[0] == 0 or logits_a.shape[1] == 0:
        raise ValueError("pair loss requires at least one pair and one bin")
    if not torch.isfinite(target_a).all() or not torch.isfinite(target_b).all():
        raise ValueError("pair targets must be finite")
    predicted_difference = (
        F.softmax(logits_a.float(), dim=-1) - F.softmax(logits_b.float(), dim=-1)
    )
    target_difference = target_a.float() - target_b.float()
    return 0.5 * (predicted_difference - target_difference).abs().sum(dim=-1).mean()


def calibrate_additive_loss_weight(base_loss, added_loss, parameters, target_ratio, eps=1e-12):
    """Scale an added loss to a fixed fraction of the base gradient norm."""
    if target_ratio <= 0 or not torch.isfinite(torch.tensor(target_ratio)):
        raise ValueError("target_ratio must be finite and positive")
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("loss calibration requires trainable parameters")

    base_gradients = torch.autograd.grad(
        base_loss, parameters, retain_graph=True, allow_unused=True
    )
    added_gradients = torch.autograd.grad(
        added_loss, parameters, allow_unused=True
    )

    def gradient_norm(gradients):
        present = [
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not present:
            return torch.tensor(0.0)
        return torch.stack(present).sum().sqrt()

    base_norm = gradient_norm(base_gradients)
    added_norm = gradient_norm(added_gradients)
    if not torch.isfinite(base_norm) or not torch.isfinite(added_norm):
        raise ValueError("loss calibration produced a non-finite gradient norm")
    if base_norm <= eps or added_norm <= eps:
        raise ValueError("loss calibration requires nonzero base and added gradients")

    weight = target_ratio * base_norm / added_norm
    if not torch.isfinite(weight) or weight <= 0:
        raise ValueError("loss calibration produced an invalid weight")
    return float(weight.item()), float(base_norm.item()), float(added_norm.item())


def weighted_smooth_l1_loss(predict_edos, edos_target, edos_cov, predict_phdos, phdos_target, phdos_cov, use_mask, peak_w, tail_w, tail_start):
    """Smooth L1 with optional physical region weighting (C1.3).
    
    Returns (loss_edos, loss_phdos) scalars.
    """
    if peak_w == 1.0 and tail_w == 1.0:
        if use_mask:
            # Average over covered bins; fall back safely when a row is empty.
            def _mmean(elem, cov):
                cw = cov.float()
                cw[cw.sum(dim=-1) == 0] = 1.0
                return ((elem * cw).sum(dim=-1) / cw.sum(dim=-1).clamp_min(1)).mean()
            loss_edos = _mmean(F.smooth_l1_loss(predict_edos, edos_target, reduction="none"), edos_cov)
            loss_phdos = _mmean(F.smooth_l1_loss(predict_phdos, phdos_target, reduction="none"), phdos_cov)
        else:
            loss_edos = F.smooth_l1_loss(predict_edos, edos_target)
            loss_phdos = F.smooth_l1_loss(predict_phdos, phdos_target)
    else:
        # Weight high-density eDOS bins and the selected phDOS tail region.
        def _w(tgt, tail=False):
            w = torch.ones_like(tgt)
            peak = tgt > (tgt.mean(dim=-1, keepdim=True) + tgt.std(dim=-1, keepdim=True))
            w = torch.where(peak, torch.full_like(w, peak_w), w)
            if tail and tail_start >= 0 and tgt.shape[-1] > tail_start:
                w[..., tail_start:] *= tail_w
            return w
        loss_edos = (F.smooth_l1_loss(predict_edos, edos_target, reduction="none")
                     * _w(edos_target)).mean()
        loss_phdos = (F.smooth_l1_loss(predict_phdos, phdos_target, reduction="none")
                      * _w(phdos_target, tail=True)).mean()
    return loss_edos, loss_phdos


def tv_loss(pred_edos, pred_phdos):
    """Total variation smoothness penalty (C1.3)."""
    return (pred_edos[:, 1:] - pred_edos[:, :-1]).abs().mean() \
        + (pred_phdos[:, 1:] - pred_phdos[:, :-1]).abs().mean()


def gradient_loss(pred_edos, pred_phdos, tgt_edos, tgt_phdos):
    """Gradient-matching loss (C1.3)."""
    ge = (pred_edos[:, 1:] - pred_edos[:, :-1]
          - (tgt_edos[:, 1:] - tgt_edos[:, :-1])).pow(2).mean()
    gp = (pred_phdos[:, 1:] - pred_phdos[:, :-1]
          - (tgt_phdos[:, 1:] - tgt_phdos[:, :-1])).pow(2).mean()
    return ge + gp
