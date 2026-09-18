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


def weighted_smooth_l1_loss(predict_edos, edos_target, edos_cov, predict_phdos, phdos_target, phdos_cov, use_mask, peak_w, tail_w, tail_start):
    """Smooth L1 with optional physical region weighting (C1.3).
    
    Returns (loss_edos, loss_phdos) scalars.
    """
    if peak_w == 1.0 and tail_w == 1.0:
        if use_mask:
            # C2.3: 覆盖bin内平均，全空行回退
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
        # C1.3 物理加权: 峰区(>均值+标准差)×peak_w + 声子尾部×tail_w
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
