import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import Transformer
from utils.builder import get_optimizer, get_lr_scheduler
from utils.metrics import MetricsRecorder
import utils.misc as utils
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
import numpy as np
import os

def compute_shape_loss(pred_shape, tgt_shape):
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

class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.dos_minmax = self.params.get("dos_minmax", False)
        self.dos_zscore = self.params.get("dos_zscore", False)
        self.apply_log = self.params.get("apply_log", False)
        self.scale_factor = self.params.get("scale_factor", 1.0)
        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None
        self.constants_len = self.params.get("constants_len", 0)
        # C1.3: TV/gradient loss + physical weighting (defaults = legacy behavior).
        self.tv_w = float(self.params.get("tv_w", 0.0))
        self.grad_w = float(self.params.get("grad_w", 0.0))
        self.peak_w = float(self.params.get("peak_w", 1.0))
        self.tail_w = float(self.params.get("tail_w", 1.0))
        self.tail_start = int(self.params.get("tail_start", -1))
        # C2.3: coverage-mask the loss (default off; eval protocol unchanged).
        self.use_mask = bool(self.params.get("use_mask", False))
        # C2.1: SumNorm-KL/W dual track + Huber (default smoothl1 legacy).
        self.loss_form = str(self.params.get("loss_form", "smoothl1"))
        self.w_w1 = float(self.params.get("w_w1", 1.0))
        self.w_huber = float(self.params.get("w_huber", 1.0))
        self.huber_delta = float(self.params.get("huber_delta", 0.02))
        self.begin_epoch = 0
        self.metric_best = 1000

        self.gscaler = amp.GradScaler(init_scale=1024, growth_interval=2000)
        
        # self.whether_final_test = self.params.get("final_test", False)
        # self.predict_length = self.params.get("predict_length", 20)

        # load model
        # print(params)
        sub_model = params.get('sub_model', {})
        # print(sub_model)
        for key in sub_model:
            if key == "transformer":
                self.model[key] = Transformer(**sub_model["transformer"])
            else:
                raise NotImplementedError('Invalid model type.')
            self.sub_model_name.append(key)

        # load optimizer and lr_scheduler
        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})
        # print(optimizer)
        # print(lr_scheduler)
        for key in self.sub_model_name:
            if key in optimizer:
                self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])
            if key in lr_scheduler:
                self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        # load metrics
        eval_metrics_list = params.get('metrics_list', [])
        if len(eval_metrics_list) > 0:
            self.eval_metrics = MetricsRecorder(eval_metrics_list)
        else:
            self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()

    def to(self, device):
        self.device = device
        for key in self.model:
            self.model[key].to(device)
        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def data_preprocess(self, data):
        # 按照 dataset.py 中 __getitem__ 的返回顺序解包（C2.3掩膜附后，无文件时为None）
        inp, pos, edos_target, phdos_target, \
        edos_mean, edos_std, edos_min, edos_max, \
        phdos_mean, phdos_std, phdos_min, phdos_max, \
        edos_cov, phdos_cov = data
        
        mask = (inp == 0)
        inp = inp.to(self.device, non_blocking=True)
        pos = pos.to(self.device, non_blocking=True)
        
        # 将两套 target 和统计量都送入 GPU
        edos_target = edos_target.to(self.device, non_blocking=True)
        phdos_target = phdos_target.to(self.device, non_blocking=True)
        
        edos_mean = edos_mean.to(self.device, non_blocking=True)
        edos_std  = edos_std.to(self.device, non_blocking=True)
        edos_min  = edos_min.to(self.device, non_blocking=True)
        edos_max  = edos_max.to(self.device, non_blocking=True)
        
        phdos_mean = phdos_mean.to(self.device, non_blocking=True)
        phdos_std  = phdos_std.to(self.device, non_blocking=True)
        phdos_min  = phdos_min.to(self.device, non_blocking=True)
        phdos_max  = phdos_max.to(self.device, non_blocking=True)

        mask = mask.clone().detach().to(dtype=torch.bool, device=self.device)

        # C2.3: coverage masks (None -> all-ones, legacy/v1 behavior).
        if edos_cov is None:
            edos_cov = torch.ones_like(edos_target, dtype=torch.bool)
        else:
            edos_cov = edos_cov.to(dtype=torch.bool, device=self.device)
        if phdos_cov is None:
            phdos_cov = torch.ones_like(phdos_target, dtype=torch.bool)
        else:
            phdos_cov = phdos_cov.to(dtype=torch.bool, device=self.device)

        return inp, pos, mask, edos_target, phdos_target, \
               edos_mean, edos_std, edos_min, edos_max, \
               phdos_mean, phdos_std, phdos_min, phdos_max, \
               edos_cov, phdos_cov

    def loss(self, predict, target):

        #norm = torch.norm(target, p=2)
                # 对非零值赋予更高的权重
        '''
        weights = target > 0
        predict[predict < 0] = 0
        diff = abs(predict-target)
        diff[weights] *= 2
        return torch.mean(diff)
        '''
        # §1.2: 纯端到端直接回归,统一为 Smooth-L1 (Huber)
        return F.smooth_l1_loss(predict, target)
        #return nn.functional.kl_div(predict.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='sum')
        #return self.lossfunc(predict, target)

    def train_one_step(self, batch_data, step):
        inp, pos, mask, edos_target, phdos_target, \
        edos_mean, edos_std, edos_min, edos_max, \
        phdos_mean, phdos_std, phdos_min, phdos_max, \
        edos_cov, phdos_cov = self.data_preprocess(batch_data)

        if len(self.model) == 1:
            outputs = self.model[list(self.model.keys())[0]](inp, mask, pos)
            predict_edos = outputs['edos']
            predict_phdos = outputs['phdos']
            if predict_edos.dim() == 3:
                predict_edos = predict_edos.squeeze(1)
            if predict_phdos.dim() == 3:
                predict_phdos = predict_phdos.squeeze(1)
        else:
            raise NotImplementedError('Invalid model type.')

        # FIX-P0-01: valid_atoms 哨兵对齐 (与 transformer.py 一致,严格对齐 src[:, 2:])
        atom_len = pos.shape[1] - 2
        valid_atoms = (~mask[:, 2:2 + atom_len]).sum(dim=-1).float()
        _ = valid_atoms  # 直接回归暂不使用求和规则,保留统计以备回退/日志

        # §1.2: 纯端到端直接回归 (Smooth-L1 / Huber),M1-M4 的 MSE 与 M5 的 5 项复合损失统一精简
        # L_total = SmoothL1(edos) + lambda_ph * SmoothL1(phdos), lambda_ph=1.0 (标准化空间等权)
        if self.loss_form == "sumnorm_klw":
            # C2.1: targets arrive sum-normalized (dataset dos_sumnorm; denorm via
            # sum slots keeps eval identical). Dual track KL + W1 + Huber on dists.
            # C2.3: masked softmax (covered bins only) when use_mask.
            def _klw(p_raw, q, cov):
                if self.use_mask:
                    eff = cov.clone()
                    eff[eff.sum(dim=-1) == 0] = True  # 全空行回退无掩膜
                    p_raw = p_raw.masked_fill(~eff, float("-inf"))
                logp = F.log_softmax(p_raw, dim=-1)
                # q含精确零(掩膜bin): 内层0*inf恒nan, 用where按eff选取丢弃(非传播).
                _inner = q * (q.clamp_min(1e-12).log() - logp)
                _sel = eff if self.use_mask else torch.ones_like(q, dtype=torch.bool)
                kl = torch.where(_sel, _inner, torch.zeros_like(_inner)).sum(dim=-1)
                p = logp.exp()
                if self.use_mask:
                    cw = eff.float()
                    w1 = ((p.cumsum(dim=-1) - q.cumsum(dim=-1)).abs() * cw).sum(dim=-1) / cw.sum(dim=-1).clamp_min(1)
                    hub = (F.huber_loss(p, q, reduction="none", delta=self.huber_delta) * cw).sum(dim=-1) / cw.sum(dim=-1).clamp_min(1)
                else:
                    w1 = (p.cumsum(dim=-1) - q.cumsum(dim=-1)).abs().mean(dim=-1)
                    hub = F.huber_loss(p, q, reduction="none", delta=self.huber_delta).mean(dim=-1)
                return kl + self.w_w1 * w1 + self.w_huber * hub
            loss_edos = _klw(predict_edos, edos_target, edos_cov).mean()
            loss_phdos = _klw(predict_phdos, phdos_target, phdos_cov).mean()
        elif self.peak_w == 1.0 and self.tail_w == 1.0:
            if self.use_mask:
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
                w = torch.where(peak, torch.full_like(w, self.peak_w), w)
                if tail and self.tail_start >= 0 and tgt.shape[-1] > self.tail_start:
                    w[..., self.tail_start:] *= self.tail_w
                return w
            loss_edos = (F.smooth_l1_loss(predict_edos, edos_target, reduction="none")
                         * _w(edos_target)).mean()
            loss_phdos = (F.smooth_l1_loss(predict_phdos, phdos_target, reduction="none")
                          * _w(phdos_target, tail=True)).mean()
        total_loss = loss_edos + 1.0 * loss_phdos
        # C1.3: TV(毛刺惩罚) + 梯度(峰形)正则
        loss_tv = torch.tensor(0.0, device=total_loss.device)
        loss_grad = torch.tensor(0.0, device=total_loss.device)
        if self.tv_w > 0:
            loss_tv = (predict_edos[:, 1:] - predict_edos[:, :-1]).abs().mean() \
                + (predict_phdos[:, 1:] - predict_phdos[:, :-1]).abs().mean()
            total_loss = total_loss + self.tv_w * loss_tv
        if self.grad_w > 0:
            ge = (predict_edos[:, 1:] - predict_edos[:, :-1]
                  - (edos_target[:, 1:] - edos_target[:, :-1])).pow(2).mean()
            gp = (predict_phdos[:, 1:] - predict_phdos[:, :-1]
                  - (phdos_target[:, 1:] - phdos_target[:, :-1])).pow(2).mean()
            loss_grad = ge + gp
            total_loss = total_loss + self.grad_w * loss_grad

        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            total_loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].step()
        else:
            raise NotImplementedError('Invalid model type.')

        return {
            'loss': total_loss.item(),
            'loss_edos': loss_edos.item(),
            'loss_phdos': loss_phdos.item(),
            'loss_tv': loss_tv.item(),
            'loss_grad': loss_grad.item(),
            'loss_shape_e': loss_shape_e.item() if 'loss_shape_e' in locals() else 0.0,
            'loss_shape_p': loss_shape_p.item() if 'loss_shape_p' in locals() else 0.0,
            'loss_scale_e': loss_scale_e.item() if 'loss_scale_e' in locals() else 0.0,
            'loss_scale_p': loss_scale_p.item() if 'loss_scale_p' in locals() else 0.0,
            'loss_gap': loss_gap.item() if 'loss_gap' in locals() else 0.0,
            'loss_sum': loss_sum.item() if 'loss_sum' in locals() else 0.0,
        }

    def multi_step_predict(self, batch_data, clim_time_mean_daily, data_std, index, batch_len):
        pass

    def test_one_step(self, batch_data, step=None, save_predict=False):
        # 1. 解包数据 (对应 dataset.py 返回的 14 个元素，末2为C2.3掩膜)
        inp, pos, mask, edos_target, phdos_target, \
        edos_mean, edos_std, edos_min, edos_max, \
        phdos_mean, phdos_std, phdos_min, phdos_max, \
        edos_cov, phdos_cov = self.data_preprocess(batch_data)

        # 2. 模型预测
        if len(self.model) == 1:
            # transformer.py 返回结果字典和 attention
            outputs = self.model[list(self.model.keys())[0]](inp, mask, pos)
            predict_edos = outputs['edos']
            predict_phdos = outputs['phdos']
            # 假设你还需要 attention 用于保存，取其中一个任务的即可
            attention = outputs.get('attn_edos', None) 
        else:
            raise NotImplementedError('Invalid model type.')

        # 3. 计算指标 (标准化空间)
        norm_metrics = {
            'NormMAE_edos': torch.mean(torch.abs(predict_edos - edos_target)).item(),
            'NormMAE_phdos': torch.mean(torch.abs(predict_phdos - phdos_target)).item(),
            'NormMSE_edos': torch.mean((predict_edos - edos_target)**2).item(),
            'NormMSE_phdos': torch.mean((predict_phdos - phdos_target)**2).item(),
        }

        # --- 内部辅助函数：逆归一化并计算所有指标 ---
        def compute_detailed_metrics(pred, target, m_mean, m_std, m_min, m_max, prefix, phys_pred=None):
            # A. 逆归一化 (Denormalization)
            if phys_pred is not None:
                # 盲测推理分支 (M5: Shape-Scale 模型，无真值 min/max 泄露)
                p_n = phys_pred.clone()
            else:
                # Oracle 逆归一化分支 (M1-M4)
                p_n = pred.clone()
                if self.dos_minmax:
                    p_n = p_n * (m_max - m_min) + m_min
                elif self.dos_zscore:
                    p_n = p_n * m_std + m_mean
                
                if self.apply_log:
                    p_n = torch.exp(p_n) - 1.0
                
                if self.scale_factor != 1.0:
                    p_n = p_n / self.scale_factor

            t_n = target.clone()
            if self.dos_minmax:
                t_n = t_n * (m_max - m_min) + m_min
            elif self.dos_zscore:
                t_n = t_n * m_std + m_mean
            
            if self.apply_log:
                t_n = torch.exp(t_n) - 1.0
            
            if self.scale_factor != 1.0:
                t_n = t_n / self.scale_factor

            p_n[p_n < 0] = 0 # 物理约束

            # B. 计算指标 (逐样本计算后取均值，与 evaluate_and_plot.py 严格一致)
            # H2 hygiene: per-sample math delegated to shared utils.metrics fn
            from utils.metrics import per_sample_spectral_metrics
            _m = per_sample_spectral_metrics(p_n, t_n)
            mae_per_sample, mse_per_sample, r2_per_sample = _m['mae'], _m['mse'], _m['r2']

            mae = torch.mean(mae_per_sample)
            mse = torch.mean(mse_per_sample)
            r2 = torch.mean(r2_per_sample)

            return {
                f'MAE_{prefix}': mae.item(),
                f'MSE_{prefix}': mse.item(),
                f'R2_{prefix}': r2.item(),
                f'pred_n_{prefix}': p_n,
                f'target_n_{prefix}': t_n
            }

        # 4. 分别执行指标计算 (若存在 phys_edos 则分别计算 Blind 与 Oracle 双轨)
        metrics_loss = {}
        if 'phys_edos' in outputs:
            # A. Blind 物理盲测指标 (主指标)
            metrics_edos = compute_detailed_metrics(
                predict_edos, edos_target, edos_mean, edos_std, edos_min, edos_max, "edos",
                phys_pred=outputs['phys_edos']
            )
            metrics_phdos = compute_detailed_metrics(
                predict_phdos, phdos_target, phdos_mean, phdos_std, phdos_min, phdos_max, "phdos",
                phys_pred=outputs['phys_phdos']
            )
            # B. Oracle 逆归一化指标 (shape * (max - min) + min)
            shape_e_orc = outputs['shape_edos'] * (edos_max - edos_min) + edos_min
            shape_p_orc = outputs['shape_phdos'] * (phdos_max - phdos_min) + phdos_min
            metrics_edos_orc = compute_detailed_metrics(
                predict_edos, edos_target, edos_mean, edos_std, edos_min, edos_max, "edos_oracle",
                phys_pred=shape_e_orc
            )
            metrics_phdos_orc = compute_detailed_metrics(
                predict_phdos, phdos_target, phdos_mean, phdos_std, phdos_min, phdos_max, "phdos_oracle",
                phys_pred=shape_p_orc
            )
            for d in [metrics_edos, metrics_phdos, metrics_edos_orc, metrics_phdos_orc]:
                metrics_loss.update({k: v for k, v in d.items() if 'pred_n' not in k and 'target_n' not in k})
        else:
            # M1-M4 传统 Oracle 逆归一化
            metrics_edos = compute_detailed_metrics(
                predict_edos, edos_target, edos_mean, edos_std, edos_min, edos_max, "edos"
            )
            metrics_phdos = compute_detailed_metrics(
                predict_phdos, phdos_target, phdos_mean, phdos_std, phdos_min, phdos_max, "phdos"
            )
            metrics_loss.update({k: v for k, v in metrics_edos.items() if 'pred_n' not in k and 'target_n' not in k})
            metrics_loss.update({k: v for k, v in metrics_phdos.items() if 'pred_n' not in k and 'target_n' not in k})
        
        # 保留原有 lp_loss 用于 checkpoint 选择 (通常用总 loss 或 edos loss)
        metrics_loss.update(norm_metrics)

        # 6. 保存逻辑 (保持原有逻辑)
        if save_predict:
            os.makedirs("dosdata", exist_ok=True)
            # 保存第一条样本的还原后数据
            np.savetxt(f"dosdata/edos_pred_{step}.txt", metrics_edos['pred_n_edos'][0].cpu().numpy())
            np.savetxt(f"dosdata/edos_tgt_{step}.txt", metrics_edos['target_n_edos'][0].cpu().numpy())
            np.savetxt(f"dosdata/phdos_pred_{step}.txt", metrics_phdos['pred_n_phdos'][0].cpu().numpy())
            np.savetxt(f"dosdata/phdos_tgt_{step}.txt", metrics_phdos['target_n_phdos'][0].cpu().numpy())
            
            if attention is not None:
                attn_data = attention.squeeze(0).cpu().numpy()
                np.save(f"dosdata/attention_{step}.npy", attn_data)

        return metrics_loss

    def train_one_epoch(self, train_data_loader, epoch, max_epoches):

        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)


        # test_logger = {}


        end_time = time.time()           
        for key in self.optimizer:              # only train model which has optimizer
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')
        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'
        for step, batch in enumerate(train_data_loader):

            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch*max_step+step)
        
            # record data read time
            data_time.update(time.time() - end_time)
            # train one step
            loss = self.train_one_step(batch, step)

            # record loss and time
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            # output to logger
            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
                # begin_time1 = time.time()
                # print("logger output time:", begin_time1-end_time)

    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        for key in checkpoint_model:
            self.model[key].load_state_dict(checkpoint_model[key])
        for key in checkpoint_optimizer:
            self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
        for key in checkpoint_lr_scheduler:
            self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        self.begin_epoch = checkpoint_dict['epoch']
        if 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])
        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=self.begin_epoch, metric_best=self.metric_best))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'): 
        checkpoint_savedir = Path(checkpoint_savedir)
        checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth' \
                            if save_type == 'save_best' else 'checkpoint_latest.pth')
        # print(save_type, checkpoint_path)
        if utils.get_world_size() > 1:
            utils.save_on_master(
                {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].module.state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                }, checkpoint_path
            )
        else:
            utils.save_on_master(
                {
                    'epoch':            epoch+1,
                    'model':            {key: self.model[key].state_dict() for key in self.model},
                    'optimizer':        {key: self.optimizer[key].state_dict() for key in self.optimizer},
                    'lr_scheduler':     {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                }, checkpoint_path
            )

    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False



    def trainer(self, train_data_loader, test_data_loader, valid_data_loader, max_epoches, checkpoint_savedir=None, resume=False):
        for epoch in range(self.begin_epoch, max_epoches):

            train_data_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            # # update lr_scheduler
            # begin_time = time.time()

            
            # begin_time1 = time.time()
            # print("lrscheduler time:", begin_time1 - begin_time)
            # test model
            #metric_logger = self.test(valid_data_loader, epoch)
            metric_logger = self.test(valid_data_loader, epoch)
            

            # begin_time2 = time.time()
            # print("test time:", begin_time2 - begin_time1)

            
            # save model
            if checkpoint_savedir is not None:
                if self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                if (epoch + 1) % 1 == 0:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')
            # end_time = time.time()
            # print("save model time", end_time - begin_time2)
        

    @torch.no_grad()
    def test(self, test_data_loader, epoch, save_predict=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # set model to eval
        for key in self.model:
            self.model[key].eval()


        for step, batch in enumerate(test_data_loader):
            loss = self.test_one_step(batch, save_predict=save_predict, step=step)
            if 'MAE_edos' in loss and 'MAE_phdos' in loss:
                loss['total_MAE'] = loss['MAE_edos'] + loss['MAE_phdos']
            
            if 'MSE_edos' in loss and 'MSE_phdos' in loss:
                loss['total_MSE'] = loss['MSE_edos'] + loss['MSE_phdos']

            if 'NormMAE_edos' in loss and 'NormMAE_phdos' in loss:
                loss['total_NormMAE'] = loss['NormMAE_edos'] + loss['NormMAE_phdos']
                loss['balanced_score'] = 0.5 * loss['NormMAE_edos'] + 0.5 * loss['NormMAE_phdos']
            
            if 'NormMSE_edos' in loss and 'NormMSE_phdos' in loss:
                loss['total_NormMSE'] = loss['NormMSE_edos'] + loss['NormMSE_phdos']

            metric_logger.update(**loss)
            
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger

    @torch.no_grad()
    def test_final(self, valid_data_loader, predict_length):
        metric_logger = []
        for i in range(predict_length):
            metric_logger.append(utils.MetricLogger(delimiter="  "))
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        data_mean, data_std = valid_data_loader.dataset.get_meanstd()
        clim_time_mean_daily = valid_data_loader.dataset.get_clim_daily()
        clim_time_mean_daily = clim_time_mean_daily.to(self.device)
        data_std = data_std.to(self.device)
        index = 0
        for step, batch in enumerate(valid_data_loader):
            #print(step)
            batch_len = batch[0].shape[0]
            losses = self.multi_step_predict(batch, clim_time_mean_daily, data_std, index, batch_len)
            for i in range(len(losses)):
                metric_logger[i].update(**losses[i])
            index += batch_len

            self.logger.info("#"*80)

            for i in range(predict_length):
                self.logger.info('  '.join(
                        [f'final valid {i}th step predict (val stats)',
                        "{meters}"]).format(
                            meters=str(metric_logger[i])
                        ))

        return None

    def stat(self):
        pass


