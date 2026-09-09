import os
import time
import argparse
import random
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from utils.builder import ConfigBuilder
from model.model import basemodel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('ablation')


def setup_ablation_seed(seed: int):
    """H1 hygiene: deterministic seeding for perfect对照 (init + shuffle + cudnn).

    NOTE: torch.backends.cudnn.deterministic=True costs speed; enabled here
    because ablation comparability outranks throughput (V100 has headroom).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL_CONFIGS = {
    'M1': {
        'desc': 'Baseline: Shared Decoder, Asymmetric Heads (1-layer eDOS / 6-layer phDOS), Oracle Scale (77.63M params)',
        'transformer_params': {
            'decoupled_decoder': False,
            'use_gated_cross_attn': False,
            'head_type': 'legacy',
            'predict_scale': False
        }
    },
    'M2': {
        'desc': 'Decoupled Decoder, Trimmed phDOS Head (3-layer 0.788M), Oracle Scale (72.96M params, -4.67M vs M1)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': False,
            'head_type': 'ph_trimmed',
            'predict_scale': False
        }
    },
    'M3': {
        'desc': 'Decoupled Decoder, Capacity Symmetric Heads (0.788M each), Oracle Scale (73.75M params, -3.88M vs M1)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': False,
            'head_type': 'symmetric',
            'predict_scale': False
        }
    },
    'M4': {
        'desc': 'Decoupled Decoder, Symmetric Heads, Post-Decoder Zero-Init Gated Cross Attention (75.85M params, +2.10M vs M3)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': True,
            'head_type': 'symmetric',
            'predict_scale': False
        }
    },
    'M5': {
        'desc': 'Full uniARPAT: Decoupled, Symmetric, Gated Attention, Shape-Scale Decoupling & Physical Loss (75.93M params)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': True,
            'head_type': 'symmetric',
            'predict_scale': True
        }
    }
}

def train_and_eval(model_name: str, epochs: int = 100, batch_size: int = 32, lr: float = 5e-5, skip_existing: bool = False, seed: int = 42):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    setup_ablation_seed(seed)

    summary_file = f"./results/test_{model_name.lower()}_summary.csv"
    if skip_existing and os.path.exists(summary_file):
        logger.info(f"[{model_name}] Already completed ({summary_file} exists). Skipping.")
        return pd.read_csv(summary_file).to_dict(orient='records')[0]

    config_info = MODEL_CONFIGS[model_name]
    logger.info("=" * 70)
    logger.info(f"   STARTING ABLATION EXPERIMENT: {model_name}")
    logger.info(f"   Description: {config_info['desc']}")
    logger.info(f"   Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Seed: {seed}")
    logger.info("=" * 70)

    save_dir = f"./output/ablation_{model_name.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    with open('configs/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['model']['params']['sub_model']['transformer'].update(config_info['transformer_params'])
    cfg['model']['params']['dos_minmax'] = True
    cfg['model']['params']['save_best'] = 'balanced_score'
    cfg['dataset']['train']['data_dir'] = './data/train4ARPAT'
    cfg['dataset']['valid']['data_dir'] = './data/train4ARPAT'
    cfg['dataset']['test']['data_dir'] = './data/train4ARPAT'

    builder = ConfigBuilder(**cfg)

    train_loader = builder.get_dataloader(split='train', dos_minmax=True, batch_size=batch_size)
    val_loader = builder.get_dataloader(split='valid', dos_minmax=True, batch_size=batch_size)
    test_loader = builder.get_dataloader(split='test', dos_minmax=True, batch_size=batch_size)

    model = builder.get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_params = sum(p.numel() for p in model.model['transformer'].parameters() if p.requires_grad)
    logger.info(f"[{model_name}] Verified Trainable Parameters: {total_params:,} ({total_params/1e6:.3f}M)")

    optimizer = model.optimizer['transformer']
    # H4 hygiene: warmup+cosine shared with train.py semantics (was bare cosine).
    from utils.builder import build_warmup_cosine_scheduler
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs)

    best_val_score = float('inf')
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        # H1 hygiene: reshuffle each epoch (DistributedSampler defaults to epoch=0
        # forever when set_epoch is never called -> identical batch order every epoch).
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.model['transformer'].train()
        train_loss = 0.0
        sub_loss_accum = {}
        n_batches = len(train_loader)
        t_start = time.time()

        for step, batch in enumerate(train_loader):
            loss_dict = model.train_one_step(batch, step=step)
            train_loss += loss_dict['loss']
            for k, v in loss_dict.items():
                sub_loss_accum[k] = sub_loss_accum.get(k, 0.0) + v
            if (step + 1) % 100 == 0 or (step + 1) == n_batches:
                logger.info(
                    f"[{model_name}] Epoch [{epoch+1:03d}/{epochs:03d}] Step [{step+1:03d}/{n_batches:03d}] | "
                    f"Loss: {loss_dict['loss']:.4f} (eDOS: {loss_dict['loss_edos']:.4f}, phDOS: {loss_dict['loss_phdos']:.4f})"
                )

        scheduler.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ep_time = time.time() - t_start
        train_loss /= n_batches
        avg_sub_losses = {f"train_{k}": v / n_batches for k, v in sub_loss_accum.items()}

        # 显存实测打点 (§6 补正 1):每轮记录峰值显存,写入 history_*.csv 的 peak_vram_mb 列
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        # Extract gate scalars if gated cross-attention is present
        alpha_e, alpha_p = 0.0, 0.0
        if model.model['transformer'].use_gated_cross_attn:
            alpha_e = model.model['transformer'].gated_cross_attn.alpha_e.item()
            alpha_p = model.model['transformer'].gated_cross_attn.alpha_p.item()

        # Validation (Dual-track Blind & Oracle for M5)
        val_metrics = evaluate_split(model, val_loader, is_m5=(model_name == 'M5'))
        balanced = 0.5 * val_metrics['mae_edos_median'] + 0.5 * val_metrics['mae_phdos_median']

        log_str = (
            f"[{model_name}] Epoch [{epoch+1:03d}/{epochs:03d}] ({ep_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val eDOS R2(med): {val_metrics['r2_edos_median']:.3f} (MAE: {val_metrics['mae_edos_median']:.3f}) | "
            f"Val phDOS R2(med): {val_metrics['r2_phdos_median']:.3f} (MAE: {val_metrics['mae_phdos_median']:.4f}) | "
            f"Balanced Score: {balanced:.4f}"
        )
        if model_name == 'M5':
            log_str += f" | Oracle eDOS R2: {val_metrics['oracle_r2_edos_median']:.3f}, phDOS R2: {val_metrics['oracle_r2_phdos_median']:.3f}"
        if model.model['transformer'].use_gated_cross_attn:
            log_str += f" | Gate (a_e={alpha_e:.4f}, a_p={alpha_p:.4f})"
        logger.info(log_str)

        history.append({
            'epoch': epoch + 1,
            'seed': seed,
            'epoch_time_s': ep_time,
            'peak_vram_mb': peak_vram_mb,
            'alpha_e': alpha_e,
            'alpha_p': alpha_p,
            **avg_sub_losses,
            **val_metrics,
            'balanced_score': balanced
        })

        # 如需逐轮独立峰值则重置计数器,下一轮重新统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if balanced < best_val_score:
            best_val_score = balanced
            best_epoch = epoch + 1
            torch.save(model.model['transformer'].state_dict(), os.path.join(save_dir, 'checkpoint_best.pth'))
            logger.info(f"[{model_name}] New best model saved at Epoch {epoch+1} (Score: {balanced:.4f})")

        torch.save(model.model['transformer'].state_dict(), os.path.join(save_dir, 'checkpoint_latest.pth'))

    # Save training history
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"./results/history_{model_name.lower()}.csv", index=False)

    # Load best checkpoint and evaluate on Test set
    logger.info(f"\nEvaluating Best Model ({model_name}, Epoch {best_epoch}) on Test Set (1371 materials)...")
    best_ckpt = torch.load(os.path.join(save_dir, 'checkpoint_best.pth'))
    model.model['transformer'].load_state_dict(best_ckpt)

    test_metrics = evaluate_split(model, test_loader, is_m5=(model_name == 'M5'), return_sample_level=True)
    df_samples = test_metrics.pop('sample_df')
    pred_p = test_metrics.pop('pred_phdos')
    pred_e = test_metrics.pop('pred_edos')
    tgt_p = test_metrics.pop('tgt_phdos')
    tgt_e = test_metrics.pop('tgt_edos')

    np.save(f"./results/pred_phdos_{model_name.lower()}.npy", pred_p)
    np.save(f"./results/pred_edos_{model_name.lower()}.npy", pred_e)
    if model_name == 'M5':
        np.save('./results/pred_phdos.npy', pred_p)
        np.save('./results/pred_edos.npy', pred_e)
        np.save('./results/tgt_phdos.npy', tgt_p)
        np.save('./results/tgt_edos.npy', tgt_e)
        df_samples.to_csv('./results/test_evaluation_summary.csv', index=False)

    df_samples.to_csv(f"./results/samples_{model_name.lower()}_test.csv", index=False)

    df_test_summary = pd.DataFrame([test_metrics])
    df_test_summary.to_csv(f"./results/test_{model_name.lower()}_summary.csv", index=False)

    logger.info("=" * 70)
    logger.info(f"   TEST EVALUATION SUMMARY ({model_name})")
    logger.info("=" * 70)
    for k, v in test_metrics.items():
        logger.info(f"   {k:28s}: {v:.4f}" if isinstance(v, float) else f"   {k:28s}: {v}")
    logger.info("=" * 70)
    return test_metrics

def evaluate_split(model, dataloader, is_m5: bool = False, return_sample_level: bool = False):
    model.model['transformer'].eval()
    records = []
    if return_sample_level:
        all_p_p, all_t_p = [], []
        all_p_e, all_t_e = [], []

    with torch.no_grad():
        for batch in dataloader:
            inp, pos, mask, edos_tgt, phdos_tgt, \
            edos_m, edos_s, edos_min, edos_max, \
            phdos_m, phdos_s, phdos_min, phdos_max = model.data_preprocess(batch)

            outputs = model.model['transformer'](inp, mask, pos)

            # Targets in true physical space
            t_e = edos_tgt * (edos_max - edos_min) + edos_min
            t_p = phdos_tgt * (phdos_max - phdos_min) + phdos_min

            if is_m5:
                # 1. Blind physical prediction
                p_e = torch.clamp(outputs['phys_edos'], min=0.0)
                p_p = torch.clamp(outputs['phys_phdos'], min=0.0)
                # 2. Oracle denormalization (with ground-truth min/max)
                p_e_orc = torch.clamp(outputs['shape_edos'] * (edos_max - edos_min) + edos_min, min=0.0)
                p_p_orc = torch.clamp(outputs['shape_phdos'] * (phdos_max - phdos_min) + phdos_min, min=0.0)
            else:
                # Oracle denormalization for M1-M4
                p_e = torch.clamp(outputs['edos'] * (edos_max - edos_min) + edos_min, min=0.0)
                p_p = torch.clamp(outputs['phdos'] * (phdos_max - phdos_min) + phdos_min, min=0.0)
                p_e_orc, p_p_orc = None, None

            # Sample-wise primary metrics (H2 hygiene: shared fn, bit-identical math)
            from utils.metrics import per_sample_spectral_metrics
            _me = per_sample_spectral_metrics(p_e, t_e)
            mae_e, mse_e, r2_e = _me['mae'], _me['mse'], _me['r2']
            _mp = per_sample_spectral_metrics(p_p, t_p)
            mae_p, mse_p, r2_p = _mp['mae'], _mp['mse'], _mp['r2']

            # Sample-wise Oracle metrics if M5
            if is_m5:
                _meo = per_sample_spectral_metrics(p_e_orc, t_e)
                mae_e_orc, mse_e_orc, r2_e_orc = _meo['mae'], _meo['mse'], _meo['r2']
                _mpo = per_sample_spectral_metrics(p_p_orc, t_p)
                mae_p_orc, mse_p_orc, r2_p_orc = _mpo['mae'], _mpo['mse'], _mpo['r2']

            if return_sample_level:
                all_p_p.append(p_p.cpu().numpy())
                all_t_p.append(t_p.cpu().numpy())
                all_p_e.append(p_e.cpu().numpy())
                all_t_e.append(t_e.cpu().numpy())

            B = inp.shape[0]
            for i in range(B):
                rec = {
                    'mae_edos': mae_e[i].item(),
                    'mse_edos': mse_e[i].item(),
                    'r2_edos': r2_e[i].item(),
                    'mae_phdos': mae_p[i].item(),
                    'mse_phdos': mse_p[i].item(),
                    'r2_phdos': r2_p[i].item(),
                }
                if is_m5:
                    rec.update({
                        'oracle_mae_edos': mae_e_orc[i].item(),
                        'oracle_mse_edos': mse_e_orc[i].item(),
                        'oracle_r2_edos': r2_e_orc[i].item(),
                        'oracle_mae_phdos': mae_p_orc[i].item(),
                        'oracle_mse_phdos': mse_p_orc[i].item(),
                        'oracle_r2_phdos': r2_p_orc[i].item(),
                    })
                records.append(rec)

    df = pd.DataFrame(records)
    summary = {
        'mae_edos_mean': float(df['mae_edos'].mean()),
        'mae_edos_std': float(df['mae_edos'].std()),
        'mae_edos_median': float(df['mae_edos'].median()),
        'mse_edos_mean': float(df['mse_edos'].mean()),
        'mse_edos_std': float(df['mse_edos'].std()),
        'r2_edos_mean': float(df['r2_edos'].mean()),
        'r2_edos_std': float(df['r2_edos'].std()),
        'r2_edos_median': float(df['r2_edos'].median()),
        'fail_rate_edos': float((df['r2_edos'] < 0).mean() * 100.0),

        'mae_phdos_mean': float(df['mae_phdos'].mean()),
        'mae_phdos_std': float(df['mae_phdos'].std()),
        'mae_phdos_median': float(df['mae_phdos'].median()),
        'mse_phdos_mean': float(df['mse_phdos'].mean()),
        'mse_phdos_std': float(df['mse_phdos'].std()),
        'r2_phdos_mean': float(df['r2_phdos'].mean()),
        'r2_phdos_std': float(df['r2_phdos'].std()),
        'r2_phdos_median': float(df['r2_phdos'].median()),
        'fail_rate_phdos': float((df['r2_phdos'] < 0).mean() * 100.0),
    }

    if is_m5:
        summary.update({
            'oracle_mae_edos_mean': float(df['oracle_mae_edos'].mean()),
            'oracle_mae_edos_median': float(df['oracle_mae_edos'].median()),
            'oracle_r2_edos_mean': float(df['oracle_r2_edos'].mean()),
            'oracle_r2_edos_median': float(df['oracle_r2_edos'].median()),
            'oracle_mae_phdos_mean': float(df['oracle_mae_phdos'].mean()),
            'oracle_mae_phdos_median': float(df['oracle_mae_phdos'].median()),
            'oracle_r2_phdos_mean': float(df['oracle_r2_phdos'].mean()),
            'oracle_r2_phdos_median': float(df['oracle_r2_phdos'].median()),
        })

    if return_sample_level:
        from thermo_props import ThermodynamicCalculator
        calc = ThermodynamicCalculator()
        p_p_arr = np.concatenate(all_p_p, axis=0)
        t_p_arr = np.concatenate(all_t_p, axis=0)
        p_e_arr = np.concatenate(all_p_e, axis=0)
        t_e_arr = np.concatenate(all_t_e, axis=0)

        cv_preds, cv_tgts = [], []
        debye_preds, debye_tgts = [], []
        for i in range(len(p_p_arr)):
            d_p = calc.compute_Debye_T(p_p_arr[i])
            d_t = calc.compute_Debye_T(t_p_arr[i])
            debye_preds.append(d_p)
            debye_tgts.append(d_t)
            cv_p = calc.compute_Cv(p_p_arr[i], T_range=[300.0])[0]
            cv_t = calc.compute_Cv(t_p_arr[i], T_range=[300.0])[0]
            cv_preds.append(cv_p)
            cv_tgts.append(cv_t)

        df['cv_pred'] = cv_preds
        df['cv_true'] = cv_tgts
        df['debye_pred'] = debye_preds
        df['debye_true'] = debye_tgts

        valid_debye = (np.array(debye_tgts) > 50) & (np.array(debye_preds) > 50)
        if np.sum(valid_debye) > 0:
            cv_mae = float(np.mean(np.abs(np.array(cv_preds)[valid_debye] - np.array(cv_tgts)[valid_debye])))
        else:
            cv_mae = float(np.mean(np.abs(np.array(cv_preds) - np.array(cv_tgts))))
        summary['cv_mae'] = cv_mae
        summary['sample_df'] = df
        summary['pred_phdos'] = p_p_arr
        summary['pred_edos'] = p_e_arr
        summary['tgt_phdos'] = t_p_arr
        summary['tgt_edos'] = t_e_arr

    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uniARPAT Ablation Experiments Runner')
    parser.add_argument('--model', type=str, default='M1', choices=['M1', 'M2', 'M3', 'M4', 'M5', 'all'], help='Model variant to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--skip_existing', action='store_true', help='Skip variant if test summary already exists')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (H1 hygiene, recorded in history CSV)')
    args = parser.parse_args()

    if args.model == 'all':
        for m in ['M1', 'M2', 'M3', 'M4', 'M5']:
            train_and_eval(m, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, skip_existing=args.skip_existing, seed=args.seed)
    else:
        train_and_eval(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, skip_existing=args.skip_existing, seed=args.seed)
