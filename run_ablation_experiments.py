import os
import time
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from utils.builder import ConfigBuilder
from model.model import basemodel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('ablation')

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

def train_and_eval(model_name: str, epochs: int = 100, batch_size: int = 32, lr: float = 5e-5):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config_info = MODEL_CONFIGS[model_name]
    logger.info("=" * 70)
    logger.info(f"   STARTING ABLATION EXPERIMENT: {model_name}")
    logger.info(f"   Description: {config_info['desc']}")
    logger.info(f"   Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = float('inf')
    best_epoch = 0
    history = []

    for epoch in range(epochs):
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
        torch.cuda.synchronize()
        ep_time = time.time() - t_start
        train_loss /= n_batches
        avg_sub_losses = {f"train_{k}": v / n_batches for k, v in sub_loss_accum.items()}

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
            'epoch_time_s': ep_time,
            'alpha_e': alpha_e,
            'alpha_p': alpha_p,
            **avg_sub_losses,
            **val_metrics,
            'balanced_score': balanced
        })

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

            # Sample-wise primary metrics
            mae_e = torch.mean(torch.abs(p_e - t_e), dim=-1)
            mse_e = torch.mean((p_e - t_e)**2, dim=-1)
            ss_res_e = torch.sum((t_e - p_e)**2, dim=-1)
            ss_tot_e = torch.sum((t_e - torch.mean(t_e, dim=-1, keepdim=True))**2, dim=-1)
            r2_e = 1.0 - (ss_res_e / (ss_tot_e + 1e-8))

            mae_p = torch.mean(torch.abs(p_p - t_p), dim=-1)
            mse_p = torch.mean((p_p - t_p)**2, dim=-1)
            ss_res_p = torch.sum((t_p - p_p)**2, dim=-1)
            ss_tot_p = torch.sum((t_p - torch.mean(t_p, dim=-1, keepdim=True))**2, dim=-1)
            r2_p = 1.0 - (ss_res_p / (ss_tot_p + 1e-8))

            # Sample-wise Oracle metrics if M5
            if is_m5:
                mae_e_orc = torch.mean(torch.abs(p_e_orc - t_e), dim=-1)
                mse_e_orc = torch.mean((p_e_orc - t_e)**2, dim=-1)
                ss_res_e_orc = torch.sum((t_e - p_e_orc)**2, dim=-1)
                r2_e_orc = 1.0 - (ss_res_e_orc / (ss_tot_e + 1e-8))

                mae_p_orc = torch.mean(torch.abs(p_p_orc - t_p), dim=-1)
                mse_p_orc = torch.mean((p_p_orc - t_p)**2, dim=-1)
                ss_res_p_orc = torch.sum((t_p - p_p_orc)**2, dim=-1)
                r2_p_orc = 1.0 - (ss_res_p_orc / (ss_tot_p + 1e-8))

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
        summary['sample_df'] = df

    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uniARPAT Ablation Experiments Runner')
    parser.add_argument('--model', type=str, default='M1', choices=['M1', 'M2', 'M3', 'M4', 'M5', 'all'], help='Model variant to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()

    if args.model == 'all':
        for m in ['M1', 'M2', 'M3', 'M4', 'M5']:
            train_and_eval(m, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        train_and_eval(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
