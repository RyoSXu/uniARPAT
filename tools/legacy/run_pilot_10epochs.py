import os
import time
import random
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from utils.builder import ConfigBuilder
from model.model import basemodel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('pilot')

PILOT_SEED = int(os.environ.get("PILOT_SEED", "42"))


def setup_pilot_seed(seed: int = PILOT_SEED):
    """H1 hygiene: same seeding contract as run_ablation_experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"[pilot] seed={seed}")

def main():
    setup_pilot_seed()
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./output/pilot_m5', exist_ok=True)

    with open('configs/default.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Configure M5 architecture
    cfg['model']['params']['sub_model']['transformer'].update({
        'decoupled_decoder': True,
        'use_gated_cross_attn': True,
        'head_type': 'symmetric',
        'predict_scale': True
    })
    cfg['model']['params']['dos_minmax'] = True
    cfg['model']['params']['save_best'] = 'balanced_score'
    cfg['dataset']['train']['data_dir'] = './data/train4ARPAT'
    cfg['dataset']['valid']['data_dir'] = './data/train4ARPAT'
    cfg['dataset']['test']['data_dir'] = './data/train4ARPAT'

    builder = ConfigBuilder(**cfg)

    logger.info("Building dataloaders for Pilot Experiment...")
    train_loader = builder.get_dataloader(split='train', dos_minmax=True, batch_size=32)
    val_loader = builder.get_dataloader(split='valid', dos_minmax=True, batch_size=32)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    logger.info("Building M5 Model...")
    model = builder.get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer & Scheduler (H4 hygiene: same warmup+cosine as ablation/train)
    from utils.builder import build_warmup_cosine_scheduler
    optimizer = model.optimizer['transformer']
    epochs = 10
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs)

    pilot_records = []
    torch.cuda.reset_peak_memory_stats()

    logger.info("=" * 65)
    logger.info(f"   STARTING 10-EPOCH PILOT EXPERIMENT (M5 Full Architecture)   ")
    logger.info("=" * 65)

    for epoch in range(epochs):
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.model['transformer'].train()
        train_loss, train_edos_loss, train_phdos_loss = 0.0, 0.0, 0.0
        n_batches = len(train_loader)

        t_epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            loss_dict = model.train_one_step(batch, step=step)
            train_loss += loss_dict['loss']
            train_edos_loss += loss_dict['loss_edos']
            train_phdos_loss += loss_dict['loss_phdos']
            if (step + 1) % 100 == 0 or (step + 1) == n_batches:
                logger.info(f"Epoch [{epoch+1:02d}/10] Step [{step+1:03d}/{n_batches:03d}] | Batch Loss: {loss_dict['loss']:.4f} (eDOS: {loss_dict['loss_edos']:.4f}, phDOS: {loss_dict['loss_phdos']:.4f})")

        scheduler.step()
        torch.cuda.synchronize()
        epoch_time = time.time() - t_epoch_start

        train_loss /= n_batches
        train_edos_loss /= n_batches
        train_phdos_loss /= n_batches

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Validation Evaluation
        model.model['transformer'].eval()
        val_loss_dict = {'MAE_e': [], 'MSE_e': [], 'R2_e': [], 'MAE_p': [], 'MSE_p': [], 'R2_p': []}
        oracle_dict = {'MAE_e': [], 'R2_e': [], 'MAE_p': [], 'R2_p': []}

        with torch.no_grad():
            for batch in val_loader:
                inp, pos, mask, edos_tgt, phdos_tgt, \
                edos_m, edos_s, edos_min, edos_max, \
                phdos_m, phdos_s, phdos_min, phdos_max, \
                edos_cov, phdos_cov, _nvalence, edos_x, phdos_x = model.data_preprocess(batch)

                outputs = model.model['transformer'](inp, mask, pos, edos_x, phdos_x)

                # 1. Blind Physical Prediction (M5)
                p_e_blind = outputs['phys_edos']
                p_p_blind = outputs['phys_phdos']

                # True Physical Targets
                t_e_phys = edos_tgt * (edos_max - edos_min) + edos_min
                t_p_phys = phdos_tgt * (phdos_max - phdos_min) + phdos_min

                # 2. Oracle Denormalization (using ground-truth scale)
                p_e_oracle = outputs['shape_edos'] * (edos_max - edos_min) + edos_min
                p_p_oracle = outputs['shape_phdos'] * (phdos_max - phdos_min) + phdos_min

                # Compute per-sample blind metrics
                mae_e = torch.mean(torch.abs(p_e_blind - t_e_phys), dim=-1)
                mse_e = torch.mean((p_e_blind - t_e_phys)**2, dim=-1)
                ss_res_e = torch.sum((t_e_phys - p_e_blind)**2, dim=-1)
                ss_tot_e = torch.sum((t_e_phys - torch.mean(t_e_phys, dim=-1, keepdim=True))**2, dim=-1)
                r2_e = 1.0 - (ss_res_e / (ss_tot_e + 1e-8))

                mae_p = torch.mean(torch.abs(p_p_blind - t_p_phys), dim=-1)
                mse_p = torch.mean((p_p_blind - t_p_phys)**2, dim=-1)
                ss_res_p = torch.sum((t_p_phys - p_p_blind)**2, dim=-1)
                ss_tot_p = torch.sum((t_p_phys - torch.mean(t_p_phys, dim=-1, keepdim=True))**2, dim=-1)
                r2_p = 1.0 - (ss_res_p / (ss_tot_p + 1e-8))

                val_loss_dict['MAE_e'].extend(mae_e.cpu().numpy())
                val_loss_dict['MSE_e'].extend(mse_e.cpu().numpy())
                val_loss_dict['R2_e'].extend(r2_e.cpu().numpy())
                val_loss_dict['MAE_p'].extend(mae_p.cpu().numpy())
                val_loss_dict['MSE_p'].extend(mse_p.cpu().numpy())
                val_loss_dict['R2_p'].extend(r2_p.cpu().numpy())

                # Oracle metrics
                mae_e_orc = torch.mean(torch.abs(p_e_oracle - t_e_phys), dim=-1)
                ss_res_e_orc = torch.sum((t_e_phys - p_e_oracle)**2, dim=-1)
                r2_e_orc = 1.0 - (ss_res_e_orc / (ss_tot_e + 1e-8))
                mae_p_orc = torch.mean(torch.abs(p_p_oracle - t_p_phys), dim=-1)
                ss_res_p_orc = torch.sum((t_p_phys - p_p_oracle)**2, dim=-1)
                r2_p_orc = 1.0 - (ss_res_p_orc / (ss_tot_p + 1e-8))

                oracle_dict['MAE_e'].extend(mae_e_orc.cpu().numpy())
                oracle_dict['R2_e'].extend(r2_e_orc.cpu().numpy())
                oracle_dict['MAE_p'].extend(mae_p_orc.cpu().numpy())
                oracle_dict['R2_p'].extend(r2_p_orc.cpu().numpy())

        # Summarize epoch metrics
        rec = {
            'epoch': epoch + 1,
            'seed': PILOT_SEED,
            'train_loss': train_loss,
            'train_edos_loss': train_edos_loss,
            'train_phdos_loss': train_phdos_loss,
            'epoch_time_s': epoch_time,
            'peak_vram_mb': peak_vram_mb,
            'blind_mae_edos_mean': float(np.mean(val_loss_dict['MAE_e'])),
            'blind_mae_edos_median': float(np.median(val_loss_dict['MAE_e'])),
            'blind_r2_edos_mean': float(np.mean(val_loss_dict['R2_e'])),
            'blind_r2_edos_median': float(np.median(val_loss_dict['R2_e'])),
            'blind_mae_phdos_mean': float(np.mean(val_loss_dict['MAE_p'])),
            'blind_mae_phdos_median': float(np.median(val_loss_dict['MAE_p'])),
            'blind_r2_phdos_mean': float(np.mean(val_loss_dict['R2_p'])),
            'blind_r2_phdos_median': float(np.median(val_loss_dict['R2_p'])),
            'oracle_mae_edos_median': float(np.median(oracle_dict['MAE_e'])),
            'oracle_r2_edos_median': float(np.median(oracle_dict['R2_e'])),
            'oracle_mae_phdos_median': float(np.median(oracle_dict['MAE_p'])),
            'oracle_r2_phdos_median': float(np.median(oracle_dict['R2_p'])),
        }
        pilot_records.append(rec)

        logger.info(
            f"Epoch [{epoch+1:02d}/10] | Time: {epoch_time:.1f}s | VRAM: {peak_vram_mb:.0f}MB | "
            f"Loss: {train_loss:.4f} | "
            f"Blind eDOS R2(med): {rec['blind_r2_edos_median']:.3f} (mean: {rec['blind_r2_edos_mean']:.3f}) | "
            f"Blind phDOS R2(med): {rec['blind_r2_phdos_median']:.3f} (mean: {rec['blind_r2_phdos_mean']:.3f}) | "
            f"Oracle eDOS R2(med): {rec['oracle_r2_edos_median']:.3f} | "
            f"Oracle phDOS R2(med): {rec['oracle_r2_phdos_median']:.3f}"
        )

    # Save results
    df_pilot = pd.DataFrame(pilot_records)
    csv_path = './results/pilot_10epochs_metrics.csv'
    df_pilot.to_csv(csv_path, index=False)
    logger.info(f"Saved pilot metrics to {csv_path}")

    ckpt_path = './output/pilot_m5/pilot_m5_epoch10.pth'
    torch.save(model.model['transformer'].state_dict(), ckpt_path)
    logger.info(f"Saved pilot model checkpoint to {ckpt_path}")

    # Plot Figure: Pilot convergence and Blind vs Oracle progression
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(df_pilot['epoch'], df_pilot['train_loss'], 'o-', color='#1f77b4', linewidth=2, label='Total Loss')
    plt.plot(df_pilot['epoch'], df_pilot['train_edos_loss'], 's--', color='#ff7f0e', label='eDOS Branch')
    plt.plot(df_pilot['epoch'], df_pilot['train_phdos_loss'], '^--', color='#2ca02c', label='phDOS Branch')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Composite Physical Loss', fontweight='bold')
    plt.title('Training Loss Convergence (10 Epochs)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(df_pilot['epoch'], df_pilot['blind_r2_edos_median'], 'o-', color='#e377c2', linewidth=2, label='Blind eDOS (Median $R^2$)')
    plt.plot(df_pilot['epoch'], df_pilot['oracle_r2_edos_median'], 's--', color='#9467bd', linewidth=2, label='Oracle eDOS (Median $R^2$)')
    plt.axhline(0.521, color='gray', linestyle=':', label='M1 Baseline (0.521)')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Median $R^2$', fontweight='bold')
    plt.title('eDOS: Blind vs Oracle Progression', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(df_pilot['epoch'], df_pilot['blind_r2_phdos_median'], 'o-', color='#17becf', linewidth=2, label='Blind phDOS (Median $R^2$)')
    plt.plot(df_pilot['epoch'], df_pilot['oracle_r2_phdos_median'], 's--', color='#bcbd22', linewidth=2, label='Oracle phDOS (Median $R^2$)')
    plt.axhline(0.694, color='gray', linestyle=':', label='M1 Baseline (0.694)')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Median $R^2$', fontweight='bold')
    plt.title('phDOS: Blind vs Oracle Progression', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    fig_path = './figures/fig_pilot_10epochs.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved pilot figure to {fig_path}")

    # Summary Output
    mean_epoch_time = df_pilot['epoch_time_s'].mean()
    max_vram = df_pilot['peak_vram_mb'].max()
    logger.info("\n" + "=" * 65)
    logger.info("             PILOT EXPERIMENT EXECUTIVE SUMMARY            ")
    logger.info("=" * 65)
    logger.info(f"1. Wall-Clock Speed: Mean {mean_epoch_time:.2f} s/epoch ({mean_epoch_time/60:.2f} min/epoch)")
    logger.info(f"   Projected 100-Epoch Full Training Time: {mean_epoch_time * 100 / 3600:.2f} hours (~{mean_epoch_time * 100 / 3600:.1f}h on single V100)")
    logger.info(f"2. Peak VRAM Consumption: {max_vram:.1f} MB ({max_vram/1024:.2f} GB / 32 GB, ~{max_vram/32768*100:.1f}% capacity)")
    logger.info(f"3. Final Epoch 10 Convergence:")
    logger.info(f"   - Total Loss: {df_pilot['train_loss'].iloc[0]:.4f} -> {df_pilot['train_loss'].iloc[-1]:.4f} ({(1 - df_pilot['train_loss'].iloc[-1]/df_pilot['train_loss'].iloc[0])*100:.1f}% drop)")
    logger.info(f"   - Blind eDOS Median R2: {df_pilot['blind_r2_edos_median'].iloc[-1]:.3f} (Mean: {df_pilot['blind_r2_edos_mean'].iloc[-1]:.3f})")
    logger.info(f"   - Blind phDOS Median R2: {df_pilot['blind_r2_phdos_median'].iloc[-1]:.3f} (Mean: {df_pilot['blind_r2_phdos_mean'].iloc[-1]:.3f})")
    logger.info(f"   - Oracle eDOS Median R2: {df_pilot['oracle_r2_edos_median'].iloc[-1]:.3f}")
    logger.info(f"   - Oracle phDOS Median R2: {df_pilot['oracle_r2_phdos_median'].iloc[-1]:.3f}")
    logger.info("=" * 65)

if __name__ == '__main__':
    main()
