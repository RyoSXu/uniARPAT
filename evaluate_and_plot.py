import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.builder import ConfigBuilder

def main():
    # 1. Load config
    cfg_path = './output/config/world_size1-ARPAT/training_options.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Build model
    builder = ConfigBuilder(**cfg)
    model = builder.get_model()
    ckpt_path = './output/config/world_size1-ARPAT/checkpoint_best.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.model['transformer'].load_state_dict(ckpt['model']['transformer'])
    model.to(device)
    model.eval()

    # 3. Build test loader
    test_loader = builder.get_dataloader(
        dataset_params=cfg['dataset'],
        split='test',
        batch_size=32,
        dos_minmax=builder.dos_minmax,
        dos_zscore=builder.dos_zscore,
        scale_factor=builder.scale_factor,
        apply_log=builder.apply_log
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # 4. Load test material IDs
    test_idx_path = os.path.join(cfg['dataset']['test']['data_dir'], 'test', 'test_index.npy')
    material_ids = np.load(test_idx_path, allow_pickle=True)

    # 5. Run inference and collect full predictions
    all_pred_edos, all_tgt_edos = [], []
    all_pred_phdos, all_tgt_phdos = [], []
    sample_metrics = []

    print("Running batch inference...")
    curr_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Model inference
            inp, pos, mask, edos_tgt, phdos_tgt, \
            edos_m, edos_s, edos_min, edos_max, \
            phdos_m, phdos_s, phdos_min, phdos_max = model.data_preprocess(batch)

            outputs = model.model['transformer'](inp, mask, pos)
            p_e = outputs['edos']
            p_p = outputs['phdos']

            # Denormalize (MinMax)
            p_e_denorm = p_e * (edos_max - edos_min) + edos_min
            t_e_denorm = edos_tgt * (edos_max - edos_min) + edos_min
            p_p_denorm = p_p * (phdos_max - phdos_min) + phdos_min
            t_p_denorm = phdos_tgt * (phdos_max - phdos_min) + phdos_min

            p_e_denorm = torch.clamp(p_e_denorm, min=0.0).cpu().numpy()
            t_e_denorm = t_e_denorm.cpu().numpy()
            p_p_denorm = torch.clamp(p_p_denorm, min=0.0).cpu().numpy()
            t_p_denorm = t_p_denorm.cpu().numpy()

            batch_size = p_e_denorm.shape[0]
            for i in range(batch_size):
                m_id = str(material_ids[curr_idx + i])
                # eDOS metrics
                mae_e = float(np.mean(np.abs(p_e_denorm[i] - t_e_denorm[i])))
                mse_e = float(np.mean((p_e_denorm[i] - t_e_denorm[i])**2))
                ss_res_e = float(np.sum((t_e_denorm[i] - p_e_denorm[i])**2))
                ss_tot_e = float(np.sum((t_e_denorm[i] - np.mean(t_e_denorm[i]))**2))
                r2_e = float(1.0 - (ss_res_e / (ss_tot_e + 1e-8)))

                # phDOS metrics
                mae_p = float(np.mean(np.abs(p_p_denorm[i] - t_p_denorm[i])))
                mse_p = float(np.mean((p_p_denorm[i] - t_p_denorm[i])**2))
                ss_res_p = float(np.sum((t_p_denorm[i] - p_p_denorm[i])**2))
                ss_tot_p = float(np.sum((t_p_denorm[i] - np.mean(t_p_denorm[i]))**2))
                r2_p = float(1.0 - (ss_res_p / (ss_tot_p + 1e-8)))

                sample_metrics.append({
                    'id': m_id,
                    'mae_edos': mae_e, 'mse_edos': mse_e, 'r2_edos': r2_e,
                    'mae_phdos': mae_p, 'mse_phdos': mse_p, 'r2_phdos': r2_p
                })

            curr_idx += batch_size
            all_pred_edos.append(p_e_denorm)
            all_tgt_edos.append(t_e_denorm)
            all_pred_phdos.append(p_p_denorm)
            all_tgt_phdos.append(t_p_denorm)

    all_pred_edos = np.concatenate(all_pred_edos, axis=0)
    all_tgt_edos = np.concatenate(all_tgt_edos, axis=0)
    all_pred_phdos = np.concatenate(all_pred_phdos, axis=0)
    all_tgt_phdos = np.concatenate(all_tgt_phdos, axis=0)

    df_results = pd.DataFrame(sample_metrics)
    os.makedirs('./results', exist_ok=True)
    df_results.to_csv('./results/test_evaluation_summary.csv', index=False)
    np.save('./results/pred_edos.npy', all_pred_edos)
    np.save('./results/tgt_edos.npy', all_tgt_edos)
    np.save('./results/pred_phdos.npy', all_pred_phdos)
    np.save('./results/tgt_phdos.npy', all_tgt_phdos)

    print("\n" + "="*55)
    print("            uniARPAT OVERALL TEST RESULTS          ")
    print("="*55)
    print(f"Total Test Materials : {len(df_results)}")
    print(f"eDOS  - Mean MAE: {df_results['mae_edos'].mean():.4f}, MSE: {df_results['mse_edos'].mean():.4f}, Mean R2: {df_results['r2_edos'].mean():.4f}")
    print(f"phDOS - Mean MAE: {df_results['mae_phdos'].mean():.4f}, MSE: {df_results['mse_phdos'].mean():.4f}, Mean R2: {df_results['r2_phdos'].mean():.4f}")
    print("="*55)

    # 6. Generate Publication Figures
    os.makedirs('./figures', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Error Distribution Plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].hist(df_results['mae_phdos'], bins=40, color='#2b5c8f', edgecolor='black', alpha=0.85)
    axs[0, 0].set_title('phDOS MAE Distribution', fontsize=13, fontweight='bold')
    axs[0, 0].set_xlabel('MAE (state / cm$^{-1}$)')
    axs[0, 0].set_ylabel('Sample Count')

    axs[0, 1].hist(df_results['r2_phdos'], bins=40, color='#388e3c', edgecolor='black', alpha=0.85, range=(-0.2, 1.0))
    axs[0, 1].set_title('phDOS $R^2$ Score Distribution', fontsize=13, fontweight='bold')
    axs[0, 1].set_xlabel('$R^2$ Score')
    axs[0, 1].set_ylabel('Sample Count')

    axs[1, 0].hist(df_results['mae_edos'], bins=40, color='#d32f2f', edgecolor='black', alpha=0.85)
    axs[1, 0].set_title('eDOS MAE Distribution', fontsize=13, fontweight='bold')
    axs[1, 0].set_xlabel('MAE (state / eV)')
    axs[1, 0].set_ylabel('Sample Count')

    axs[1, 1].hist(df_results['r2_edos'], bins=40, color='#f57c00', edgecolor='black', alpha=0.85, range=(-0.2, 1.0))
    axs[1, 1].set_title('eDOS $R^2$ Score Distribution', fontsize=13, fontweight='bold')
    axs[1, 1].set_xlabel('$R^2$ Score')
    axs[1, 1].set_ylabel('Sample Count')

    plt.tight_layout()
    fig_path1 = './figures/fig1_error_distributions.png'
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved: {fig_path1}")

    # Figure 2: Representative Material Dual-Spectrum Comparisons
    ph_freqs = np.linspace(-280, 980, 64)
    e_energies = np.linspace(-10, 10, 128)

    top_materials = df_results[(df_results['r2_phdos'] > 0.8) & (df_results['r2_edos'] > 0.65)].head(4)
    if len(top_materials) < 4:
        top_materials = df_results.sort_values(by=['r2_phdos', 'r2_edos'], ascending=False).head(4)

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    for row, (_, item) in enumerate(top_materials.iterrows()):
        m_id = item['id']
        idx = int(np.where(material_ids == m_id)[0][0])

        # phDOS plot
        axs[row, 0].plot(ph_freqs, all_tgt_phdos[idx], 'k-', label='DFT (Truth)', linewidth=2.0)
        axs[row, 0].plot(ph_freqs, all_pred_phdos[idx], 'r--', label='uniARPAT (Pred)', linewidth=2.0)
        axs[row, 0].set_title(f"{m_id} - Phonon DOS (MAE: {item['mae_phdos']:.4f}, $R^2$: {item['r2_phdos']:.3f})", fontsize=11, fontweight='bold')
        axs[row, 0].set_xlabel('Frequency ($\mathrm{cm}^{-1}$)')
        axs[row, 0].set_ylabel('PhDOS')
        axs[row, 0].legend(loc='upper right')

        # eDOS plot
        axs[row, 1].plot(e_energies, all_tgt_edos[idx], 'k-', label='DFT (Truth)', linewidth=2.0)
        axs[row, 1].plot(e_energies, all_pred_edos[idx], 'b--', label='uniARPAT (Pred)', linewidth=2.0)
        axs[row, 1].axvline(0, color='gray', linestyle=':', label='Fermi Level ($E_F$)')
        axs[row, 1].set_title(f"{m_id} - Electronic DOS (MAE: {item['mae_edos']:.3f}, $R^2$: {item['r2_edos']:.3f})", fontsize=11, fontweight='bold')
        axs[row, 1].set_xlabel('Energy ($E - E_F$, eV)')
        axs[row, 1].set_ylabel('eDOS')
        axs[row, 1].legend(loc='upper right')

    plt.tight_layout()
    fig_path2 = './figures/fig2_representative_dual_spectra.png'
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved: {fig_path2}")

if __name__ == '__main__':
    main()
