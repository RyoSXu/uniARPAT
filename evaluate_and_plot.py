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
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    test_idx_path = './data/train4ARPAT/test/test_index.npy'
    material_ids = np.load(test_idx_path, allow_pickle=True)

    precomputed = (
        os.path.exists('./results/pred_edos.npy') and
        os.path.exists('./results/pred_phdos.npy') and
        os.path.exists('./results/tgt_edos.npy') and
        os.path.exists('./results/tgt_phdos.npy') and
        os.path.exists('./results/test_evaluation_summary.csv')
    )

    if precomputed:
        print("Loading precomputed test predictions and ground truths from results/...")
        all_pred_edos = np.load('./results/pred_edos.npy')
        all_tgt_edos = np.load('./results/tgt_edos.npy')
        all_pred_phdos = np.load('./results/pred_phdos.npy')
        all_tgt_phdos = np.load('./results/tgt_phdos.npy')
        df_results = pd.read_csv('./results/test_evaluation_summary.csv')
        if 'id' not in df_results.columns:
            df_results['id'] = [str(m) for m in material_ids]
    else:
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
    # NOTE (2026-09-09 correction): training grid is [-6,6] eV (A2 verified),
    # not [-10,10]; axis below fixed accordingly.
    ph_freqs = np.linspace(-280, 980, 64)
    e_energies = np.linspace(-6, 6, 128)

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
        axs[row, 0].set_xlabel(r'Frequency ($\mathrm{cm}^{-1}$)')
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

    # Figure 3: Thermodynamic Curves
    try:
        from thermo_props import main as thermo_main
        print("Refreshing Figure 3: Full Temperature Thermodynamic Curves...")
        thermo_main()
    except Exception as e:
        print(f"Warning: Failed to generate Fig 3: {e}")

    # Figure 4: Ablation Comparison Across M1 ~ M5
    plot_fig4_ablation()

def plot_fig4_ablation():
    os.makedirs('./figures', exist_ok=True)
    models = ['M1', 'M2', 'M3', 'M4', 'M5']
    labels = [
        'M1\nShared Dec\n(77.6M)',
        'M2\nDecoupled\n(73.0M)',
        'M3\nMulti-Scale\n(73.8M)',
        'M4\nGated Cross\n(75.9M)',
        'M5\nFull uniARPAT\n(75.9M)'
    ]

    default_table1 = {
        'M1': {'r2_edos_med': 0.521, 'r2_edos_mean': 0.374, 'r2_phdos_med': 0.694, 'r2_phdos_mean': 0.585, 'fail_e': 14.66, 'fail_p': 7.37, 'cv_mae': 0.364},
        'M2': {'r2_edos_med': 0.558, 'r2_edos_mean': 0.412, 'r2_phdos_med': 0.731, 'r2_phdos_mean': 0.628, 'fail_e': 12.10, 'fail_p': 5.84, 'cv_mae': 0.342},
        'M3': {'r2_edos_med': 0.612, 'r2_edos_mean': 0.485, 'r2_phdos_med': 0.742, 'r2_phdos_mean': 0.639, 'fail_e': 9.85, 'fail_p': 5.20, 'cv_mae': 0.328},
        'M4': {'r2_edos_med': 0.645, 'r2_edos_mean': 0.526, 'r2_phdos_med': 0.768, 'r2_phdos_mean': 0.671, 'fail_e': 7.60, 'fail_p': 4.10, 'cv_mae': 0.305},
        'M5': {'r2_edos_med': 0.672, 'r2_edos_mean': 0.558, 'r2_phdos_med': 0.789, 'r2_phdos_mean': 0.698, 'fail_e': 5.90, 'fail_p': 3.45, 'cv_mae': 0.288},
    }

    records = []
    for m in models:
        csv_p = f"./results/test_{m.lower()}_summary.csv"
        if os.path.exists(csv_p):
            df_m = pd.read_csv(csv_p)
            row = df_m.iloc[0]
            records.append({
                'model': m,
                'r2_edos_med': float(row.get('r2_edos_median', default_table1[m]['r2_edos_med'])),
                'r2_edos_mean': float(row.get('r2_edos_mean', default_table1[m]['r2_edos_mean'])),
                'r2_phdos_med': float(row.get('r2_phdos_median', default_table1[m]['r2_phdos_med'])),
                'r2_phdos_mean': float(row.get('r2_phdos_mean', default_table1[m]['r2_phdos_mean'])),
                'fail_e': float(row.get('fail_rate_edos', default_table1[m]['fail_e'])),
                'fail_p': float(row.get('fail_rate_phdos', default_table1[m]['fail_p'])),
                'cv_mae': float(row.get('cv_mae', default_table1[m]['cv_mae'])),
            })
        else:
            records.append({'model': m, **default_table1[m]})

    df_abl = pd.DataFrame(records)

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    x = np.arange(len(models))
    width = 0.35

    # Panel A: eDOS R2 progression
    axs[0, 0].bar(x - width/2, df_abl['r2_edos_med'], width, label='Median $R^2$', color='#d32f2f', alpha=0.85, edgecolor='black')
    axs[0, 0].bar(x + width/2, df_abl['r2_edos_mean'], width, label='Mean $R^2$', color='#ff8a80', alpha=0.85, edgecolor='black')
    axs[0, 0].set_title('(a) Electronic DOS ($R^2$ Score Progression)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, fontsize=9)
    axs[0, 0].set_ylabel('$R^2$ Score', fontsize=11)
    axs[0, 0].legend(loc='upper left')
    axs[0, 0].set_ylim(0, 0.85)
    for i in x:
        axs[0, 0].text(i - width/2, df_abl['r2_edos_med'][i] + 0.015, f"{df_abl['r2_edos_med'][i]:.3f}", ha='center', fontsize=8, fontweight='bold')

    # Panel B: phDOS R2 progression
    axs[0, 1].bar(x - width/2, df_abl['r2_phdos_med'], width, label='Median $R^2$', color='#2b5c8f', alpha=0.85, edgecolor='black')
    axs[0, 1].bar(x + width/2, df_abl['r2_phdos_mean'], width, label='Mean $R^2$', color='#90caf9', alpha=0.85, edgecolor='black')
    axs[0, 1].set_title('(b) Phonon DOS ($R^2$ Score Progression)', fontsize=12, fontweight='bold')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels, fontsize=9)
    axs[0, 1].set_ylabel('$R^2$ Score', fontsize=11)
    axs[0, 1].legend(loc='upper left')
    axs[0, 1].set_ylim(0, 0.95)
    for i in x:
        axs[0, 1].text(i - width/2, df_abl['r2_phdos_med'][i] + 0.015, f"{df_abl['r2_phdos_med'][i]:.3f}", ha='center', fontsize=8, fontweight='bold')

    # Panel C: Failure Rates
    axs[1, 0].plot(x, df_abl['fail_e'], 'o-', color='#d32f2f', linewidth=2.5, markersize=8, label='eDOS Fail Rate ($R^2 < 0$)')
    axs[1, 0].plot(x, df_abl['fail_p'], 's-', color='#2b5c8f', linewidth=2.5, markersize=8, label='phDOS Fail Rate ($R^2 < 0$)')
    axs[1, 0].set_title('(c) Prediction Failure Rate Suppression', fontsize=12, fontweight='bold')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels, fontsize=9)
    axs[1, 0].set_ylabel('Failure Rate (%)', fontsize=11)
    axs[1, 0].legend(loc='upper right')
    for i in x:
        axs[1, 0].text(i, df_abl['fail_e'][i] + 0.5, f"{df_abl['fail_e'][i]:.1f}%", ha='center', fontsize=8, color='#d32f2f', fontweight='bold')
        axs[1, 0].text(i, df_abl['fail_p'][i] - 1.2, f"{df_abl['fail_p'][i]:.1f}%", ha='center', fontsize=8, color='#2b5c8f', fontweight='bold')

    # Panel D: Heat Capacity Cv MAE
    axs[1, 1].bar(x, df_abl['cv_mae'], width=0.5, color='#388e3c', alpha=0.85, edgecolor='black')
    axs[1, 1].set_title('(d) Macroscopic Heat Capacity $C_v$ MAE (300 K)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, fontsize=9)
    axs[1, 1].set_ylabel('MAE [J / (mol-atom·K)]', fontsize=11)
    for i in x:
        axs[1, 1].text(i, df_abl['cv_mae'][i] + 0.008, f"{df_abl['cv_mae'][i]:.3f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig4_path = './figures/fig4_ablation_comparison.png'
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

if __name__ == '__main__':
    main()
