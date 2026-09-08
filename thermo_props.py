import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Physical constants
KB = 1.380649e-23        # J / K
H_PLANCK = 6.62607015e-34 # J s
C_LIGHT = 2.99792458e10  # cm / s
NA = 6.02214076e23       # Avogadro constant
R_GAS = KB * NA          # 8.314 J / (mol K)

# Conversion factor: x = h * c * freq / (k_B * T) = FACTOR * freq / T
# with freq in cm^-1, T in K
FREQ_TO_K = (H_PLANCK * C_LIGHT) / KB # ~ 1.438777 K / cm^-1

class ThermodynamicCalculator:
    def __init__(self, ph_freq_min=-280.0, ph_freq_max=980.0, ph_bins=64):
        self.freqs = np.linspace(ph_freq_min, ph_freq_max, ph_bins)
        # Filter strictly positive frequencies for vibrational thermodynamics
        self.pos_mask = self.freqs > 10.0
        self.pos_freqs = self.freqs[self.pos_mask]
        self.d_freq = self.freqs[1] - self.freqs[0]

    def compute_Cv(self, phdos, T_range=np.linspace(10, 1000, 100)):
        """
        Compute lattice heat capacity Cv(T) in J / (mol * K) per formula unit or per atom.
        Assuming normalized phdos (normalized to 1 mode distribution).
        Cv(T) = 3 * R * integral [ (x/2)^2 / sinh^2(x/2) * g(nu) dnu ]
        """
        g_pos = phdos[self.pos_mask]
        norm = np.sum(g_pos) * self.d_freq
        if norm <= 1e-8:
            return np.zeros_like(T_range)
        g_norm = g_pos / norm

        Cv_list = []
        for T in T_range:
            x = FREQ_TO_K * self.pos_freqs / T
            # avoid overflow/underflow
            x = np.clip(x, 1e-6, 100.0)
            integrand = (0.5 * x)**2 / (np.sinh(0.5 * x)**2) * g_norm
            cv = 3.0 * R_GAS * np.sum(integrand) * self.d_freq
            Cv_list.append(cv)
        return np.array(Cv_list)

    def compute_Sv(self, phdos, T_range=np.linspace(10, 1000, 100)):
        """
        Compute vibrational entropy Sv(T) in J / (mol * K)
        Sv(T) = 3 * R * integral [ (x / (e^x - 1) - ln(1 - e^-x)) * g(nu) dnu ]
        """
        g_pos = phdos[self.pos_mask]
        norm = np.sum(g_pos) * self.d_freq
        if norm <= 1e-8:
            return np.zeros_like(T_range)
        g_norm = g_pos / norm

        Sv_list = []
        for T in T_range:
            x = FREQ_TO_K * self.pos_freqs / T
            x = np.clip(x, 1e-6, 100.0)
            term1 = x / (np.exp(x) - 1.0)
            term2 = -np.log(1.0 - np.exp(-x) + 1e-12)
            integrand = (term1 + term2) * g_norm
            sv = 3.0 * R_GAS * np.sum(integrand) * self.d_freq
            Sv_list.append(sv)
        return np.array(Sv_list)

    def compute_Fvib(self, phdos, T_range=np.linspace(10, 1000, 100)):
        """
        Compute vibrational Helmholtz free energy F_vib(T) in kJ / mol
        F_vib(T) = 3 * R * T * integral [ (x / 2 + ln(1 - e^-x)) * g(nu) dnu ]
        """
        g_pos = phdos[self.pos_mask]
        norm = np.sum(g_pos) * self.d_freq
        if norm <= 1e-8:
            return np.zeros_like(T_range)
        g_norm = g_pos / norm

        F_list = []
        for T in T_range:
            x = FREQ_TO_K * self.pos_freqs / T
            x = np.clip(x, 1e-6, 100.0)
            term_zpe = 0.5 * x
            term_thermal = np.log(1.0 - np.exp(-x) + 1e-12)
            integrand = (term_zpe + term_thermal) * g_norm
            # in J/mol -> convert to kJ/mol
            f_val = (3.0 * R_GAS * T * np.sum(integrand) * self.d_freq) / 1000.0
            F_list.append(f_val)
        return np.array(F_list)

    def compute_Debye_T(self, phdos):
        """
        Estimate Debye temperature Theta_D (K) from second moment of phDOS
        Theta_D = (h*c / k_B) * sqrt(5/3 * <nu^2>)
        """
        g_pos = phdos[self.pos_mask]
        norm = np.sum(g_pos) * self.d_freq
        if norm <= 1e-8:
            return 0.0
        g_norm = g_pos / norm
        mean_nu2 = np.sum((self.pos_freqs**2) * g_norm) * self.d_freq
        theta_D = FREQ_TO_K * np.sqrt(5.0 / 3.0 * mean_nu2)
        return float(theta_D)

    def compute_Slack_kappaL(self, phdos, M_avg=50.0, volume_per_atom=20.0, n_atoms=1, T=300.0, gamma=1.5, A=3.1e-6):
        """
        Calibrated Julian-Slack equation estimate of lattice thermal conductivity kappa_L (W / (m * K)) at T=300 K.
        Standard Julian-Slack formula:
          kappa_L = A * (M_avg * Theta_D^3 * delta) / (gamma^2 * T * n_atoms^(2/3))
        Where:
          M_avg: average atomic weight in amu (g/mol)
          delta: average interatomic distance in Angstrom (volume_per_atom^(1/3))
          n_atoms: number of atoms in unit cell (n_atoms^(2/3) acoustic branch suppression factor)
          Theta_D: Debye temperature in K
          gamma: Grüneisen parameter (~1.5)
          A = 3.1e-6: standard Julian-Slack proportionality constant for amu, Angstrom, K -> W/(m*K)
        """
        theta_D = self.compute_Debye_T(phdos)
        if theta_D <= 10.0:
            return 0.0
        delta_angstrom = float(volume_per_atom)**(1.0 / 3.0) # atomic length in Angstrom
        n_factor = float(max(n_atoms, 1)) ** (2.0 / 3.0)
        kappa_L = A * (M_avg * (theta_D**3) * delta_angstrom) / ((gamma**2) * T * n_factor)
        return float(kappa_L)


def main():
    print("Loading test predictions, targets, and crystal structures...")
    pred_phdos = np.load('./results/pred_phdos.npy')
    tgt_phdos = np.load('./results/tgt_phdos.npy')
    pred_edos = np.load('./results/pred_edos.npy')
    tgt_edos = np.load('./results/tgt_edos.npy')
    df_eval = pd.read_csv('./results/test_evaluation_summary.csv')

    elem_test = np.load('data/train4ARPAT/test/elements_test.npy')
    pos_test = np.load('data/train4ARPAT/test/positions_test.npy').reshape(-1, 82, 3)

    from pymatgen.core import Element

    calc = ThermodynamicCalculator()
    T_range = np.linspace(50, 1000, 100)

    # 1. Compute thermodynamic properties for all test samples
    debye_preds, debye_tgts = [], []
    cv_300_preds, cv_300_tgts = [], []
    kappa_preds, kappa_tgts = [], []
    kappa_unif_preds, kappa_unif_tgts = [], []

    print(f"Calculating macroscopic properties across {len(df_eval)} test crystals...")
    for i in range(len(df_eval)):
        p_ph = pred_phdos[i]
        t_ph = tgt_phdos[i]

        d_p = calc.compute_Debye_T(p_ph)
        d_t = calc.compute_Debye_T(t_ph)
        debye_preds.append(d_p)
        debye_tgts.append(d_t)

        cv_p = calc.compute_Cv(p_ph, T_range=[300.0])[0]
        cv_t = calc.compute_Cv(t_ph, T_range=[300.0])[0]
        cv_300_preds.append(cv_p)
        cv_300_tgts.append(cv_t)

        # Parse crystal material-specific parameters
        atoms = elem_test[i, 2:]
        valid_atoms = atoms[(atoms > 0) & (atoms < 120)].astype(int)
        n = len(valid_atoms) if len(valid_atoms) > 0 else 1
        masses = [float(Element.from_Z(int(z)).atomic_mass) for z in valid_atoms] if len(valid_atoms) > 0 else [50.0]
        m_avg = float(np.mean(masses))

        a, b = float(pos_test[i, 0, 0]), float(pos_test[i, 0, 1])
        c = float(1.0 / pos_test[i, 0, 2]) if pos_test[i, 0, 2] > 1e-4 else 10.0
        alpha, beta, gamma = np.radians(pos_test[i, 1])
        val = 1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
        vol_cell = a * b * c * np.sqrt(max(val, 1e-6))
        v_atom = float(vol_cell / n)

        # Material-specific Julian-Slack calculation (with n^(2/3) and A=3.1e-6)
        k_p = calc.compute_Slack_kappaL(p_ph, M_avg=m_avg, volume_per_atom=v_atom, n_atoms=n, T=300.0)
        k_t = calc.compute_Slack_kappaL(t_ph, M_avg=m_avg, volume_per_atom=v_atom, n_atoms=n, T=300.0)
        kappa_preds.append(k_p)
        kappa_tgts.append(k_t)

        # Old uniform baseline for reference (unified A=3.1e-6 to isolate M, V, n effects)
        k_u_p = calc.compute_Slack_kappaL(p_ph, M_avg=50.0, volume_per_atom=20.0, n_atoms=1, T=300.0, A=3.1e-6)
        k_u_t = calc.compute_Slack_kappaL(t_ph, M_avg=50.0, volume_per_atom=20.0, n_atoms=1, T=300.0, A=3.1e-6)
        kappa_unif_preds.append(k_u_p)
        kappa_unif_tgts.append(k_u_t)

    df_eval['Debye_T_pred'] = debye_preds
    df_eval['Debye_T_true'] = debye_tgts
    df_eval['Cv_300K_pred'] = cv_300_preds
    df_eval['Cv_300K_true'] = cv_300_tgts
    df_eval['kappaL_300K_pred'] = kappa_preds
    df_eval['kappaL_300K_true'] = kappa_tgts
    df_eval['kappaL_300K_unif_pred'] = kappa_unif_preds
    df_eval['kappaL_300K_unif_true'] = kappa_unif_tgts

    df_eval.to_csv('./results/test_thermo_properties.csv', index=False)

    # Print summary metrics for macroscopic properties
    valid_debye = (df_eval['Debye_T_true'] > 50) & (df_eval['Debye_T_pred'] > 50)
    debye_mae = np.mean(np.abs(df_eval.loc[valid_debye, 'Debye_T_pred'] - df_eval.loc[valid_debye, 'Debye_T_true']))
    cv_mae = np.mean(np.abs(df_eval.loc[valid_debye, 'Cv_300K_pred'] - df_eval.loc[valid_debye, 'Cv_300K_true']))

    k_p_arr = np.array(kappa_preds)
    k_t_arr = np.array(kappa_tgts)
    k_u_p_arr = np.array(kappa_unif_preds)
    k_u_t_arr = np.array(kappa_unif_tgts)

    print("\n" + "="*65)
    print("      MACROSCOPIC THERMODYNAMIC PREDICTION RESULTS     ")
    print("="*65)
    print(f"Debye Temperature Theta_D  - MAE: {debye_mae:.2f} K")
    print(f"Heat Capacity Cv at 300K   - MAE: {cv_mae:.2f} J/(mol-atom*K) [per-atom]")
    print(f"--- Material-Specific Julian-Slack kappa_L (A=3.1e-6, n^(2/3)) ---")
    print(f"Pred kappa_L: Mean = {np.mean(k_p_arr):.2f}, Median = {np.median(k_p_arr):.2f}, Std = {np.std(k_p_arr):.2f} W/(m*K)")
    print(f"True kappa_L: Mean = {np.mean(k_t_arr):.2f}, Median = {np.median(k_t_arr):.2f}, Std = {np.std(k_t_arr):.2f} W/(m*K)")
    print(f"Pred kappa_L < 5 W/(m*K): {(k_p_arr < 5).sum()}/{len(k_p_arr)} ({(k_p_arr < 5).mean()*100:.2f}%)")
    print(f"True kappa_L < 5 W/(m*K): {(k_t_arr < 5).sum()}/{len(k_t_arr)} ({(k_t_arr < 5).mean()*100:.2f}%)")
    print(f"--- Uniform Baseline Reference (M=50, V=20, no n^(2/3)) ---")
    print(f"Unif Pred kappa_L mean: {np.mean(k_u_p_arr):.2f}, <5 W/(m*K): {(k_u_p_arr < 5).sum()} ({(k_u_p_arr < 5).mean()*100:.2f}%)")
    print(f"Unif True kappa_L mean: {np.mean(k_u_t_arr):.2f}, <5 W/(m*K): {(k_u_t_arr < 5).sum()} ({(k_u_t_arr < 5).mean()*100:.2f}%)")
    print("="*65)

    # 2. Generate Figure 3: Full Temperature Thermodynamic Curves
    sample_indices = [0, 5, 12, 25] # 4 distinct materials
    fig, axs = plt.subplots(len(sample_indices), 3, figsize=(16, 12))

    for row, s_idx in enumerate(sample_indices):
        m_id = df_eval.iloc[s_idx]['id']
        p_ph = pred_phdos[s_idx]
        t_ph = tgt_phdos[s_idx]

        # Heat capacity Cv(T)
        cv_p = calc.compute_Cv(p_ph, T_range)
        cv_t = calc.compute_Cv(t_ph, T_range)
        axs[row, 0].plot(T_range, cv_t, 'k-', linewidth=2, label='DFT Ground Truth')
        axs[row, 0].plot(T_range, cv_p, 'r--', linewidth=2, label='uniARPAT Predicted')
        axs[row, 0].set_title(f"{m_id}: Heat Capacity $C_v(T)$", fontweight='bold')
        axs[row, 0].set_xlabel('Temperature (K)')
        axs[row, 0].set_ylabel('$C_v$ (J / mol K)')
        if row == 0:
            axs[row, 0].legend()

        # Vibrational Entropy Sv(T)
        sv_p = calc.compute_Sv(p_ph, T_range)
        sv_t = calc.compute_Sv(t_ph, T_range)
        axs[row, 1].plot(T_range, sv_t, 'k-', linewidth=2, label='DFT Ground Truth')
        axs[row, 1].plot(T_range, sv_p, 'g--', linewidth=2, label='uniARPAT Predicted')
        axs[row, 1].set_title(f"{m_id}: Vibrational Entropy $S_v(T)$", fontweight='bold')
        axs[row, 1].set_xlabel('Temperature (K)')
        axs[row, 1].set_ylabel('$S_v$ (J / mol K)')

        # Helmholtz Free Energy Fvib(T)
        f_p = calc.compute_Fvib(p_ph, T_range)
        f_t = calc.compute_Fvib(t_ph, T_range)
        axs[row, 2].plot(T_range, f_t, 'k-', linewidth=2, label='DFT Ground Truth')
        axs[row, 2].plot(T_range, f_p, 'm--', linewidth=2, label='uniARPAT Predicted')
        axs[row, 2].set_title(f"{m_id}: Free Energy $F_{{vib}}(T)$", fontweight='bold')
        axs[row, 2].set_xlabel('Temperature (K)')
        axs[row, 2].set_ylabel('$F_{{vib}}$ (kJ / mol)')

    plt.tight_layout()
    fig_path3 = './figures/fig3_thermodynamic_curves.png'
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Saved: {fig_path3}")

if __name__ == '__main__':
    main()
