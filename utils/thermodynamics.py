"""Vibrational thermodynamic calculations used by current regression tests."""

import numpy as np

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
