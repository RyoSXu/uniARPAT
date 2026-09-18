import torch
import numpy as np
from torch import nn


class EnergyCode(nn.Module):
    """C1.2: bin-center energy code (Fourier + RFF), zero-init residual add-on.

    code(x) = ZeroInitLinear([Fourier_L8(x); RFF_D64(x)]) -> d_model, added to
    the learned query. Day-0 output is exactly zero => M1 behavior preserved;
    any gain is attributable to the energy information, not a capacity shock.
    RFF (high-freq, Van Hove spikes) is eDOS-only by design; phDOS untouched.
    Frozen: L=8 bands, D=64, sigma=10.0 (periods down to ~0.1eV), seed=42.
    """

    def __init__(self, d_model, bands=8, rff_dim=64, sigma=10.0, seed=42,
                 x_range=6.0):
        super().__init__()
        self.bands = bands
        self.x_range = x_range
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("rff_B", torch.randn(rff_dim, 1, generator=g) * sigma)
        self.proj = nn.Linear(2 * bands + 2 * rff_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """x: [E] bin centers (same unit the grid was built in)."""
        xn = x / self.x_range
        feats = []
        for k in range(self.bands):
            f = (2 ** k) * np.pi * xn
            feats += [torch.sin(f).unsqueeze(-1), torch.cos(f).unsqueeze(-1)]
        r = 2 * np.pi * (x.unsqueeze(-1) @ self.rff_B.T)  # [E, D]
        feats += [torch.sin(r), torch.cos(r)]
        return self.proj(torch.cat(feats, dim=-1))  # [E, 2B+2D] -> d_model


class CoordTrunk(nn.Module):
    """E9-P0 Q1: plain coordinate MLP, zero-init residual add-on (Design-E 9).

    Trunk(x) = ZeroInitLinear(GELU(Linear(x_norm))) -> d_model, added to the
    learned decoder query. Day-0 output is exactly zero => base behavior
    preserved; any pilot delta is attributable to coordinate information.
    One trunk per task (units/zeros differ: eDOS eV@Fermi, phDOS cm^-1@nu=0);
    normalization is a fixed zero-preserving scale (no shift, no fitting).
    Fourier/RFF generalization is Q2 scope, NOT here.
    """

    def __init__(self, d_model, hidden_dim=128, x_scale=1.0):
        super().__init__()
        self.x_scale = float(x_scale)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """x: [..., E] bin centers in task physical units."""
        h = self.fc1(x.unsqueeze(-1) / self.x_scale)
        return self.proj(self.act(h))  # [..., E, d_model]


class FourierTrunk(nn.Module):
    """E9-P0 Q2: RFF coordinate trunk, zero-init residual add-on (Design-E 9).

    feats(x) = [x_norm, sin(2πBx_norm), cos(2πBx_norm)] -> MLP -> ZeroInit
    Day-0 output is exactly zero => base behavior preserved.
    Task sigmas (frozen B ~ N(0, sigma^2), seed-fixed): eDOS HIGH frequency
    (Van Hove spikes, 64 freqs sigma=8) / phDOS LOW frequency (smooth
    phonons, 32 freqs sigma=2). Inputs clip to the training range at
    inference (no-op on the frozen grid; future warp grids need it).
    Q1's plain MLP is the ablation control for the Fourier factor.
    """

    def __init__(self, d_model, hidden_dim=128, n_freq=64, sigma=8.0,
                 x_scale=1.0, seed=42, x_min=-1.0, x_max=1.0):
        super().__init__()
        self.x_scale = float(x_scale)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("freq_B", torch.randn(n_freq, generator=g) * sigma)
        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))
        self.fc1 = nn.Linear(1 + 2 * n_freq, hidden_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """x: [..., E] bin centers in task physical units."""
        xn = (x / self.x_scale).clamp(self.x_min.item(), self.x_max.item())
        ang = 2 * np.pi * xn.unsqueeze(-1) * self.freq_B  # [..., E, F]
        h = self.fc1(torch.cat(
            [xn.unsqueeze(-1), torch.sin(ang), torch.cos(ang)], dim=-1))
        return self.proj(self.act(h))  # [..., E, d_model]

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, kernel_size=3, padding=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 添加卷积层
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )

            if i < num_layers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        # 输入形状: [B, D, L]
        for layer in self.layers:
            x = layer(x)
        return x  # 输出形状: [B, output_dim, L]


class DeepConv1dHead(nn.Module):
    """
    phDOS Output Head: 3-layer deep Conv1d (512 -> 256 -> 256 -> 1)
    Total parameters: 787,969 (~0.788M)
    Precisely symmetric in capacity with MultiScaleResidualHead.
    """
    def __init__(self, d_model=512, hidden_dim=256, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=kernel_size, padding=p),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=p),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=p),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=kernel_size, padding=p)
        )

    def forward(self, x):
        # x: [B, d_model, L] -> returns [B, 1, L]
        return self.net(x)


class MultiScaleResidualHead(nn.Module):
    """
    eDOS Output Head: Multi-Scale Residual Convolutional Head
    Parallel multi-scale branches (k=3: 128ch, k=5: 64ch, k=7: 64ch -> concat 256ch)
    Residual Conv1d block (256 -> 256) + GroupNorm + GELU
    Projection Conv1d (256 -> 1)
    Total parameters: 788,225 (~0.788M)
    Precisely symmetric in capacity with DeepConv1dHead (<0.05% diff).
    """
    def __init__(self, d_model=512, hidden_dim=256):
        super().__init__()
        self.branch_a = nn.Conv1d(d_model, 128, kernel_size=3, padding=1)
        self.branch_b = nn.Conv1d(d_model, 64, kernel_size=5, padding=2)
        self.branch_c = nn.Conv1d(d_model, 64, kernel_size=7, padding=3)
        self.act = nn.GELU()
        self.res_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.out_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, d_model, L]
        h = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1) # [B, 256, L]
        h = self.act(h)
        res = self.act(self.norm(self.res_conv(h)))
        h = h + res
        return self.out_conv(h) # [B, 1, L]


class PostDecoderGatedCrossAttention(nn.Module):
    """
    Post-Decoder Zero-Initialized Gated Multi-Head Cross-Attention.
    Allows mutual latent modulation between eDOS queries and phDOS queries.
    alpha_e and alpha_p are initialized to 0.0, ensuring strict identity behavior at step 0.
    """
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn_e2p = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_p2e = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_e = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        # Zero-initialized gating scalars
        self.alpha_e = nn.Parameter(torch.zeros(1))
        self.alpha_p = nn.Parameter(torch.zeros(1))

    def forward(self, hs_edos, hs_phdos):
        # hs_edos: [B, L_e, d_model], hs_phdos: [B, L_p, d_model]
        attn_e, _ = self.cross_attn_e2p(query=hs_edos, key=hs_phdos, value=hs_phdos)
        out_edos = self.norm_e(hs_edos + self.alpha_e * attn_e)

        attn_p, _ = self.cross_attn_p2e(query=hs_phdos, key=hs_edos, value=hs_edos)
        out_phdos = self.norm_p(hs_phdos + self.alpha_p * attn_p)

        return out_edos, out_phdos


class EtaHead(nn.Module):
    """
    H1 bounded coverage head: predicts (eta_phonon, gamma_edos) in [0,1]
    from pooled crystal features. Sigmoid output; zero-init bias => day-0
    (0.5, 0.5). Supervised by windowed/total ratios (labels + Z0 sidecar).
    Params: ~74k (same skeleton as ScaleHead).
    Output: [B, 2] -> [eta_ph, gamma_e].
    """
    def __init__(self, d_model=512, hidden_dim=128, out_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.mlp[-2].bias.zero_()  # pre-sigmoid bias 0 => eta_0 = 0.5

    def forward(self, h_crystal):
        # h_crystal: [B, d_model] -> [B, 2]
        return self.mlp(h_crystal)


class ScaleHead(nn.Module):
    """
    Scale Head MLP: Predicts log-scale factors for eDOS and phDOS from pooled crystal features.
    Params: ~74,050 (~74k)
    Output: [B, 2] -> [log_scale_edos, log_scale_phdos]
    """
    def __init__(self, d_model=512, hidden_dim=128, out_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, h_crystal):
        # h_crystal: [B, d_model] -> [B, 2]
        return self.mlp(h_crystal)


def global_masked_pool(memory, mask):
    """
    Global masked average pooling over crystal atoms.
    memory: [B, L, d_model]
    mask: [B, L] bool where True indicates padded token.
    returns: [B, d_model] crystal-level latent representation
    """
    valid_weight = (~mask).float().unsqueeze(-1) # [B, L, 1]
    sum_feats = torch.sum(memory * valid_weight, dim=1) # [B, d_model]
    count = torch.sum(valid_weight, dim=1).clamp(min=1.0) # [B, 1]
    return sum_feats / count