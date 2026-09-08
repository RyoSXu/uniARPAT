import torch
from torch import nn

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