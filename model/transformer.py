import copy
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.heads import (
    CNN, CoordQueryGenerator, CoordTrunk, DeepConv1dHead, EnergyCode, EtaHead, FourierTrunk,
    MultiScaleResidualHead, PointwiseMLPHead, PostDecoderGatedCrossAttention,
    ScaleHead, global_masked_pool
)
from utils.atom_feature import AtomFeatureEncoder
from utils.macro_lattice import macro_lattice_features, raw_atomic_mass_table
from utils.relative_features import compute_relative_features
from utils.rp_encoding import RPEncoding


def safe_shape_norm(z: Tensor) -> Tensor:
    """
    Self-healing shape normalization:
    1. Primary: ReLU(z) / (max(ReLU(z)) + 1e-6) guarantees strict non-negativity and preserves exact zero bandgaps.
    2. Fallback: If max(ReLU(z)) <= 1e-4 (e.g. dying ReLU or all-negative pre-activations), smoothly falls back to
       Softplus(z) / (max(Softplus(z)) + 1e-6), which provides non-zero gradients to resurrect the representations.
    """
    pos = F.relu(z)
    max_val = torch.max(pos, dim=-1, keepdim=True).values
    fallback = F.softplus(z)
    fallback_max = torch.max(fallback, dim=-1, keepdim=True).values
    return torch.where(max_val > 1e-4, pos / (max_val + 1e-6), fallback / (fallback_max + 1e-6))


class PeriodicEdgeMessage(nn.Module):
    """G2a radial edge-conditioned Value residual (one instance per encoder layer).

    Implements the pre-registered G2a module only: 64-center Gaussian RBF over
    the multi-image distance, sigmoid gating on the sender Value, quintic
    smooth cutoff, ``1/sqrt(max(1, indegree))`` aggregation, output projection
    and a zero-initialized per-layer scalar ``alpha``. No direction, no top-k,
    no dense ``[B,L,L,S]`` tensor; flat sparse edges aggregated with
    ``index_add``. With ``alpha == 0`` the forward is exactly the identity.
    """

    def __init__(self, d_model=512, rbf_num=64, r_cut=5.5):
        super().__init__()
        self.d_model = int(d_model)
        self.rbf_num = int(rbf_num)
        self.r_cut = float(r_cut)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(rbf_num, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.zeros(()))
        centers = torch.linspace(0.01 * self.r_cut, 0.99 * self.r_cut, self.rbf_num)
        self.register_buffer("rbf_centers", centers)
        self.register_buffer("rbf_width", centers[1] - centers[0])

    def forward(self, h, edge_batch, edge_dst, edge_src, edge_dist):
        # h: [B, L, d]; edges: [E] (empty allowed).
        if edge_batch.numel() == 0:
            return h
        B, L, D = h.shape
        dist = edge_dist.to(device=h.device, dtype=h.dtype)
        centers = self.rbf_centers.to(device=h.device, dtype=h.dtype)
        width = self.rbf_width.to(device=h.device, dtype=h.dtype)
        phi = torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))
        gate = torch.sigmoid(self.W_g(phi))  # [E, D]
        v = self.W_v(h)  # [B, L, D]
        v_src = v[edge_batch, edge_src]  # [E, D]
        x = (dist / self.r_cut).clamp(0.0, 1.0)
        cut = (1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5).clamp(0.0, 1.0)
        m = cut.unsqueeze(-1) * (v_src * gate)  # [E, D]
        lin = edge_batch * L + edge_dst  # [E]
        agg_flat = torch.zeros(B * L, D, device=h.device, dtype=h.dtype)
        agg_flat.index_add_(0, lin, m)
        agg = agg_flat.view(B, L, D)
        deg_flat = torch.zeros(B * L, device=h.device, dtype=h.dtype)
        deg_flat.index_add_(0, lin, torch.ones_like(dist))
        deg = deg_flat.view(B, L)
        agg = agg * (1.0 / torch.sqrt(torch.clamp(deg, min=1.0))).unsqueeze(-1)
        return h + self.alpha * self.W_o(agg)


class Transformer(nn.Module):

    def __init__(self, token_num=118, d_model=512, nhead=8, edos_num=128, phdos_num=64, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False,
                 decoupled_decoder=False, use_gated_cross_attn=False,
                 head_type="legacy", predict_scale=False, atom_feat_mode="legacy3",
                  energy_code="none", edos_grid=None, scale_mode="none",
                  scalar_mode="none", use_g1=False, g1_r_cut=5.5,
                  g1_max_neighbors=48, g1_t_range=2, use_g2=False, g2_r_cut=5.5,
                  q1_coord=False,
                  q1_hidden=128, q2_fourier=False, c5_moe=False,
                  r1a_point=False, r1b_coord=False, use_macro_lattice=False,
                  macro_lattice_mean=(2.96373232, 1.22763338),
                  macro_lattice_std=(0.41023959, 0.51526549)):
        super().__init__()
        self.decoupled_decoder = decoupled_decoder
        self.use_gated_cross_attn = use_gated_cross_attn
        self.head_type = head_type
        self.predict_scale = predict_scale
        self.atom_feat_mode = atom_feat_mode
        self.energy_code = energy_code
        self.scale_mode = scale_mode
        self.scalar_mode = scalar_mode
        # C5: token-level mixture-of-experts is intentionally restricted to
        # M1's shared decoder.  PhysMoE/decoupled routing is a later C3 task.
        self.c5_moe = bool(c5_moe)
        if self.c5_moe:
            assert not decoupled_decoder, "C5 token MoE requires M1's shared decoder"
        # E10: CIF-only global crystal state.  This branch is absent unless
        # enabled, keeping every legacy state dict and off-path bitwise stable.
        self.use_macro_lattice = bool(use_macro_lattice)
        if self.use_macro_lattice:
            self.register_buffer("macro_atomic_masses", raw_atomic_mass_table())
            self.register_buffer("macro_lattice_mean",
                                 torch.tensor(macro_lattice_mean, dtype=torch.float32))
            self.register_buffer("macro_lattice_std",
                                 torch.tensor(macro_lattice_std, dtype=torch.float32))
            if self.macro_lattice_mean.shape != (2,) or self.macro_lattice_std.shape != (2,):
                raise ValueError("E10 macro lattice statistics must each have exactly two values")
            if not torch.isfinite(self.macro_lattice_std).all() or (self.macro_lattice_std <= 0).any():
                raise ValueError("E10 macro lattice standard deviations must be finite and positive")
            self.macro_lattice_mlp = nn.Sequential(
                nn.Linear(2, 64), nn.GELU(), nn.Linear(64, d_model))
            self.macro_lattice_alpha = nn.Parameter(torch.zeros(()))
        # R1a: parameter-matched pointwise MLP readout heads.
        self.r1a_point = bool(r1a_point)
        # R1b: coordinate query generator requires R1a pointwise readout head.
        self.r1b_coord = bool(r1b_coord)
        if self.r1b_coord:
            self.r1a_point = True
        # Optional sparse periodic graph. The hub token starts with zero weight.
        self.use_g1 = bool(use_g1)
        self.g1_r_cut = float(g1_r_cut)
        self.g1_max_neighbors = int(g1_max_neighbors)
        self.g1_t_range = int(g1_t_range)
        # G2a: periodic multi-image radial Value residual (default off).
        # Single-factor discipline: G2a sits on B7 dense attention, never on G1.
        self.use_g2 = bool(use_g2)
        self.g2_r_cut = float(g2_r_cut)
        if self.use_g1 and self.use_g2:
            raise ValueError("G2a is a single-factor module on B7; use_g1 and use_g2 are mutually exclusive")
        if self.use_g1:
            self.g1_global = nn.Parameter(torch.zeros(1, 1, d_model))
        # Optional coordinate-conditioned output trunks.
        # eDOS coordinates are in eV relative to Fermi energy; phDOS
        # coordinates are in cm^-1 relative to zero. Fourier mode replaces the
        # plain coordinate MLP while retaining the same input pathway.
        self.q1_coord = bool(q1_coord or q2_fourier)
        self.q2_fourier = bool(q2_fourier)
        if self.q1_coord:
            if self.q2_fourier:
                self.edos_trunk = FourierTrunk(
                    d_model, hidden_dim=q1_hidden, n_freq=64, sigma=8.0,
                    x_scale=6.0, seed=42, x_min=-1.0, x_max=1.0)
                self.phdos_trunk = FourierTrunk(
                    d_model, hidden_dim=q1_hidden, n_freq=32, sigma=2.0,
                    x_scale=980.0, seed=43, x_min=-280.0 / 980.0, x_max=1.0)
            else:
                self.edos_trunk = CoordTrunk(d_model, hidden_dim=q1_hidden, x_scale=6.0)
                self.phdos_trunk = CoordTrunk(d_model, hidden_dim=q1_hidden, x_scale=980.0)
        # Optional scale head: log eDOS scale per atom and log phDOS total.
        if scale_mode == "decoupled":
            self.scale_head_c24 = ScaleHead(d_model, hidden_dim=128, out_dim=2)
            with torch.no_grad():
                self.scale_head_c24.mlp[-1].bias.copy_(
                    torch.tensor([1.5, 3.2]))  # ~log(4.4), ~log(25)
        # Bounded coverage head (phonon eta and eDOS gamma) for auxiliary loss.
        if scale_mode == "eta":
            self.eta_head = EtaHead(d_model, hidden_dim=128, out_dim=2)
        # Optional bounded boundary scalars.
        if scalar_mode == "s1":
            self.scalar_head = EtaHead(d_model, hidden_dim=128, out_dim=3)
        # Optional eDOS energy encoding, initialized as a zero residual.
        if energy_code == "edos":
            assert edos_grid is not None, "edos_grid bin centers required"
            self.edos_energy = EnergyCode(d_model)
            self.register_buffer("edos_grid",
                                 torch.tensor(np.asarray(edos_grid, dtype=np.float32)))
        else:
            self.edos_energy = None

        # Atom type embedding
        self.tok_emb = nn.Embedding(token_num, d_model)
        # Numeric atomic-feature embedding.
        _feat_dim = 24 if atom_feat_mode == "mendeleev24" else 3
        self.num_emb_encoder = AtomFeatureEncoder(input_dim=_feat_dim, out_dim=d_model,
                                                  feat=atom_feat_mode)
        # LayerNorms for matching distributions
        self.atom_norm = nn.LayerNorm(d_model)
        self.num_norm  = nn.LayerNorm(d_model)
        # Fusion projection
        self.fuse_proj = nn.Linear(d_model * 2, d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        if self.use_g2:
            # Six independent per-layer residuals (~0.559M params each).
            # Off-path creates no attribute, keeping legacy state dicts intact.
            self.encoder.g2_msgs = nn.ModuleList([
                PeriodicEdgeMessage(d_model, rbf_num=64, r_cut=self.g2_r_cut)
                for _ in range(num_encoder_layers)
            ])

        # Decoder initialization (shared vs decoupled)
        if decoupled_decoder:
            self.edos_decoder = TransformerDecoder(
                TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before),
                num_decoder_layers, nn.LayerNorm(d_model)
            )
            self.phdos_decoder = TransformerDecoder(
                TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before),
                num_decoder_layers, nn.LayerNorm(d_model)
            )
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward,
                dropout, activation, normalize_before
            )
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm,
                c5_moe=self.c5_moe)

        # Post-Decoder Zero-Initialized Gated Multi-Head Cross-Attention
        if use_gated_cross_attn:
            self.gated_cross_attn = PostDecoderGatedCrossAttention(d_model, nhead, dropout=dropout)

        # --- (EDOS / PhDOS queries) ---
        if self.r1b_coord:
            q_hidden = min(128, d_model * 4)
            self.q_e = CoordQueryGenerator(d_model=d_model, hidden_dim=q_hidden, x_scale=6.0)
            self.q_p = CoordQueryGenerator(d_model=d_model, hidden_dim=q_hidden, x_scale=980.0)
            self.register_buffer("default_edos_x",
                                 torch.linspace(-6.0 + 6.0 / edos_num, 6.0 - 6.0 / edos_num, edos_num))
            self.register_buffer("default_phdos_x",
                                 torch.linspace(-280.0 + 630.0 / phdos_num, 980.0 - 630.0 / phdos_num, phdos_num))
            self.edos_query_embed = None
            self.edos_tgt = None
            self.phdos_query_embed = None
            self.phdos_tgt = None
        else:
            # --- (EDOS)  ---
            self.edos_query_embed = nn.Parameter(torch.zeros(edos_num, d_model))
            self.edos_tgt = nn.Parameter(torch.zeros(edos_num, d_model))

            # --- (PhDOS)  ---
            self.phdos_query_embed = nn.Parameter(torch.zeros(phdos_num, d_model))
            self.phdos_tgt = nn.Parameter(torch.zeros(phdos_num, d_model))
            self.q_e = None
            self.q_p = None

        self._reset_parameters()
        if atom_feat_mode == "mendeleev24":
            # Token embedding starts silent and learns corrections to numeric features.
            with torch.no_grad():
                self.tok_emb.weight.zero_()
        if energy_code == "edos":
            # Apply zero initialization after generic parameter initialization.
            with torch.no_grad():
                self.edos_energy.proj.weight.zero_()
                self.edos_energy.proj.bias.zero_()
        if self.use_g1:
            # The hub token starts silent and learns through gradients.
            with torch.no_grad():
                self.g1_global.zero_()
        if self.use_macro_lattice:
            with torch.no_grad():
                self.macro_lattice_alpha.zero_()
        if self.q1_coord:
            # Coordinate trunks start silent and learn through gradients.
            with torch.no_grad():
                for _tr in (self.edos_trunk, self.phdos_trunk):
                    _tr.proj.weight.zero_()
                    _tr.proj.bias.zero_()

        # Output Heads (~0.788M params each for symmetric configuration)
        if self.r1a_point:
            if d_model == 512:
                self.edos_out_head = PointwiseMLPHead([512, 3, 1])
                self.phdos_out_head = PointwiseMLPHead([512, 2704, 2704, 2704, 2704, 2704, 1])
            else:
                self.edos_out_head = PointwiseMLPHead([d_model, 3, 1])
                h_dim = max(16, d_model * 2)
                self.phdos_out_head = PointwiseMLPHead([d_model, h_dim, h_dim, h_dim, h_dim, h_dim, 1])
        elif head_type == "symmetric":
            self.edos_out_head = MultiScaleResidualHead(d_model, hidden_dim=256)
            self.phdos_out_head = DeepConv1dHead(d_model, hidden_dim=256)
            nn.init.constant_(self.edos_out_head.out_conv.bias, 1.0)
            nn.init.constant_(self.phdos_out_head.net[-1].bias, 1.0)
        elif head_type == "ph_trimmed":
            self.edos_out_head = CNN(d_model, d_model * 3, output_dim=1, num_layers=1)
            self.phdos_out_head = DeepConv1dHead(d_model, hidden_dim=256)
            nn.init.constant_(self.phdos_out_head.net[-1].bias, 1.0)
        else:  # "legacy"
            self.edos_out_head = CNN(d_model, d_model * 3, output_dim=1, num_layers=1)
            self.phdos_out_head = CNN(d_model, d_model * 3, output_dim=1, num_layers=6)

        # Scale Head MLP for blind physical inference (Shape-Scale decoupled regression)
        if predict_scale:
            self.scale_head = ScaleHead(d_model, hidden_dim=128, out_dim=2)
            with torch.no_grad():
                self.scale_head.mlp[-1].bias.copy_(torch.tensor([3.0, -1.5]))
            assert torch.allclose(self.scale_head.mlp[-1].bias, torch.tensor([3.0, -1.5])), "ScaleHead bias verification failed!"

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, src, mask, pos, edos_x=None, phdos_x=None):
        # src: [B, L] atom indices; pos carries lattice+coords
        # edos_x/phdos_x: [E] or [B,E] bin centers in task units, supplied by
        # the dataset whenever coordinate conditioning is enabled.
        B, Lp, _ = pos.shape
        atom_len = Lp - 2
        mask_atom = mask[:, 2:2 + atom_len]  # 严格对齐 src[:, 2:] 剥离哨兵后的原子区间

        # Extract atom indices and numeric features
        atom_idx = src[:, 2:]  # [B, L]
        atom_emb = self.tok_emb(atom_idx)         # [B, L, d_model]
        num_emb  = self.num_emb_encoder(atom_idx) # [B, L, d_model]

        # Normalize each stream
        atom_emb = self.atom_norm(atom_emb)
        num_emb  = self.num_norm(num_emb)

        # Fuse into unified embedding
        fused = torch.cat([atom_emb, num_emb], dim=-1)  # [B, L, 2*d_model]
        atom_src = self.fuse_proj(fused)                # [B, L, d_model]

        if self.use_macro_lattice:
            macro = macro_lattice_features(
                pos, atom_idx, mask_atom, self.macro_atomic_masses,
                self.macro_lattice_mean, self.macro_lattice_std)
            atom_src = atom_src + self.macro_lattice_alpha * self.macro_lattice_mlp(macro).unsqueeze(1)

        # Compute relative geometry features
        if self.use_g1:
            # Sparse periodic graph and one global hub token.
            # Decoder/memory contract unchanged: hub is stripped before return.
            from utils.g1_graph import build_g1_graph
            g_d, g_u, g_adj, g_sm = build_g1_graph(
                pos, mask_atom, r_cut=self.g1_r_cut,
                max_neighbors=self.g1_max_neighbors, t_range=self.g1_t_range)
            L = atom_src.size(1)
            atom_src_ext = torch.cat([atom_src, self.g1_global.expand(B, -1, -1)], dim=1)
            mask_ext = torch.cat(
                [mask_atom, torch.zeros(B, 1, dtype=torch.bool, device=mask_atom.device)], dim=1)
            Le = L + 1
            d_ext = torch.zeros(B, Le, Le, device=pos.device, dtype=g_d.dtype)
            d_ext[:, :L, :L] = g_d
            u_ext = torch.zeros(B, Le, Le, 3, device=pos.device, dtype=g_u.dtype)
            u_ext[:, :L, :L] = g_u
            adj_ext = torch.zeros(B, Le, Le, dtype=torch.bool, device=mask_atom.device)
            adj_ext[:, :L, :L] = g_adj
            valid_ext = ~mask_ext
            adj_ext[:, L, :] = valid_ext  # hub query sees all valid + self
            adj_ext[:, :, L] = True  # every query sees the hub (padded rows: harmless)
            sm_ext = torch.zeros(B, Le, Le, device=pos.device, dtype=g_sm.dtype)
            sm_ext[:, :L, :L] = g_sm
            sm_ext[:, L, :] = 1.0  # hub edges carry no geometry (rp==0 there)
            sm_ext[:, :, L] = 1.0
            sm_ext = torch.where(adj_ext, sm_ext, torch.zeros_like(sm_ext))
            memory_ext = self.encoder(
                src=atom_src_ext,
                src_key_padding_mask=mask_ext,
                pos=pos,
                rel_diss=d_ext,
                rel_dirs=u_ext,
                g1_adj=adj_ext,
                g1_smooth=sm_ext,
            )
            memory = memory_ext[:, :L, :]
        else:
            distances, unit_dirs = compute_relative_features(pos)
            g2_edges = None
            if self.use_g2:
                from utils.g2_periodic_edges import build_g2_edges
                g2_edges = build_g2_edges(pos, mask_atom, r_cut=self.g2_r_cut)

            # Encoder -- sharing
            memory = self.encoder(
                src=atom_src,
                src_key_padding_mask=mask_atom,
                pos=pos,
                rel_diss=distances,
                rel_dirs=unit_dirs,
                g2_edges=g2_edges,
            )

        results = {}

        # Decoder queries
        if self.r1b_coord:
            ex = edos_x if edos_x is not None else self.default_edos_x
            px = phdos_x if phdos_x is not None else self.default_phdos_x
            ex = ex.to(device=pos.device, dtype=pos.dtype)
            px = px.to(device=pos.device, dtype=pos.dtype)
            edos_query = self.q_e(ex)
            if edos_query.shape[0] == 1 and B > 1:
                edos_query = edos_query.expand(B, -1, -1)
            phdos_query = self.q_p(px)
            if phdos_query.shape[0] == 1 and B > 1:
                phdos_query = phdos_query.expand(B, -1, -1)
            E = edos_query.shape[1]
            P = phdos_query.shape[1]
            edos_tgt_input = torch.zeros(B, E, self.d_model, device=pos.device, dtype=edos_query.dtype)
            phdos_tgt_input = torch.zeros(B, P, self.d_model, device=pos.device, dtype=phdos_query.dtype)
        else:
            edos_query = self.edos_query_embed.unsqueeze(0).repeat(B, 1, 1)
            if self.edos_energy is not None:
                edos_query = edos_query + self.edos_energy(self.edos_grid).unsqueeze(0)
            if self.q1_coord:
                # Coordinate-conditioned residual; zero initialization preserves the base path.
                assert edos_x is not None and phdos_x is not None, \
                    "q1_coord needs edos_x/phdos_x from the dataset batch"
                assert edos_x.shape[-1] == edos_query.shape[1], \
                    f"edos_x bins {edos_x.shape[-1]} != {edos_query.shape[1]}"
                edos_query = edos_query + self.edos_trunk(edos_x.to(dtype=edos_query.dtype))
            edos_tgt_input = self.edos_tgt.unsqueeze(0).repeat(B, 1, 1)
            phdos_query = self.phdos_query_embed.unsqueeze(0).repeat(B, 1, 1)
            phdos_tgt_input = self.phdos_tgt.unsqueeze(0).repeat(B, 1, 1)
            if self.q1_coord:
                assert phdos_x.shape[-1] == phdos_query.shape[1], \
                    f"phdos_x bins {phdos_x.shape[-1]} != {phdos_query.shape[1]}"
                phdos_query = phdos_query + self.phdos_trunk(phdos_x.to(dtype=phdos_query.dtype))

        if self.decoupled_decoder:
            hs_edos, _ = self.edos_decoder(
                edos_tgt_input, memory,
                memory_key_padding_mask=mask_atom,
                pos=pos,
                query_pos=edos_query
            )
            hs_phdos, _ = self.phdos_decoder(
                phdos_tgt_input, memory,
                memory_key_padding_mask=mask_atom,
                pos=pos,
                query_pos=phdos_query
            )
        else:
            hs_edos, _ = self.decoder(
                edos_tgt_input, memory,
                memory_key_padding_mask=mask_atom,
                pos=pos,
                query_pos=edos_query
            )
            c5_balance_edos = self.decoder.last_c5_moe_balance
            c5_load_edos = self.decoder.last_c5_moe_load
            hs_phdos, _ = self.decoder(
                phdos_tgt_input, memory,
                memory_key_padding_mask=mask_atom,
                pos=pos,
                query_pos=phdos_query
            )
            c5_balance_phdos = self.decoder.last_c5_moe_balance
            c5_load_phdos = self.decoder.last_c5_moe_load

        # Post-Decoder Gated Cross Attention
        if self.use_gated_cross_attn:
            hs_edos, hs_phdos = self.gated_cross_attn(hs_edos, hs_phdos)

        # Output heads
        if self.r1a_point:
            out_edos = self.edos_out_head(hs_edos).squeeze(-1) # [B, edos_num, 1] -> [B, edos_num]
            out_phdos = self.phdos_out_head(hs_phdos).squeeze(-1) # [B, phdos_num, 1] -> [B, phdos_num]
        else:
            out_edos = self.edos_out_head(hs_edos.permute(0, 2, 1)).squeeze(1) # -> [B, edos_num]
            out_phdos = self.phdos_out_head(hs_phdos.permute(0, 2, 1)).squeeze(1) # -> [B, phdos_num]

        results['edos'] = out_edos
        results['phdos'] = out_phdos
        if self.c5_moe:
            # The shared decoder runs once for each task; balance both paths
            # equally so the auxiliary term cannot favor the longer eDOS axis.
            results['c5_moe_balance'] = 0.5 * (c5_balance_edos + c5_balance_phdos)
            self.last_c5_moe_load = 0.5 * (c5_load_edos + c5_load_phdos)
        else:
            self.last_c5_moe_load = None

        # Optional scale predictions from pooled crystal features.
        if self.scale_mode == "decoupled":
            h_cry = global_masked_pool(memory, mask_atom)
            results['log_scale'] = self.scale_head_c24(h_cry)  # [B,2]

        # Optional bounded coverage predictions for auxiliary supervision.
        if self.scale_mode == "eta":
            h_cry = global_masked_pool(memory, mask_atom)
            results['eta'] = self.eta_head(h_cry)  # [B,2]: [:,0]=eta_ph, [:,1]=gamma_e

        # S1 boundary scalars (aux-only). [:,0]=wmax_n, [:,1]=eg_n, [:,2]=eval_n.
        if self.scalar_mode == "s1":
            h_cry = global_masked_pool(memory, mask_atom)
            results['scalars'] = self.scalar_head(h_cry)  # [B,3]

        # Shape-Scale branch if enabled
        if self.predict_scale:
            shape_edos = safe_shape_norm(out_edos)
            shape_phdos = safe_shape_norm(out_phdos)

            # Physical boundary constraint (§5.1): phDOS at far negative frequency boundary (-280 cm^-1) is strictly zero
            shape_phdos = shape_phdos.clone()
            shape_phdos[:, 0] = 0.0

            # Crystal-level pooling for scale prediction
            h_crystal = global_masked_pool(memory, mask_atom)
            log_scales = self.scale_head(h_crystal) # [B, 2]
            scale_edos = torch.exp(log_scales[:, 0:1])
            scale_phdos = torch.exp(log_scales[:, 1:2])

            results['shape_edos'] = shape_edos
            results['shape_phdos'] = shape_phdos
            results['log_scale_edos'] = log_scales[:, 0]
            results['log_scale_phdos'] = log_scales[:, 1]
            results['scale_edos'] = scale_edos
            results['scale_phdos'] = scale_phdos
            results['phys_edos'] = shape_edos * scale_edos
            results['phys_phdos'] = shape_phdos * scale_phdos

        return results

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # H3 hygiene: single shared RPEncoding for all layers. The relative
        # geometry (distances/dirs) is identical across layers while RPEncoding
        # holds no learnable weights, so per-layer instances only repeated the
        # same expensive spherical-harmonics computation 6x per batch.
        self.rp_encoder = RPEncoding(num_radial=64, lmax=2, cutoff=10.0)
        
    def forward(self, src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            rel_diss=None,
            rel_dirs=None,
            g1_adj=None,
            g1_smooth=None,
            g2_edges=None):
        
        output = src
        # Compute once, reuse across all layers (H3).
        rp_base = self.rp_encoder(rel_diss, rel_dirs) if rel_diss is not None else None
        if rp_base is not None and g1_smooth is not None:
            # Smooth distance envelope; hub edges intentionally carry no geometry.
            rp_base = rp_base * g1_smooth.unsqueeze(-1)
        g2_msgs = getattr(self, "g2_msgs", None)
        if g2_msgs is not None and g2_edges is None:
            raise ValueError("G2a enabled but no g2_edges supplied to the encoder")
        if g2_msgs is None and g2_edges is not None:
            raise ValueError("g2_edges supplied but the encoder has no G2a residuals")
    
        for li, layer in enumerate(self.layers):
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos,
                           rp_base=rp_base,
                           g1_adj=g1_adj)
            if g2_msgs is not None:
                output = g2_msgs[li](
                    output,
                    g2_edges["batch"], g2_edges["dst"],
                    g2_edges["src"], g2_edges["distances"],
                )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, c5_moe=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.c5_moe = bool(c5_moe)
        # C5 changes only the final two decoder FFNs.  Keeping the earlier
        # layers untouched makes the factor and its parameter budget explicit.
        if self.c5_moe:
            assert num_layers >= 2, "C5 needs at least two decoder layers"
            for layer in self.layers[-2:]:
                layer.c5_moe = C5TokenMoEResidual(
                    d_model=layer.linear1.in_features,
                    expert_hidden=768,
                    num_experts=4,
                    top_k=2,
                    dropout=layer.dropout.p,
                )
        self.last_c5_moe_balance = None
        self.last_c5_moe_load = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        balance_terms = []
        load_terms = []
        for layer in self.layers:
            output, attention = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if layer.last_c5_moe_balance is not None:
                balance_terms.append(layer.last_c5_moe_balance)
                load_terms.append(layer.last_c5_moe_load)

        if self.norm is not None:
            output = self.norm(output)

        if balance_terms:
            self.last_c5_moe_balance = torch.stack(balance_terms).mean()
            self.last_c5_moe_load = torch.stack(load_terms).mean(dim=0)
        else:
            self.last_c5_moe_balance = output.new_zeros(())
            self.last_c5_moe_load = output.new_zeros(0)

        return output, attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="leaky_relu", normalize_before=False):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.dim = d_model
        self.nhead = nhead

        # H3 hygiene: removed dead modules (~1.09M params/layer, ~6.56M total):
        #   - self_attn (forward uses hand-rolled scores, never this module)
        #   - rbf_encoder / rel_proj / dir_proj (defined, never called)
        # Per-layer RPEncoding hoisted to TransformerEncoder (shared, buffer-only).
        # rp_proj stays per-layer (learned); 576 = 64 radial * (1+3+5) spherical (lmax=2).
        self.rp_proj = nn.Linear(64 * 9, d_model)
        
        # 用于前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        
    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     rp_base=None,
                     g1_adj=None):

        B, L, _ = src.size()
        
        q, k, v = src, src, src

        if rp_base is None:
            # Fallback for direct layer calls without encoder context.
            rp_base = torch.zeros(B, L, L, 64 * 9, device=src.device, dtype=src.dtype)
        rp_emb = self.rp_proj(rp_base)  # [B, L, L, d_model]
        d_model_head = self.dim // self.nhead
        rp_emb = rp_emb.view(B, L, L, self.nhead, d_model_head)
        q_heads = q.view(B, L, self.nhead, d_model_head)
        rp_scores = (q_heads.unsqueeze(2) * rp_emb).sum(-1)
        rp_scores = rp_scores.permute(0, 3, 1, 2).reshape(B * self.nhead, L, L)
        
        q_scaled = q / (d_model_head ** 0.5)
        q_heads2 = q_scaled.view(B, L, self.nhead, d_model_head).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_model_head)
        k_heads = k.view(B, L, self.nhead, d_model_head).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_model_head)
        base_scores = torch.bmm(q_heads2, k_heads.transpose(1, 2))
        
        # 新的总打分
        total_scores = base_scores + rp_scores  # [B * nhead, L, L]

        # 2. 将 src_key_padding_mask 正确注入重算打分中（阻断 padding 污染）
        if src_key_padding_mask is not None:
            # src_key_padding_mask: [B, L] -> [B * nhead, L, L]，其中 L = src.size(1)
            # 注意：必须用 -1e9 而非 float('-inf')，否则全 padding 行经 softmax 会产生 NaN
            B_, L_ = src.size(0), src.size(1)
            mask_expanded = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            mask_expanded = mask_expanded.repeat(1, self.nhead, L_, 1).view(B_ * self.nhead, L_, L_)
            total_scores = total_scores.masked_fill(mask_expanded, -1e9)

        # G1: exact sparse periodic adjacency (union with padding mask above).
        if g1_adj is not None:
            B_, L_ = src.size(0), src.size(1)
            g1_blocked = (~g1_adj).unsqueeze(1).repeat(
                1, self.nhead, 1, 1).view(B_ * self.nhead, L_, L_)
            total_scores = total_scores.masked_fill(g1_blocked, -1e9)

        # 计算新的注意力权重并输出
        attn_weights_new = F.softmax(total_scores, dim=-1)
        
        # 重新计算注意力输出：将新的权重作用于 v（同样需要拆分成头）
        v_heads = v.view(B, L, self.nhead, d_model_head).permute(0, 2, 1, 3).reshape(B * self.nhead, L, d_model_head)
        attn_output_new = torch.bmm(attn_weights_new, v_heads)
        attn_output_new = attn_output_new.view(B, self.nhead, L, d_model_head).permute(0, 2, 1, 3).reshape(B, L, self.dim)
        
        # 使用新的注意力输出
        src2 = attn_output_new

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class C5TokenMoEResidual(nn.Module):
    """Top-2 token MoE used only by C5's final shared-decoder FFNs.

    The base FFN remains in place.  A zero-initialized scalar controls this
    residual branch, so loading a B7 checkpoint into C5 is exactly output
    preserving before optimization.  The auxiliary value is the standard
    importance/load product minus its constant minimum of one.
    """

    def __init__(self, d_model=512, expert_hidden=768, num_experts=4,
                 top_k=2, dropout=0.05):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, d_model),
            )
            for _ in range(num_experts)
        ])
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [batch, spectrum tokens, d_model].  Only selected experts are
        # evaluated, rather than materializing all-expert activations.
        logits = self.gate(x)
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        self.last_top_indices = top_indices.detach()
        routed = torch.zeros_like(x)
        dispatch = F.one_hot(top_indices, num_classes=self.num_experts).float()
        for expert_id, expert in enumerate(self.experts):
            selected = dispatch[..., expert_id].any(dim=-1)
            if not selected.any():
                continue
            weights = (top_weights * dispatch[..., expert_id]).sum(dim=-1)
            routed[selected] = expert(x[selected]) * weights[selected].unsqueeze(-1)

        # Both soft routing importance and hard Top-2 load participate.  The
        # subtraction removes the constant one at uniform routing.
        importance = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
        load = dispatch.mean(dim=(0, 1, 2))
        balance = self.num_experts * (importance * load).sum() - 1.0
        return self.alpha * routed, balance, load


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.c5_moe = None
        self.last_c5_moe_balance = None
        self.last_c5_moe_load = None

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                      pos: Optional[Tensor] = None,
                      query_pos: Optional[Tensor] = None):
        # 在 Self-Attention 中注入 query_pos (类似 DETR 标准设计)
        q = tgt if query_pos is None else tgt + query_pos
        k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(query=q, key=k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 在 Cross-Attention 中注入 query_pos
        q = tgt if query_pos is None else tgt + query_pos
        k = memory  # 若未来扩展晶格位置，此处可为 memory + pos
        tgt2, attention_v = self.multihead_attn(query=q, key=k, value=memory,
                                                attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        if self.c5_moe is not None:
            c5_delta, self.last_c5_moe_balance, self.last_c5_moe_load = self.c5_moe(tgt)
            tgt2 = tgt2 + c5_delta
        else:
            self.last_c5_moe_balance = None
            self.last_c5_moe_load = None
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_v

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)  # 添加 LeakyReLU 支持
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
