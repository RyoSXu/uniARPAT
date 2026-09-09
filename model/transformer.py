import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.heads import (
    CNN, DeepConv1dHead, MultiScaleResidualHead,
    PostDecoderGatedCrossAttention, ScaleHead, global_masked_pool
)
from utils.atom_feature import AtomFeatureEncoder
from utils.relative_features import compute_relative_features
from utils.rbf_encoding import RBFEncoding
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


class Transformer(nn.Module):

    def __init__(self, token_num=118, d_model=512, nhead=8, edos_num=128, phdos_num=64, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False,
                 decoupled_decoder=False, use_gated_cross_attn=False,
                 head_type="legacy", predict_scale=False):
        super().__init__()
        self.decoupled_decoder = decoupled_decoder
        self.use_gated_cross_attn = use_gated_cross_attn
        self.head_type = head_type
        self.predict_scale = predict_scale

        # Atom type embedding
        self.tok_emb = nn.Embedding(token_num, d_model)
        # Numeric atomic feature embedding
        self.num_emb_encoder = AtomFeatureEncoder(input_dim=3, out_dim=d_model)
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
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # Post-Decoder Zero-Initialized Gated Multi-Head Cross-Attention
        if use_gated_cross_attn:
            self.gated_cross_attn = PostDecoderGatedCrossAttention(d_model, nhead, dropout=dropout)

        # --- (EDOS)  ---
        self.edos_query_embed = nn.Parameter(torch.zeros(edos_num, d_model))
        self.edos_tgt = nn.Parameter(torch.zeros(edos_num, d_model))

        # --- (PhDOS)  ---
        self.phdos_query_embed = nn.Parameter(torch.zeros(phdos_num, d_model))
        self.phdos_tgt = nn.Parameter(torch.zeros(phdos_num, d_model))

        self._reset_parameters()

        # Output Heads (~0.788M params each for symmetric configuration)
        if head_type == "symmetric":
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

    def forward(self, src, mask, pos):
        # src: [B, L] atom indices; pos carries lattice+coords
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

        # Compute relative geometry features
        distances, unit_dirs = compute_relative_features(pos)

        # Encoder -- sharing
        memory = self.encoder(
            src=atom_src,
            src_key_padding_mask=mask_atom,
            pos=pos,
            rel_diss=distances,
            rel_dirs=unit_dirs
        )

        results = {}

        # Decoder queries
        edos_query = self.edos_query_embed.unsqueeze(0).repeat(B, 1, 1)
        edos_tgt_input = self.edos_tgt.unsqueeze(0).repeat(B, 1, 1)
        phdos_query = self.phdos_query_embed.unsqueeze(0).repeat(B, 1, 1)
        phdos_tgt_input = self.phdos_tgt.unsqueeze(0).repeat(B, 1, 1)

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
            hs_phdos, _ = self.decoder(
                phdos_tgt_input, memory,
                memory_key_padding_mask=mask_atom,
                pos=pos,
                query_pos=phdos_query
            )

        # Post-Decoder Gated Cross Attention
        if self.use_gated_cross_attn:
            hs_edos, hs_phdos = self.gated_cross_attn(hs_edos, hs_phdos)

        # Output heads
        out_edos = self.edos_out_head(hs_edos.permute(0, 2, 1)).squeeze(1) # -> [B, edos_num]
        out_phdos = self.phdos_out_head(hs_phdos.permute(0, 2, 1)).squeeze(1) # -> [B, phdos_num]

        results['edos'] = out_edos
        results['phdos'] = out_phdos

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
        
    def forward(self, src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            rel_diss=None,
            rel_dirs=None):
        
        output = src
    
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos,
                           rel_diss=rel_diss,
                           rel_dirs=rel_dirs)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output, attention = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="leaky_relu", normalize_before=False, rbf_encoder=None):
        super().__init__()
        self.activation = _get_activation_fn(activation)
        self.dim = d_model
        self.nhead = nhead
        
        # 标准多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 用于前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        
        self.rbf_encoder = RBFEncoding(num_centers=64, cutoff=10.0)
        self.rel_proj = nn.Linear(self.rbf_encoder.num_centers, d_model)  
        self.max_ell = 3  # 球谐函数最大阶数，自己调整
        dim_sph = sum([2 * l + 1 for l in range(self.max_ell + 1)])  # 球谐展开维度
        self.dir_proj = nn.Linear(dim_sph, d_model)

        self.rp_encoder = RPEncoding(num_radial=64, lmax=2, cutoff=10.0)
        self.rp_proj = nn.Linear(self.rp_encoder.out_dim, d_model)  
        
    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     rel_diss=None,
                     rel_dirs=None):

        B, L, _ = src.size()
        
        q, k, v = src, src, src
        
        rp_emb = self.rp_encoder(rel_diss, rel_dirs)
        rp_emb = self.rp_proj(rp_emb)  # [B, L, L, d_model]
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