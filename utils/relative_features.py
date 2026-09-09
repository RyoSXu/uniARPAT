# utils/relative_features.py
import torch
import math

def build_cell_from_lattice(pos):
    """
    从 pos 中提取晶格参数并构造 cell 矩阵，同时返回原子分数坐标。
    pos: [B, L+2, 3], 其中
        pos[:,0] = (a,b,c)
        pos[:,1] = (alpha,beta,gamma) 单位度
        pos[:,2:] = fractional coords
    returns:
        cell: [B,3,3] 晶胞矩阵
        frac_coords: [B, L, 3] 分数坐标
    """
    B, Lp, _ = pos.shape
    abc    = pos[:, 0]       # [B, 3] = (a, b, inv_c) 归一化特征契约
    angles = pos[:, 1]       # [B, 3] = (alpha, beta, gamma) 单位度
    a = abc[:, 0]
    b = abc[:, 1]
    # FIX-P0-02: 第三列存的是 1/c 归一化特征，必须倒数还原为真实实空间晶格常数 c
    inv_c = abc[:, 2].clamp(min=1e-4)
    c = 1.0 / inv_c

    # 防御性物理断言:过滤非法晶格常数 (Hygiene 2026-09-09: 上界从 60 放宽至 1000,
    # 训练集实测 16 个合法长轴样本 c=65~110A (层状/链状), 旧上界会误杀导致整轮崩溃。
    # 下界 0.1 仍可捕获 1/c 契约违反类 bug (忘倒数 -> c~0.01; 误用 c 作 inv_c -> c~0.14))。
    assert ((a > 0.1) & (a < 1000) & (b > 0.1) & (b < 1000) & (c > 0.1) & (c < 1000)).all(), \
        f"Illegal lattice parameters detected: a=[{a.min()},{a.max()}], b=[{b.min()},{b.max()}], c=[{c.min()},{c.max()}]"
    α = angles[:,0] * math.pi/180
    β = angles[:,1] * math.pi/180
    γ = angles[:,2] * math.pi/180

    cosα, cosβ, cosγ = torch.cos(α), torch.cos(β), torch.cos(γ)
    sinγ = torch.sin(γ)

    cell = torch.zeros(B,3,3, device=pos.device, dtype=pos.dtype)
    # a 向量
    cell[:,0,0] = a
    # b 向量
    cell[:,1,0] = b * cosγ
    cell[:,1,1] = b * sinγ
    # c 向量
    cell[:,2,0] = c * cosβ
    cell[:,2,1] = c * (cosα - cosβ * cosγ) / (sinγ + 1e-8)
    tmp = 1 - cosβ**2 - ((cosα - cosβ * cosγ)/(sinγ + 1e-8))**2
    cell[:,2,2] = c * torch.sqrt(torch.clamp(tmp, min=0.0))

    frac_coords = pos[:,2: , :]  # [B, L, 3]
    return cell, frac_coords

def compute_relative_features(pos, cutoff=10.0):
    """
    计算相对距离 & 单位方向（含 PBC 校正和实坐标转换）
    pos: [B, L+2, 3]，前两行是晶格参数，后续是分数坐标
    cutoff: 用于 clamp 距离的上限（可选）
    返回:
        distances: [B, L, L] 真实坐标下欧氏距离
        unit_dirs: [B, L, L, 3] 单位方向向量
    """
    # 1. 构造 cell & 取出分数坐标
    cell, frac_coords = build_cell_from_lattice(pos)
    B, L, _ = frac_coords.shape

    # 2. PBC 最短镜像（fractional）
    diff_frac = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)  # [B,L,L,3]
    diff_frac = diff_frac - diff_frac.round()                        # 折回 [-0.5,0.5]

    # 3. 转到实坐标 (frac → cartesian)
    # cell: [B,3,3], diff_frac: [B,L,L,3]
    # diff_cart[b,i,j,:] = diff_frac[b,i,j,:] @ cell[b].T
    diff_cart = torch.einsum('bijd,bdk->bijk', diff_frac, cell)

    # 4. 计算距离 & 单位方向
    distances = torch.norm(diff_cart, dim=-1)            # [B,L,L]
    distances = distances.clamp(max=cutoff*1.1)          # 可选 clamp
    unit_dirs = diff_cart / (distances.unsqueeze(-1) + 1e-8)  # [B,L,L,3]

    return distances, unit_dirs
