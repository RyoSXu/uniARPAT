"""uniARPAT-v2 P0 修复 TDD 验证套件 (Step 1).

依据《docs/01_开发文档/2026-09-09_uniARPAT_v2生产级集成与P0修复开发方案.md》第 4 节，
新建独立单元测试脚本，在修改代码前先行编写测试，保障修复验收无死角。

覆盖映射:
  test_mask_alignment              -> FIX-P0-01 (transformer.py:125-126 + model.py:197)
  test_cell_reciprocal_recovery    -> FIX-P0-02 (utils/relative_features.py:17-37)
  test_encoder_redundancy_and_masking -> FIX-P0-03 (model/transformer.py:320-345)
  test_decoder_query_pos_gradient  -> FIX-P0-04 (model/transformer.py:382-395)
  test_cif2dos_smoke               -> cif2dos.py (§3, 纯端到端 CIF->DOS)

设计原则 (TDD):
  每个测试断言修复后的正确行为，在修复前如期失败、修复后通过。
  全部测试 CPU 可跑、无需 checkpoint/大数据、确定性种子、轻量模型。
"""

import math
import re
import unittest
from pathlib import Path

import numpy as np
import torch

from model.transformer import Transformer, TransformerEncoderLayer
from utils.relative_features import build_cell_from_lattice
from thermo_props import ThermodynamicCalculator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_transformer(edos_num=8, phdos_num=4, d_model=32, nhead=4):
    """轻量 Transformer,仅用于单测(1 层 encoder/decoder,无 dropout)."""
    torch.manual_seed(0)
    return Transformer(
        token_num=128,
        d_model=d_model,
        nhead=nhead,
        edos_num=edos_num,
        phdos_num=phdos_num,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        activation="gelu",
        normalize_before=False,
        decoupled_decoder=True,
        use_gated_cross_attn=False,
        head_type="legacy",
        predict_scale=False,
    )


def _make_mask_fixture():
    """FIX-P0-01 契约: src=[126,127]+5 真实原子+75 个 0 padding,总长 82."""
    src = torch.tensor(
        [[126, 127, 14, 14, 8, 8, 8] + [0] * 75], dtype=torch.long
    )  # [1, 82]
    mask = (src == 0)  # 哨兵恒 False,padding 才 True
    atom_len = 80  # Lp - 2,其中 Lp=82
    return src, mask, atom_len


# ---------------------------------------------------------------------------
# FIX-P0-01: Padding Mask 2 列错位
# ---------------------------------------------------------------------------

def test_mask_alignment():
    """FIX-P0-01:掩码必须与 src[:, 2:] 严格对齐.

    验收标准(方案 §4):
      mask_atom[4] is False (第 5 个真实原子有效),
      mask_atom[5] is True  (第 1 个 padding 被遮蔽),
      mask_atom 全长 80 并与 src[:, 2:] 一一对应,
      且 model.py valid_atoms == 5.
    """
    src, mask, atom_len = _make_mask_fixture()

    # 1) 语义契约:正确切片应为 mask[:, 2:2+atom_len]
    mask_atom_expected = mask[:, 2:2 + atom_len]
    assert mask_atom_expected.shape == (1, 80), (
        f"FIX-P0-01: mask_atom 长度应为 80,实得 {mask_atom_expected.shape}"
    )
    # 与 src[:, 2:] 一一对应:有效位 == (atom_idx != 0)
    atom_idx = src[:, 2:]  # [1, 80]
    assert atom_idx.shape == mask_atom_expected.shape
    assert torch.equal(mask_atom_expected, (atom_idx == 0)), (
        "FIX-P0-01: mask_atom 必须与 src[:, 2:] 一一对应 (mask=(inp==0))"
    )
    assert mask_atom_expected[0, 4].item() is False, (
        "FIX-P0-01: mask_atom[4] 应为 False(第 5 个真实原子有效)"
    )
    assert mask_atom_expected[0, 5].item() is True, (
        "FIX-P0-01: mask_atom[5] 应为 True(第 1 个 padding 被遮蔽)"
    )
    # valid_atoms 语义 (model.py:197 同理)
    valid_atoms = (~mask[:, 2:2 + atom_len]).sum(dim=-1).float()
    assert valid_atoms.item() == 5.0, (
        f"FIX-P0-01: valid_atoms 应为 5,实得 {valid_atoms.item()} "
        "(错位切片 mask[:, :atom_len] 会误算为 7)"
    )
    # 反证:错位切片确实是错的(保证本测试能暴露缺陷)
    buggy = mask[:, :atom_len]
    assert buggy[0, 5].item() is False, "测试自检:错位切片 [5] 应为 False(误杀漏杀证据)"
    assert (~buggy).sum().item() == 7.0, "测试自检:错位 valid_atoms 应为 7"

    # 2) 行为回归:真实 Transformer.forward 传入 encoder 的 mask 必须已对齐
    torch.manual_seed(0)
    model = _tiny_transformer()
    model.eval()
    B, Lp = 1, 82
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])  # (a,b,inv_c) 合法晶格
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = torch.rand(B, Lp - 2, 3) * 0.9 + 0.05
    captured = {}
    orig_enc_forward = model.encoder.forward

    def spy_encoder(src, mask=None, src_key_padding_mask=None, pos=None,
                    rel_diss=None, rel_dirs=None, **kwargs):
        captured["mask"] = (
            src_key_padding_mask.detach().cpu()
            if src_key_padding_mask is not None
            else None
        )
        return orig_enc_forward(
            src, mask=mask, src_key_padding_mask=src_key_padding_mask,
            pos=pos, rel_diss=rel_diss, rel_dirs=rel_dirs, **kwargs
        )

    model.encoder.forward = spy_encoder
    with torch.no_grad():
        model(src, mask, pos)
    model.encoder.forward = orig_enc_forward
    assert captured.get("mask") is not None, "未能捕获 encoder mask"
    got = captured["mask"]
    assert got.shape == (1, 80), f"encoder mask 形状应为 [1,80],实得 {got.shape}"
    assert got[0, 4].item() is False, (
        "FIX-P0-01 未修复:encoder 收到的 mask_atom[4] 应为 False"
    )
    assert got[0, 5].item() is True, (
        "FIX-P0-01 未修复:encoder 收到的 mask_atom[5] 应为 True, "
        "当前仍为错位切片 mask[:, :atom_len]"
    )

    # 3) 静态回归:model.py valid_atoms 必须同步改为 mask[:, 2:2+atom_len]
    repo_root = Path(__file__).resolve().parents[1]
    model_py = (repo_root / "model" / "model.py").read_text(encoding="utf-8")
    assert re.search(
        r"mask\s*\[\s*:\s*,\s*2\s*:\s*2\s*\+\s*atom_len\s*\]", model_py
    ), "FIX-P0-01 未修复:model.py valid_atoms 仍误用 mask[:, :atom_len]"


# ---------------------------------------------------------------------------
# FIX-P0-02: 晶格 1/c 倒数反解
# ---------------------------------------------------------------------------

def test_cell_reciprocal_recovery():
    """FIX-P0-02:position.csv 第 3 列是 1/c,必须倒数还原为真实 c.

    输入 a=4.19,b=4.19,inv_c=0.23865 (立方,mp-242 近似),
    恢复体积应在 4.19^3 ±0.01,而非被压缩的 0.238*4.19^2 ≈ 4.19.
    """
    a, b, inv_c = 4.19, 4.19, 0.23865
    pos = torch.tensor(
        [[[a, b, inv_c], [90.0, 90.0, 90.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )  # [1, 3, 3]: L=1,仅测晶胞构造
    cell, _ = build_cell_from_lattice(pos)
    assert cell.shape == (1, 3, 3)
    vol = torch.linalg.det(cell[0]).item()
    expected = 4.19 ** 3  # ≈ 73.560059
    assert math.isfinite(vol), f"FIX-P0-02:体积非有限值 {vol}"
    assert vol > 50.0, (
        f"FIX-P0-02 未修复:体积 {vol:.4f} 仍被压缩在 ~4.19 "
        f"(直接把 1/c 当 c 用),期望 ≈ {expected:.4f}"
    )
    assert abs(vol - expected) < 0.01 + 1e-6, (
        f"FIX-P0-02:恢复体积 {vol:.5f} 不在 4.19^3 ± 0.01 内 "
        f"(期望 {expected:.5f})"
    )


# ---------------------------------------------------------------------------
# FIX-P0-03: Encoder 冗余 MHA + Padding Mask 注入
# ---------------------------------------------------------------------------

def test_encoder_redundancy_and_masking():
    """FIX-P0-03:(1) 剔除冗余 self.self_attn 调用;(2) 被 mask 原子权重严格为 0."""
    torch.manual_seed(0)
    d_model, nhead, B, L = 32, 4, 2, 8
    layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=64, dropout=0.0
    )
    layer.eval()
    src = torch.randn(B, L, d_model)
    rel_diss = torch.rand(B, L, L) * 5.0
    rel_dirs = torch.randn(B, L, L, 3)
    rel_dirs = rel_dirs / (rel_dirs.norm(dim=-1, keepdim=True) + 1e-8)
    # 遮蔽每样本后 3 个原子
    skpm = torch.zeros(B, L, dtype=torch.bool)
    skpm[:, -3:] = True

    # (1) 冗余调用计数:H3 hygiene 已彻底删除 self_attn 死模块,
    # 缓存 RP 由 TransformerEncoder 统一计算一次传入 rp_base。
    assert not hasattr(layer, "self_attn"), (
        "H3 未落实:TransformerEncoderLayer 仍残留死模块 self_attn "
        "(~1.05M 参数/层),应彻底删除"
    )
    assert not hasattr(layer, "rbf_encoder") and not hasattr(layer, "rel_proj"), (
        "H3 未落实:死模块 rbf_encoder/rel_proj 仍残留"
    )
    from utils.rp_encoding import RPEncoding
    rp_base = RPEncoding(num_radial=64, lmax=2, cutoff=10.0)(rel_diss, rel_dirs)
    with torch.no_grad():
        layer(
            src,
            src_key_padding_mask=skpm,
            rp_base=rp_base,
        )

    # (2a) 行为:有效原子输出必须对 padding 取值不变(无注意力泄露)
    torch.manual_seed(1)
    layer2 = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=64, dropout=0.0
    )
    layer2.eval()
    src_a = torch.randn(B, L, d_model)
    src_b = src_a.clone()
    src_b[:, -3:] += 10.0  # 强扰动被遮蔽位
    with torch.no_grad():
        out_a = layer2(
            src_a,
            src_key_padding_mask=skpm,
            rp_base=rp_base,
        )
        out_b = layer2(
            src_b,
            src_key_padding_mask=skpm,
            rp_base=rp_base,
        )
    leak = (out_a[:, : L - 3, :] - out_b[:, : L - 3, :]).abs().max().item()
    assert leak < 1e-5, (
        f"FIX-P0-03 未修复:padding 污染泄露到有效原子 "
        f"(valid 输出 max diff={leak:.4f},期望 <1e-5)"
    )

    # (2b) 直接:被 mask 列经 softmax 后权重严格为 0.0
    import model.transformer as Tmod

    captured = {}
    orig_softmax = torch.nn.functional.softmax

    def spy_softmax(inp, dim=-1, dtype=None):
        out = orig_softmax(inp, dim=dim, dtype=dtype)
        if (
            inp.dim() == 3
            and inp.shape[0] == B * nhead
            and inp.shape[1] == L
            and inp.shape[2] == L
        ):
            captured["w"] = out.detach()
        return out

    import unittest.mock as mock

    with mock.patch.object(Tmod.F, "softmax", side_effect=spy_softmax):
        with torch.no_grad():
            layer2(
                src_a,
                src_key_padding_mask=skpm,
                rp_base=rp_base,
            )
    assert "w" in captured, "未能捕获注意力权重"
    w = captured["w"]  # [B*nhead, L, L],最后一维是 key
    masked_max = w[:, :, -3:].abs().max().item()
    assert masked_max < 1e-6, (
        f"FIX-P0-03 未修复:被 mask 原子注意力权重 max={masked_max:.6f}, "
        "期望 softmax 后严格为 0.0(应以 -1e9 masked_fill 注入)"
    )


# ---------------------------------------------------------------------------
# FIX-P0-04: Decoder 死参 query_pos
# ---------------------------------------------------------------------------

def test_decoder_query_pos_gradient():
    """FIX-P0-04:query_pos 必须进入计算图,梯度非 None 且范数 > 0."""
    torch.manual_seed(0)
    model = _tiny_transformer(edos_num=8, phdos_num=4)
    model.train()  # 确保梯度流通(dropout=0,无随机性影响)
    B, n_atom, Lp = 2, 6, 8
    src = torch.randint(1, 30, (B, Lp), dtype=torch.long)
    src[:, 0] = 126
    src[:, 1] = 127
    mask = (src == 0)  # 全有效(无 padding),隔离 query_pos 效应
    pos = torch.zeros(B, Lp, 3)
    pos[:, 0, :] = torch.tensor([5.0, 5.0, 0.2])
    pos[:, 1, :] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:, :] = torch.rand(B, n_atom, 3) * 0.9 + 0.05

    model.zero_grad()
    outputs = model(src, mask, pos)
    loss = outputs["edos"].sum() + outputs["phdos"].sum()
    assert torch.isfinite(loss).item(), "前向 loss 非有限,无法做梯度测试"
    loss.backward()

    ge = model.edos_query_embed.grad
    gp = model.phdos_query_embed.grad
    assert ge is not None and gp is not None, (
        "FIX-P0-04 未修复:edos/phdos_query_embed.grad 为 None "
        "(query_pos 未加入 q/k,死参不在计算图;直接 .norm() 会 AttributeError)"
    )
    ne = ge.detach().float().norm().item()
    np_ = gp.detach().float().norm().item()
    assert math.isfinite(ne) and math.isfinite(np_), "query 梯度范数非有限"
    assert ne > 0.0, f"FIX-P0-04 未修复:edos query 梯度范数为 0 (死参),实得 {ne}"
    assert np_ > 0.0, f"FIX-P0-04 未修复:phdos query 梯度范数为 0 (死参),实得 {np_}"


# ---------------------------------------------------------------------------
# cif2dos 冒烟:合成 Si 金刚石端到端
# ---------------------------------------------------------------------------

def test_cif2dos_smoke():
    """cif2dos 冒烟:合成 Si 结构 -> 特征契约(1/c) -> 前向 -> 物性.

    验证输出 shape、无 NaN/Inf,且 Cv/Theta_D/kappa_L 计算有限正常。
    镜像未来 cif2dos.py §3.2 核心流程,当前以管线级断言先行锁定契约。
    """
    from pymatgen.core import Lattice, Structure

    torch.manual_seed(0)
    # 合成硅金刚石结构 (a=5.43 Å, Fd-3m 近似)
    lattice = Lattice.cubic(5.43)
    struct = Structure(
        lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    a, b, c = struct.lattice.a, struct.lattice.b, struct.lattice.c
    inv_c = 1.0 / c  # 特征契约:第 3 列存 1/c
    atomic_numbers = list(struct.atomic_numbers)
    assert len(atomic_numbers) == 2

    N = 82  # 与训练数据一致:2 哨兵行 + 80 原子槽
    src = torch.zeros(1, N, dtype=torch.long)
    src[0, 0] = 126
    src[0, 1] = 127
    src[0, 2:2 + len(atomic_numbers)] = torch.tensor(
        atomic_numbers, dtype=torch.long
    )
    mask = (src == 0)
    pos = torch.zeros(1, N, 3)
    pos[0, 0, :] = torch.tensor([a, b, inv_c])
    pos[0, 1, :] = torch.tensor(struct.lattice.angles)
    pos[0, 2:2 + len(struct.frac_coords), :] = torch.tensor(
        np.asarray(struct.frac_coords), dtype=torch.float32
    )

    # 几何契约:实空间体积必须 ≈ a*b*c(修复后),而非 a*b*inv_c
    cell, _ = build_cell_from_lattice(pos)
    vol = torch.linalg.det(cell[0]).item()
    assert math.isfinite(vol) and vol > 50.0, (
        f"冒烟几何契约失败:Si 体积 {vol:.3f} Å³ 异常 "
        f"(期望 ≈ {a * b * c:.3f};若 ≈ {a * b * inv_c:.3f} 则 P0-02 未修复)"
    )

    # 端到端前向(轻量模型,生产维度 128/64)
    model = _tiny_transformer(edos_num=128, phdos_num=64)
    model.eval()
    with torch.no_grad():
        outputs = model(src, mask, pos)
    edos = outputs["edos"]
    phdos = outputs["phdos"]
    assert edos.shape == (1, 128), f"eDOS shape 应为 [1,128],实得 {tuple(edos.shape)}"
    assert phdos.shape == (1, 64), f"phDOS shape 应为 [1,64],实得 {tuple(phdos.shape)}"
    for name, t in (("eDOS", edos), ("phDOS", phdos)):
        assert torch.isfinite(t).all().item(), f"冒烟失败:{name} 含 NaN/Inf"
        assert t.abs().max().item() < 1e6, f"冒烟失败:{name} 量级爆炸"

    # 宏观物性联动(Julian-Slack κL + Bose-Einstein Cv/ΘD)
    calc = ThermodynamicCalculator()
    ph = np.clip(phdos[0].cpu().numpy(), 0.0, None)
    t_range = np.linspace(10, 1000, 10)
    cv = calc.compute_Cv(ph, T_range=t_range)
    theta_d = calc.compute_Debye_T(ph)
    kappa = calc.compute_Slack_kappaL(
        ph, M_avg=28.0855, volume_per_atom=vol / 2.0, n_atoms=2, T=300.0
    )
    assert np.all(np.isfinite(cv)), "冒烟失败:Cv(T) 含 NaN/Inf"
    assert np.all(cv >= 0.0), "冒烟失败:Cv(T) 出现负值"
    assert math.isfinite(theta_d) and theta_d >= 0.0, "冒烟失败:ΘD 非法"
    assert math.isfinite(kappa) and kappa >= 0.0, "冒烟失败:κL 非法"


# ---------------------------------------------------------------------------
# unittest 兼容层 (pytest 直接收集顶层函数;本类仅为无 pytest 环境提供
# `python -m unittest` 入口,设置 __test__=False 避免 pytest 重复收集)
# ---------------------------------------------------------------------------

class TestP0Fixes(unittest.TestCase):
    __test__ = False

    def test_mask_alignment(self):
        test_mask_alignment()

    def test_cell_reciprocal_recovery(self):
        test_cell_reciprocal_recovery()

    def test_encoder_redundancy_and_masking(self):
        test_encoder_redundancy_and_masking()

    def test_decoder_query_pos_gradient(self):
        test_decoder_query_pos_gradient()

    def test_cif2dos_smoke(self):
        test_cif2dos_smoke()


if __name__ == "__main__":
    unittest.main()
