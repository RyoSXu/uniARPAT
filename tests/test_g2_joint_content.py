"""Candidate-1 joint edge content contracts (CPU, synthetic inputs only).

Frozen scope: ``docs/design/design-model-upgrade-candidates.md``
"2026-09-27实施授权与冻结范围" plus the candidate-1 formula block. No real
data, no Q1 cache, no training runs, no GPU, no checkpoint files.

Every behavioral check runs either with ``alpha != 0`` or with a
receiver/sender perturbation, so the suite cannot pass on the ``alpha == 0``
identity shortcut: dropping the receiver path or silently disabling the joint
branch fails ``test_joint_receiver_state_sensitivity_radial_control``.

Covered contracts:
  keys:     radial state-dict strict roundtrip, default adds no keys,
            disabled/radial branch instantiates no joint parameters,
            fixed D512 per-layer parameter count 1346305, hidden fixed at 256
  loading:  B7-shared weights -> joint missing exactly the new branch keys,
            alpha=0 eval equality of eDOS/phDOS and H1 eta/gamma, global
            strict restore semantics unchanged
  math:     literal formula reference on multi-image edges, multi-image and
            degree semantics, empty edges, permutation equivariance,
            padding isolation, finite receiver/sender/distance gradients
  wiring:   CLI choices + config default, runner early rejection of illegal
            combinations before directories/data, mode reaches the actual
            Transformer constructor and the config_used record
"""

import contextlib
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
import yaml

import run_ablation_experiments as runner
from model.transformer import PeriodicEdgeMessage, Transformer
from tests.test_g2_periodic_edges import _pos
from utils.builder import ConfigBuilder
from utils.experiment_config import ExperimentConfig
from utils.g2_periodic_edges import build_g2_edges, g2_quintic_cutoff, g2_rbf_features

# Frozen key sets for the 1-layer contract models (num_encoder_layers=1).
_G2_LAYER = "encoder.g2_msgs.0."
_RADIAL_G2_KEYS = {_G2_LAYER + k for k in (
    "W_v.weight", "W_v.bias", "W_g.weight", "W_g.bias",
    "W_o.weight", "W_o.bias", "alpha", "rbf_centers", "rbf_width")}
_JOINT_BRANCH_KEYS = {_G2_LAYER + k for k in (
    "W_i.weight", "W_i.bias", "phi1.weight", "phi1.bias",
    "phi2.weight", "phi2.bias")}


def _tiny(d_model=32, nhead=4, use_g2=False, g2_content_mode=None,
          scale_mode="eta", seed=0):
    """Small contract model; ``g2_content_mode=None`` keeps the keyword absent."""
    torch.manual_seed(seed)
    extra = {}
    if g2_content_mode is not None:
        extra["g2_content_mode"] = g2_content_mode
    return Transformer(
        token_num=128, d_model=d_model, nhead=nhead, edos_num=8, phdos_num=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, activation="gelu", normalize_before=False,
        decoupled_decoder=True, use_gated_cross_attn=False, head_type="legacy",
        predict_scale=False, use_g2=use_g2, scale_mode=scale_mode, **extra,
    )


def _model_inputs(n_atoms=4, seed=11):
    """Full-model (src, mask, pos) triple on a simple cubic cell."""
    torch.manual_seed(seed)
    fracs = torch.rand(n_atoms, 3) * 0.9 + 0.05
    pos = _pos(4.0, 5.0, 6.0, (90.0, 90.0, 90.0), fracs)
    src = torch.zeros(1, 2 + n_atoms, dtype=torch.long)
    src[0, 0] = 126
    src[0, 1] = 127
    src[0, 2:] = torch.randint(1, 30, (n_atoms,))
    return src, src.eq(0), pos


def _reference_joint(msg, h, edge_batch, edge_dst, edge_src, edge_dist):
    """Literal candidate-1 formula, written from the frozen design block.

    Uses the shared G2a helpers (``g2_rbf_features``/``g2_quintic_cutoff``)
    instead of the module's inlined math, so it is an independent path.
    """
    B, L, D = h.shape
    dist = edge_dist.to(dtype=h.dtype)
    phi = g2_rbf_features(dist, r_cut=msg.r_cut,
                          centers=msg.rbf_centers, width=msg.rbf_width)
    e = msg.W_g(phi)                                   # [E, D]
    u = msg.W_i(h)[edge_batch, edge_dst]               # [E, D]
    v = msg.W_v(h)[edge_batch, edge_src]               # [E, D]
    c = msg.phi2(F.silu(msg.phi1(torch.cat([u, v, e], dim=-1))))
    m = g2_quintic_cutoff(dist, r_cut=msg.r_cut).unsqueeze(-1) * c
    lin = edge_batch * L + edge_dst
    agg_flat = torch.zeros(B * L, D, dtype=h.dtype)
    agg_flat.index_add_(0, lin, m)
    deg = torch.zeros(B * L, dtype=h.dtype)
    deg.index_add_(0, lin, torch.ones_like(dist))
    agg = agg_flat.view(B, L, D) * torch.rsqrt(deg.view(B, L).clamp(min=1.0)).unsqueeze(-1)
    return h + msg.alpha * msg.W_o(agg)


# ---------------------------------------------------------------------------
# 1. keys and parameter registration
# ---------------------------------------------------------------------------

def test_radial_default_state_dict_strict_roundtrip_and_no_new_keys():
    default = _tiny(use_g2=True)  # keyword absent: pre-change construction
    explicit = _tiny(use_g2=True, g2_content_mode="radial")
    assert default.encoder.g2_msgs[0].content_mode == "radial"
    assert set(default.state_dict()) == set(explicit.state_dict())
    msg = default.encoder.g2_msgs[0]
    assert set(dict(msg.named_parameters())) == {
        "W_v.weight", "W_v.bias", "W_g.weight", "W_g.bias",
        "W_o.weight", "W_o.bias", "alpha"}
    assert set(dict(msg.named_buffers())) == {"rbf_centers", "rbf_width"}
    assert not any(".W_i." in k or ".phi1." in k or ".phi2." in k
                   for k in default.state_dict())
    for mode in ("radial", "joint"):
        model = _tiny(use_g2=True, g2_content_mode=mode)
        twin = _tiny(use_g2=True, g2_content_mode=mode)
        ret = twin.load_state_dict(model.state_dict())  # strict=True default
        assert ret.missing_keys == [] and ret.unexpected_keys == []
        assert set(model.state_dict()) == set(twin.state_dict())
        for k, v in model.state_dict().items():
            assert torch.equal(twin.state_dict()[k], v), k


def test_disabled_or_radial_branch_has_no_new_params():
    off = _tiny(use_g2=False)
    assert not hasattr(off.encoder, "g2_msgs")
    assert not any("g2_msgs" in k for k in off.state_dict())
    radial = _tiny(use_g2=True, g2_content_mode="radial")
    assert not any(".W_i." in k or ".phi1." in k or ".phi2." in k
                   for k in radial.state_dict())
    joint = _tiny(use_g2=True, g2_content_mode="joint")
    names = set(dict(joint.encoder.g2_msgs[0].named_parameters()))
    assert names == {
        "W_i.weight", "W_i.bias", "W_v.weight", "W_v.bias",
        "W_g.weight", "W_g.bias", "phi1.weight", "phi1.bias",
        "phi2.weight", "phi2.bias", "W_o.weight", "W_o.bias", "alpha"}


def test_joint_d512_param_count_and_fixed_hidden_width():
    msg = PeriodicEdgeMessage(512, content_mode="joint")
    assert sum(p.numel() for p in msg.parameters()) == 1346305
    radial = PeriodicEdgeMessage(512, content_mode="radial")
    assert sum(p.numel() for p in radial.parameters()) == 558593
    assert msg.phi1.in_features == 3 * 512 and msg.phi1.out_features == 256
    assert msg.phi2.in_features == 256 and msg.phi2.out_features == 512
    try:
        PeriodicEdgeMessage(512, content_mode="joint", joint_hidden=128)
    except TypeError:
        pass
    else:
        raise AssertionError("joint hidden width must not be configurable")


def test_model_rejects_illegal_g2_content_combinations():
    for kwargs in (dict(g2_content_mode="bogus", use_g2=True),
                   dict(g2_content_mode="joint")):  # joint without use_g2
        try:
            _tiny(use_g2=kwargs.get("use_g2", False),
                  g2_content_mode=kwargs["g2_content_mode"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"model must reject {kwargs}")
    try:
        Transformer(d_model=32, nhead=4, use_g1=True, use_g2=True,
                    num_encoder_layers=1, num_decoder_layers=1)
    except ValueError:
        pass
    else:
        raise AssertionError("G1/G2 mutual exclusion must stay in place")


# ---------------------------------------------------------------------------
# 2. formula and edge semantics at nonzero alpha
# ---------------------------------------------------------------------------

def test_joint_matches_reference_formula_on_multi_image_edges():
    torch.manual_seed(0)
    D = 16
    msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="joint")
    with torch.no_grad():
        msg.alpha.fill_(1.0)  # nonzero alpha: never the identity shortcut
    msg.eval()
    h = torch.randn(1, 4, D)
    fracs = torch.tensor([[0.05, 0.05, 0.05], [0.35, 0.10, 0.20],
                          [0.60, 0.60, 0.60], [0.85, 0.20, 0.50]])
    pos = _pos(3.2, 3.2, 3.2, (90.0, 90.0, 90.0), fracs)  # small cell: images
    edges = build_g2_edges(pos, torch.zeros(1, 4, dtype=torch.bool))
    assert edges["batch"].numel() > 4, "fixture must contain multi-image edges"
    with torch.no_grad():
        y = msg(h, edges["batch"], edges["dst"], edges["src"], edges["distances"])
        y_ref = _reference_joint(msg, h, edges["batch"], edges["dst"],
                                 edges["src"], edges["distances"])
    assert torch.isfinite(y).all()
    assert (y - y_ref).abs().max().item() < 1e-5


def test_joint_multi_image_and_degree_semantics():
    torch.manual_seed(1)
    D = 8
    msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="joint")
    with torch.no_grad():
        msg.alpha.fill_(1.0)
        msg.W_o.weight.copy_(torch.eye(D))  # read the aggregate directly
        msg.W_o.bias.zero_()
    msg.eval()
    h = torch.randn(1, 3, D)
    with torch.no_grad():
        one = msg(h, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                  torch.ones(1, dtype=torch.long), torch.tensor([2.0]))
        d1 = one[0, 0] - h[0, 0]
        assert d1.abs().sum().item() > 0
        # k identical images of one sender: sum then 1/sqrt(k) -> sqrt(k) * one
        for k in (2, 3):
            yk = msg(h, torch.zeros(k, dtype=torch.long), torch.zeros(k, dtype=torch.long),
                     torch.ones(k, dtype=torch.long), torch.full((k,), 2.0))
            dk = yk[0, 0] - h[0, 0]
            assert (dk - math.sqrt(k) * d1).abs().max().item() < 1e-5
        # two different senders add before the degree normalization
        eb = torch.zeros(2, dtype=torch.long)
        dst = torch.zeros(2, dtype=torch.long)
        both = msg(h, eb, dst, torch.tensor([1, 2]), torch.tensor([2.0, 3.0]))
        d_both = both[0, 0] - h[0, 0]
        da = msg(h, eb[:1], dst[:1], torch.tensor([1]), torch.tensor([2.0]))[0, 0] - h[0, 0]
        db = msg(h, eb[:1], dst[:1], torch.tensor([2]), torch.tensor([3.0]))[0, 0] - h[0, 0]
        assert (d_both - (da + db) / math.sqrt(2.0)).abs().max().item() < 1e-5
        # untouched receivers stay identical
        assert torch.equal(both[0, 1], h[0, 1]) and torch.equal(both[0, 2], h[0, 2])


def test_joint_receiver_state_sensitivity_radial_control():
    torch.manual_seed(2)
    D = 8
    joint = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="joint")
    radial = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="radial")
    for m in (joint, radial):
        with torch.no_grad():
            m.alpha.fill_(0.7)  # nonzero alpha on both arms
        m.eval()
    eb = torch.zeros(1, dtype=torch.long)
    dst = torch.zeros(1, dtype=torch.long)
    src = torch.ones(1, dtype=torch.long)
    dist = torch.tensor([2.3])
    h = torch.randn(1, 3, D)
    h_recv = h.clone()
    h_recv[0, 0] = torch.randn(D)  # only the receiver state changes
    h_send = h.clone()
    h_send[0, 1] = torch.randn(D)  # only the sender state changes
    with torch.no_grad():
        j1 = joint(h, eb, dst, src, dist)[0, 0] - h[0, 0]
        j2 = joint(h_recv, eb, dst, src, dist)[0, 0] - h_recv[0, 0]
        j3 = joint(h_send, eb, dst, src, dist)[0, 0] - h_send[0, 0]
        r1 = radial(h, eb, dst, src, dist)[0, 0] - h[0, 0]
        r2 = radial(h_recv, eb, dst, src, dist)[0, 0] - h_recv[0, 0]
    # receiver-conditioned content: the joint message must move with h_i
    assert (j1 - j2).abs().max().item() > 1e-6
    # control fixture: G2a radial content is receiver-independent here
    assert (r1 - r2).abs().max().item() < 1e-7
    # sender-conditioned content still moves
    assert (j1 - j3).abs().max().item() > 1e-6


def test_joint_message_permutation_equivariance():
    torch.manual_seed(3)
    L, D = 5, 16
    msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="joint")
    with torch.no_grad():
        msg.alpha.fill_(1.0)
    msg.train()
    h = torch.randn(1, L, D)
    fracs = torch.rand(L, 3) * 0.9 + 0.05
    pos = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs)
    mask = torch.zeros(1, L, dtype=torch.bool)
    edges = build_g2_edges(pos, mask)
    perm = torch.randperm(L)
    pos_p = _pos(5.0, 5.0, 5.0, (90.0, 90.0, 90.0), fracs[perm])
    edges_p = build_g2_edges(pos_p, mask)
    with torch.no_grad():
        y = msg(h, edges["batch"], edges["dst"], edges["src"], edges["distances"])
        y_p = msg(h[:, perm, :], edges_p["batch"], edges_p["dst"],
                  edges_p["src"], edges_p["distances"])
    inv = torch.empty(L, dtype=torch.long)
    inv[perm] = torch.arange(L)
    assert (y_p[:, inv, :] - y).abs().max().item() < 1e-5


def test_joint_padding_does_not_affect_valid_atoms():
    torch.manual_seed(4)
    n_valid, n_pad, D = 3, 2, 8
    msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="joint")
    with torch.no_grad():
        msg.alpha.fill_(1.0)
    msg.eval()
    fracs = torch.rand(n_valid, 3) * 0.9 + 0.05
    pos = _pos(3.5, 3.5, 3.5, (90.0, 90.0, 90.0), fracs)
    pos = torch.cat([pos, torch.full((1, n_pad, 3), 0.123)], dim=1)
    mask = torch.zeros(1, n_valid + n_pad, dtype=torch.bool)
    mask[0, n_valid:] = True
    edges = build_g2_edges(pos, mask)
    valid = set(range(n_valid))
    for i in range(edges["batch"].numel()):
        assert edges["dst"][i].item() in valid, "padded receivers must have no edges"
        assert edges["src"][i].item() in valid, "padded senders must have no edges"
    h = torch.randn(1, n_valid + n_pad, D)
    h_pad = h.clone()
    h_pad[0, n_valid:] = torch.randn(n_pad, D) * 100.0  # garbage in padding
    with torch.no_grad():
        y1 = msg(h, edges["batch"], edges["dst"], edges["src"], edges["distances"])
        y2 = msg(h_pad, edges["batch"], edges["dst"], edges["src"], edges["distances"])
    # the contract: valid atoms never see padding content
    assert torch.equal(y1[:, :n_valid], y2[:, :n_valid])
    # padded rows have no edges; like any edge-free receiver they follow the
    # shared G2a semantics h + alpha * W_o(0), independent of the other rows
    with torch.no_grad():
        zero = torch.zeros(n_pad, D)
        bias_shift = msg.alpha * msg.W_o(zero)
    assert (y1[:, n_valid:] - (h[:, n_valid:] + bias_shift)).abs().max().item() < 1e-6
    assert (y2[:, n_valid:] - (h_pad[:, n_valid:] + bias_shift)).abs().max().item() < 1e-6


def test_joint_empty_edges_identity():
    torch.manual_seed(5)
    D = 8
    for mode in ("radial", "joint"):
        msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode=mode)
        with torch.no_grad():
            msg.alpha.fill_(1.0)  # identity comes from the empty edge set
        h = torch.randn(1, 3, D)
        empty = torch.zeros(0, dtype=torch.long)
        y = msg(h, empty, empty, empty, torch.zeros(0))
        assert torch.equal(y, h), mode


# ---------------------------------------------------------------------------
# 3. gradients (synthetic backward; at most one optimizer step in the suite)
# ---------------------------------------------------------------------------

def test_joint_gradient_paths_receiver_sender_distance():
    torch.manual_seed(6)
    D = 8
    msg = PeriodicEdgeMessage(D, rbf_num=64, r_cut=5.5, content_mode="joint")
    with torch.no_grad():
        msg.alpha.fill_(0.7)  # nonzero alpha so the message path is live
    msg.train()
    h = torch.randn(1, 3, D, requires_grad=True)
    eb = torch.zeros(2, dtype=torch.long)
    dst = torch.zeros(2, dtype=torch.long)
    src = torch.tensor([1, 2])
    dist = torch.tensor([1.7, 2.9], requires_grad=True)
    y = msg(h, eb, dst, src, dist)
    (y * y).sum().backward()
    g = h.grad
    assert g is not None and torch.isfinite(g).all()
    assert g[0, 0].abs().sum().item() > 0, "receiver path carries no gradient"
    assert g[0, 1].abs().sum().item() > 0 and g[0, 2].abs().sum().item() > 0
    assert dist.grad is not None and torch.isfinite(dist.grad).all()
    assert dist.grad.abs().sum().item() > 0, "distance/radial path has no gradient"
    for name in ("W_i", "phi1", "phi2", "W_v", "W_g", "W_o"):
        ps = list(getattr(msg, name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps), name
        assert any(p.grad.abs().sum().item() > 0 for p in ps), name
    assert torch.isfinite(msg.alpha.grad).all() and msg.alpha.grad.abs().item() > 0


def test_joint_alpha_zero_first_step_grad_and_manual_alpha_internal_grads():
    torch.manual_seed(7)
    model = _tiny(use_g2=True, g2_content_mode="joint")
    model.train()
    msg = model.encoder.g2_msgs[0]
    assert float(msg.alpha) == 0.0
    src, mask, pos = _model_inputs(n_atoms=4, seed=21)
    out = model(src, mask, pos)
    (out["edos"].square().mean() + out["phdos"].square().mean()).backward()
    # alpha=0 must never early-exit: the first step still reaches alpha
    assert msg.alpha.grad is not None and torch.isfinite(msg.alpha.grad).all()
    assert msg.alpha.grad.abs().item() > 0
    # the joint branch stays in the graph at alpha=0 (zero grads, not None)
    for name in ("W_i", "phi1", "phi2"):
        assert all(p.grad is not None for p in getattr(msg, name).parameters()), name
    # contract allows at most one optimizer step here: alpha must leave zero
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    opt.step()
    assert float(msg.alpha) != 0.0
    # manually enabled alpha: internal branch parameters must receive signal
    model.zero_grad()
    with torch.no_grad():
        msg.alpha.fill_(0.7)
    out = model(src, mask, pos)
    (out["edos"].square().mean() + out["phdos"].square().mean()).backward()
    for name in ("W_i", "phi1", "phi2", "W_v", "W_g", "W_o"):
        ps = list(getattr(msg, name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps), name
        assert any(p.grad.abs().sum().item() > 0 for p in ps), f"{name} got no signal"


# ---------------------------------------------------------------------------
# 4. checkpoint loading contract (B7-shared weights, alpha=0 equivalence)
# ---------------------------------------------------------------------------

def test_b7_shared_weights_load_alpha0_dos_h1_equivalence():
    base = _tiny(use_g2=False, seed=0)                                # B7-equivalent
    radial = _tiny(use_g2=True, seed=5)                               # G2a radial
    joint = _tiny(use_g2=True, g2_content_mode="joint", seed=9)       # candidate 1
    sd = base.state_dict()
    # B7 -> joint: only the new G2 message branch (radial + joint keys) may miss
    ret_j = runner.load_joint_initial_state(joint, sd)
    assert set(ret_j.missing_keys) == _RADIAL_G2_KEYS | _JOINT_BRANCH_KEYS
    assert ret_j.unexpected_keys == []
    # B7 -> radial: only the G2a keys may miss
    ret_r = radial.load_state_dict(sd, strict=False)
    assert set(ret_r.missing_keys) == _RADIAL_G2_KEYS
    assert ret_r.unexpected_keys == []
    # shared weights are bit-identical to B7 after the non-strict load
    joint_sd = joint.state_dict()
    for k, v in sd.items():
        assert torch.equal(joint_sd[k], v), k
    # the global strict restore rule is untouched: a B7 dict still fails strict
    try:
        _tiny(use_g2=True, g2_content_mode="joint", seed=17).load_state_dict(sd)
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict loading of a B7 state dict must still fail")
    # alpha=0 eval: eDOS/phDOS and H1 eta/gamma stay value-identical to B7
    src, mask, pos = _model_inputs(n_atoms=4, seed=31)
    base.eval()
    with torch.no_grad():
        ob = base(src, mask, pos)
        for other in (radial, joint):
            other.eval()
            oo = other(src, mask, pos)
            for key in ("edos", "phdos", "eta"):
                assert torch.equal(ob[key], oo[key]), (key, other is joint)


# ---------------------------------------------------------------------------
# 5. CLI / config / runner wiring
# ---------------------------------------------------------------------------

def test_cli_and_config_g2_content_mode_contract():
    from run_ablation_experiments import build_arg_parser
    parser = build_arg_parser()
    cfg = ExperimentConfig.from_args(parser.parse_args(["--model", "M1"]))
    assert cfg.g2_content_mode == "radial" and cfg.use_g2 is False
    cfg = ExperimentConfig.from_args(parser.parse_args(
        ["--model", "M1", "--use_g2", "--g2_content_mode", "joint"]))
    assert cfg.g2_content_mode == "joint" and cfg.use_g2 is True
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--model", "M1", "--g2_content_mode", "bogus"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse must reject an unknown g2_content_mode")
    assert ExperimentConfig().g2_content_mode == "radial"


class _NeverIterated:
    """Data-loader stand-in: this contract must never start runner training."""

    def __iter__(self):
        raise AssertionError("the CPU contract must not start runner training")


class TestG2RunnerContracts(unittest.TestCase):
    """Early rejection and pass-through through the actual train_and_eval flow."""

    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "configs").mkdir()
        (self.root / "configs/default.yaml").write_text(
            (repo / "configs/default.yaml").read_text())
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def test_illegal_combinations_rejected_before_dirs_and_data(self):
        bad = (
            dict(g2_content_mode="joint"),                     # joint without use_g2
            dict(g2_content_mode="joint", use_g2=True, use_g1=True),
            dict(g2_content_mode="bogus", use_g2=True),
            dict(use_g1=True, use_g2=True),                    # G1/G2 exclusion
        )
        for extra in bad:
            cfg = ExperimentConfig(model_name="M1", tag="_g2bad", epochs=1, **extra)
            builder = Mock()
            builder.get_dataloader.side_effect = AssertionError("data entry reached")
            with patch.object(runner, "ConfigBuilder", return_value=builder) as ctor:
                with self.assertRaises(ValueError, msg=str(extra)):
                    runner.train_and_eval(cfg)
                ctor.assert_not_called()
                builder.get_dataloader.assert_not_called()
            self.assertFalse((self.root / "output/ablation_m1_g2bad").exists())
            self.assertFalse((self.root / "results/history_m1_g2bad.csv").exists())

    def test_joint_mode_reaches_model_params_and_config_used(self):
        captured = {}
        built = {}
        real_transformer = Transformer

        def _spy(**kwargs):
            captured.update(kwargs)
            tiny = dict(kwargs)
            tiny.update(d_model=32, nhead=4, num_encoder_layers=1,
                        num_decoder_layers=1, dim_feedforward=64, dropout=0.0)
            built["model"] = real_transformer(**tiny)
            return built["model"]

        class _NoDataBuilder(ConfigBuilder):
            """Real config->model chain with a forbidden data entry."""

            def get_dataloader(self, *args, **kwargs):
                return _NeverIterated()

        cfg = ExperimentConfig(model_name="M1", epochs=0, batch_size=2,
                               tag="_g2chain", data_dir="unused",
                               use_g2=True, g2_content_mode="joint",
                               init_ckpt="synthetic-b7.pth",
                               skip_test_eval=True)
        def _b7_state(*args, **kwargs):
            return {"model": {k: v for k, v in built["model"].state_dict().items()
                              if not k.startswith("encoder.g2_msgs.")}}

        with patch.object(runner, "ConfigBuilder", _NoDataBuilder), \
                patch("model.model.Transformer", side_effect=_spy) as ctor, \
                patch.object(torch, "load", side_effect=_b7_state), \
                patch.object(runner, "load_joint_initial_state",
                             wraps=runner.load_joint_initial_state) as init, \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(runner, "logger", Mock()):
            result = runner.train_and_eval(cfg)
        init.assert_called_once()
        self.assertEqual(result, {"test_skipped": True})
        self.assertTrue(ctor.called)
        self.assertEqual(captured.get("g2_content_mode"), "joint")
        self.assertTrue(captured.get("use_g2"))
        self.assertEqual(built["model"].encoder.g2_msgs[0].content_mode, "joint")
        run_dir = self.root / "output/ablation_m1_g2chain"
        # epochs=0: zero training steps, so no checkpoints may exist
        self.assertFalse((run_dir / "checkpoint_latest.pth").exists())
        self.assertFalse((run_dir / "checkpoint_best.pth").exists())
        config = yaml.safe_load((run_dir / "config_used.yaml").read_text())
        self.assertEqual(config["cli"]["g2_content_mode"], "joint")
        self.assertEqual(
            config["config"]["model"]["params"]["sub_model"]["transformer"]["g2_content_mode"],
            "joint")


class TestG2JointContent(unittest.TestCase):
    def test_joint_init_rejects_incomplete_or_wrong_mode_before_mutation(self):
        base = _tiny().state_dict()
        joint = _tiny(use_g2=True, g2_content_mode="joint", seed=17)
        original = {k: v.clone() for k, v in joint.state_dict().items()}
        shared_key = next(iter(base))
        bad_states = [
            {k: v for k, v in base.items() if k != shared_key},
            dict(base, unknown_tensor=torch.zeros(1)),
            dict(base, **{shared_key: torch.zeros(1)}),
            _tiny(use_g2=True).state_dict(),
            {k: v for k, v in original.items() if k != "encoder.g2_msgs.0.phi1.bias"},
        ]
        for state in bad_states:
            with self.assertRaises(ValueError):
                runner.load_joint_initial_state(joint, state)
            for key, value in joint.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]), key)
        complete = _tiny(use_g2=True, g2_content_mode="joint", seed=23).state_dict()
        ret = runner.load_joint_initial_state(joint, complete)
        self.assertFalse(ret.missing_keys or ret.unexpected_keys)
        for key, value in joint.state_dict().items():
            self.assertTrue(torch.equal(value, complete[key]), key)
        for state in (base, _tiny(use_g2=True).state_dict()):
            with self.assertRaises(RuntimeError):
                joint.load_state_dict(state, strict=True)

    def test_default_roundtrip(self):
        test_radial_default_state_dict_strict_roundtrip_and_no_new_keys()

    def test_off_path(self):
        test_disabled_or_radial_branch_has_no_new_params()

    def test_param_count(self):
        test_joint_d512_param_count_and_fixed_hidden_width()

    def test_illegal_model_combos(self):
        test_model_rejects_illegal_g2_content_combinations()

    def test_formula_reference(self):
        test_joint_matches_reference_formula_on_multi_image_edges()

    def test_degree_semantics(self):
        test_joint_multi_image_and_degree_semantics()

    def test_receiver_sensitivity(self):
        test_joint_receiver_state_sensitivity_radial_control()

    def test_permutation(self):
        test_joint_message_permutation_equivariance()

    def test_padding(self):
        test_joint_padding_does_not_affect_valid_atoms()

    def test_empty_edges(self):
        test_joint_empty_edges_identity()

    def test_gradients(self):
        test_joint_gradient_paths_receiver_sender_distance()

    def test_alpha_contract(self):
        test_joint_alpha_zero_first_step_grad_and_manual_alpha_internal_grads()

    def test_b7_load_alpha0_equivalence(self):
        test_b7_shared_weights_load_alpha0_dos_h1_equivalence()

    def test_cli_config(self):
        test_cli_and_config_g2_content_mode_contract()


if __name__ == "__main__":
    unittest.main()
