"""L3 loss-term ablation plumbing tests (Design-E section 10).

Gates: w_w1/w_huber reach the model, KL-only loss differs from full
(single factor present), finiteness + gradient flow, determinism.
CPU, real-data batch, no training loop.
"""

import unittest

import torch
import yaml

from utils.builder import ConfigBuilder


def _model_with(w_w1=None, w_huber=None, seed=0):
    torch.manual_seed(seed)
    with open("configs/config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["model"]["params"]["loss_form"] = "sumnorm_klw"
    cfg["model"]["params"]["dos_minmax"] = True
    if w_w1 is not None:
        cfg["model"]["params"]["w_w1"] = float(w_w1)
    if w_huber is not None:
        cfg["model"]["params"]["w_huber"] = float(w_huber)
    b = ConfigBuilder(**cfg)
    model = b.get_model()
    model.device = torch.device("cpu")
    loader = b.get_dataloader(split="train", dos_minmax=True, batch_size=4,
                              dos_sumnorm=True)
    return model, next(iter(loader))


def test_l3_weights_plumb():
    m, _ = _model_with()
    assert m.w_w1 == 1.0 and m.w_huber == 1.0, "defaults must be 1.0/1.0"
    m0, _ = _model_with(w_w1=0, w_huber=0)
    assert m0.w_w1 == 0.0 and m0.w_huber == 0.0, "zero weights must land"


def test_l3_kl_only_differs_and_flows():
    import math
    m_full, batch = _model_with(seed=0)
    m_kl, _ = _model_with(w_w1=0, w_huber=0, seed=0)
    m_full.model["transformer"].train()
    m_kl.model["transformer"].train()
    out_full = m_full.train_one_step(batch, 0)
    out_kl = m_kl.train_one_step(batch, 0)
    lf = out_full["loss_edos"] + out_full["loss_phdos"]
    lk = out_kl["loss_edos"] + out_kl["loss_phdos"]
    for v in (lf, lk):
        assert math.isfinite(v) and v > 0.0
    assert abs(lf - lk) > 1e-6, "KL-only must differ (factor present)"
    # KL-only <= full (fewer nonnegative terms), Huber/W1 >= 0 by construction
    assert lk <= lf + 1e-5
    # train_one_step backward()s internally -> grads must be populated
    n = sum(p.grad.abs().max().item() for p in
            m_kl.model["transformer"].parameters()
            if p.grad is not None)
    assert n > 0.0, "no gradient flow under KL-only"


def test_l3_determinism():
    m1, batch = _model_with(w_w1=0, seed=0)
    m2, _ = _model_with(w_w1=0, seed=0)
    m1.model["transformer"].train()
    m2.model["transformer"].train()
    # reseed before each forward: construction of m2 advances global RNG,
    # which would otherwise give different dropout masks (same weights).
    torch.manual_seed(123)
    o1 = m1.train_one_step(batch, 0)
    torch.manual_seed(123)
    o2 = m2.train_one_step(batch, 0)
    assert o1["loss"] == o2["loss"]


class TestL3Loss(unittest.TestCase):
    __test__ = False

    def test_plumb(self):
        test_l3_weights_plumb()

    def test_differs(self):
        test_l3_kl_only_differs_and_flows()

    def test_determinism(self):
        test_l3_determinism()


if __name__ == "__main__":
    unittest.main()
