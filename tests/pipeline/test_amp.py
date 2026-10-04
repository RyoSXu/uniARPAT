"""C4 AMP plumbing contracts that are safe to run on CPU."""
import logging
import unittest

import torch

from model.model import basemodel
from utils.experiment_config import ExperimentConfig


def _params(use_amp=None):
    p = dict(
        loss_form="sumnorm_klw", metrics_list=[],
        sub_model=dict(transformer=dict(
            token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
            num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
            dropout=0.0, activation="gelu", normalize_before=False,
            decoupled_decoder=False, use_gated_cross_attn=False, head_type="legacy",
            predict_scale=False, scale_mode="eta")),
        optimizer=dict(transformer=dict(type="AdamW", params=dict(lr=5e-5))),
        lr_scheduler={},
    )
    if use_amp is not None:
        p["use_amp"] = use_amp
    return p


def _input():
    src = torch.tensor([[126, 127, 14, 8, 0, 0]])
    mask = src.eq(0)
    pos = torch.zeros(1, 6, 3)
    pos[:, 0] = torch.tensor([3.0, 3.0, 0.25])
    pos[:, 1] = torch.tensor([90.0, 90.0, 90.0])
    pos[:, 2:4] = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    return src, mask, pos


def _model(use_amp=None):
    logger = logging.getLogger("c4amp-test")
    logger.addHandler(logging.NullHandler()) if not logger.handlers else None
    torch.manual_seed(42)
    model = basemodel(logger, **_params(use_amp))
    model.to(torch.device("cpu"))
    return model


def test_c4_config_default_and_cpu_rejection():
    assert not ExperimentConfig().use_amp
    assert ExperimentConfig(use_amp=True).use_amp
    model = _model(use_amp=True)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "requires a CUDA device"):
        model.amp_autocast()


def test_c4_off_path_is_bitwise_equal_to_explicit_false():
    implicit = _model()
    explicit = _model(use_amp=False)
    assert not implicit.gscaler.is_enabled()
    assert not explicit.gscaler.is_enabled()
    src, mask, pos = _input()
    implicit.model["transformer"].eval()
    explicit.model["transformer"].eval()
    with torch.no_grad(), implicit.amp_autocast():
        got_implicit = implicit.fp32_outputs(implicit.model["transformer"](src, mask, pos))
    with torch.no_grad(), explicit.amp_autocast():
        got_explicit = explicit.fp32_outputs(explicit.model["transformer"](src, mask, pos))
    for key in ("edos", "phdos", "eta"):
        assert torch.equal(got_implicit[key], got_explicit[key]), key


class TestC4Amp(unittest.TestCase):
    def test_config_cpu(self):
        test_c4_config_default_and_cpu_rejection()

    def test_off_path(self):
        test_c4_off_path_is_bitwise_equal_to_explicit_false()


if __name__ == "__main__":
    unittest.main()
