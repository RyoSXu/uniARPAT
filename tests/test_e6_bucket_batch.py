"""E6 length bucketing and dynamic padding-trim contracts."""
import unittest
import logging
import math

import torch
from torch.utils.data import SequentialSampler

from datasets.dataset import Dos_Dataset
from model.transformer import Transformer
from utils.bucket_batch import LengthBucketBatchSampler, trim_atom_padding_collate
from utils.experiment_config import ExperimentConfig


def test_e6_bucket_coverage_and_determinism():
    lengths = torch.tensor([8, 2, 7, 1, 6, 3, 5, 4])
    sampler = SequentialSampler(range(len(lengths)))
    bucket = LengthBucketBatchSampler(sampler, lengths, batch_size=2, window_batches=2)
    first = list(bucket)
    second = list(bucket)
    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(len(lengths)))
    # Only each four-index window is sorted; the second window never crosses it.
    assert first == [[3, 1], [2, 0], [5, 7], [6, 4]]


def test_e6_trim_preserves_tiny_model_outputs_on_q1_samples():
    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="train")
    # Select short structures so the batch is guaranteed to crop below 82 rows.
    lengths = (ds.elements[:, 2:] != 0).sum(dim=1)
    ids = torch.nonzero(lengths <= 2).flatten()[:2].tolist()
    assert len(ids) == 2
    static = torch.utils.data._utils.collate.default_collate([ds[i] for i in ids])
    trimmed = trim_atom_padding_collate([ds[i] for i in ids])
    assert trimmed[0].shape[1] < static[0].shape[1] == 82
    torch.manual_seed(42)
    model = Transformer(token_num=128, d_model=32, nhead=4, edos_num=8, phdos_num=4,
                        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
                        dropout=0.0, activation="gelu", normalize_before=False,
                        decoupled_decoder=False, use_gated_cross_attn=False,
                        head_type="legacy", predict_scale=False)
    model.eval()
    with torch.no_grad():
        a = model(static[0], static[0].eq(0), static[1])
        b = model(trimmed[0], trimmed[0].eq(0), trimmed[1])
    for key in ("edos", "phdos"):
        # Changing a masked attention matrix's length can alter floating-point
        # reduction order; semantic equivalence is therefore numerical, not
        # bitwise.  The observed FP32 margin is below 5e-7.
        assert (a[key] - b[key]).abs().max().item() < 1e-5, key


def test_e6_config_default_and_q1_crop():
    assert not ExperimentConfig().use_bucket_batch
    assert ExperimentConfig(use_bucket_batch=True).use_bucket_batch
    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="train")
    lengths = (ds.elements[:, 2:] != 0).sum(dim=1)
    ids = torch.argsort(lengths)[:32].tolist()
    batch = trim_atom_padding_collate([ds[i] for i in ids])
    assert batch[0].shape[1] == 2 + int(lengths[ids].max())
    assert batch[14] is not None


def test_e6_q1_bucket_train_step_keeps_h1_finite():
    from model.model import basemodel
    from torch.utils.data import DataLoader
    logger = logging.getLogger("e6-test")
    logger.addHandler(logging.NullHandler()) if not logger.handlers else None
    params = dict(
        loss_form="sumnorm_klw", use_mask=False, lambda_ph=1.0, grad_clip=0.0,
        w_w1=1.0, w_huber=1.0, huber_delta=0.02, eta_sup_w=1.0,
        delta_edos=0.09375, delta_phdos=19.6875, metrics_list=[],
        sub_model=dict(transformer=dict(
            token_num=118, d_model=32, nhead=4, edos_num=128, phdos_num=64,
            num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64,
            dropout=0.0, activation="gelu", normalize_before=False,
            decoupled_decoder=False, use_gated_cross_attn=False, head_type="legacy",
            predict_scale=False, scale_mode="eta", atom_feat_mode="legacy3")),
        optimizer=dict(transformer=dict(type="AdamW", params=dict(lr=5e-5))), lr_scheduler={})
    model = basemodel(logger, **params)
    model.to(torch.device("cpu"))
    ds = Dos_Dataset(data_dir="./data/train4ARPAT", split="train", dos_sumnorm=True)
    lengths = (ds.elements[:, 2:] != 0).sum(dim=1)
    sampler = LengthBucketBatchSampler(SequentialSampler(ds), lengths, batch_size=2, window_batches=20)
    batch = next(iter(DataLoader(ds, batch_sampler=sampler, collate_fn=trim_atom_padding_collate)))
    result = model.train_one_step(batch, 0)
    assert batch[0].shape[1] < 82 and result["loss_eta"] > 0
    assert all(math.isfinite(value) for value in result.values())


class TestE6BucketBatch(unittest.TestCase):
    def test_coverage(self):
        test_e6_bucket_coverage_and_determinism()

    def test_forward(self):
        test_e6_trim_preserves_tiny_model_outputs_on_q1_samples()

    def test_config_crop(self):
        test_e6_config_default_and_q1_crop()

    def test_q1_train(self):
        test_e6_q1_bucket_train_step_keeps_h1_finite()


if __name__ == "__main__":
    unittest.main()
