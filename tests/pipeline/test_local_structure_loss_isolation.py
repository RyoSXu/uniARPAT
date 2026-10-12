"""Objective isolation and matched dropout using the real production loss path."""
import logging
import unittest

import torch

from model.model import basemodel
from tools.eval.local_structure_loss_isolation import objective_step


def make_model(objective, dropout=.05):
    torch.manual_seed(42)
    model = basemodel(logging.getLogger("loss-isolation-test"),
        loss_form="sumnorm_klw", metrics_list=[],
        sub_model={"transformer": {
            "token_num": 128, "d_model": 32, "nhead": 4, "edos_num": 8, "phdos_num": 4,
            "num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 64,
            "dropout": dropout, "activation": "gelu", "scale_mode": "eta"}},
        optimizer={"transformer": {"type": "AdamW", "params": {"lr": 5e-5}}},
        lr_scheduler={})
    model.to(torch.device("cpu"))
    model.pair_aux_arm = "control" if objective == "joint" else "candidate"
    model.pair_aux_calibrated, model.pair_lambda = True, 3.677382707595825
    model.model["transformer"].train()
    return model


def make_batch():
    src = torch.tensor([[126, 127, 14, 8, 0, 0], [126, 127, 14, 8, 0, 0]])
    pos = torch.zeros(2, 6, 3)
    pos[:, 0] = torch.tensor([3., 3., .25])
    pos[:, 1] = torch.tensor([90., 90., 90.])
    pos[:, 2:4] = torch.tensor([[[.1, .2, .3], [.4, .5, .6]],
                              [[.3, .2, .1], [.7, .3, .4]]])
    edos = torch.zeros(2, 8)
    edos[0, 1], edos[1, 4] = 1., 1.
    phdos = torch.zeros(2, 4)
    phdos[0, 0], phdos[1, 1] = 1., 1.
    zero, one = torch.zeros(2, 1), torch.ones(2, 1)
    return [src, pos, edos, phdos, zero, one, zero, one,
            zero, one, zero, one, torch.ones(2, 8, dtype=torch.bool),
            torch.ones(2, 4, dtype=torch.bool), torch.full((2,), 8.)]


class LossIsolationTest(unittest.TestCase):
    def test_pair_only_update_matches_independent_pair_gradient(self):
        got, expected = make_model("pair_only"), make_model("pair_only")
        batch = make_batch()
        torch.manual_seed(81)
        values = objective_step(got, batch, 1, 72, "pair_only", audit=True)
        self.assertEqual(values["removed_joint_gradient_max"], 0.)
        torch.manual_seed(81)
        data = expected.data_preprocess(batch)
        # Consume the first forward's dropout without connecting its labels.
        with torch.no_grad():
            expected.model["transformer"](data[0], data[2], data[1], data[-2], data[-1])
        for p in expected.model["transformer"].parameters():
            p.grad = torch.zeros_like(p)
        pair, _, _, _ = expected._pair_forward_loss(batch)
        (expected.pair_lambda * pair).backward()
        expected.optimizer["transformer"].step()
        self.assertAlmostEqual(values["loss"], float(expected.pair_lambda * pair.detach()))
        for (name, p), (_, q) in zip(got.model["transformer"].named_parameters(),
                                    expected.model["transformer"].named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-8, msg=name)
            torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-8, msg=name)

    def test_removed_phdos_and_scale_labels_do_not_change_pair_only_updates(self):
        left, right = make_model("pair_only"), make_model("pair_only")
        original, changed = make_batch(), make_batch()
        changed[3] = changed[3].flip(-1)
        changed[7] = changed[7] * 10
        changed[11] = changed[11] * .001
        torch.manual_seed(93)
        a = objective_step(left, original, 1, 72, "pair_only", audit=True)
        torch.manual_seed(93)
        b = objective_step(right, changed, 1, 72, "pair_only", audit=True)
        self.assertNotAlmostEqual(a["joint_loss_unweighted"], b["joint_loss_unweighted"])
        self.assertEqual(a["loss"], b["loss"])
        for key, value in left.model["transformer"].state_dict().items():
            self.assertTrue(torch.equal(value, right.model["transformer"].state_dict()[key]), key)

    def test_joint_objective_responds_to_phdos_labels(self):
        left, right = make_model("joint"), make_model("joint")
        original, changed = make_batch(), make_batch()
        changed[3] = changed[3].flip(-1)
        torch.manual_seed(93)
        a = objective_step(left, original, 1, 72, "joint")
        torch.manual_seed(93)
        b = objective_step(right, changed, 1, 72, "joint")
        self.assertNotAlmostEqual(a["loss"], b["loss"])
        self.assertTrue(any(not torch.equal(p, right.model["transformer"].state_dict()[key])
                            for key, p in left.model["transformer"].state_dict().items()))

    def test_both_objectives_consume_identical_dropout_streams(self):
        joint, pair_only = make_model("joint"), make_model("pair_only")
        torch.manual_seed(81)
        objective_step(joint, make_batch(), 1, 72, "joint")
        joint_rng = torch.get_rng_state()
        torch.manual_seed(81)
        objective_step(pair_only, make_batch(), 1, 72, "pair_only")
        self.assertTrue(torch.equal(joint_rng, torch.get_rng_state()))
        for model in (joint, pair_only):
            self.assertTrue(all(p.requires_grad for p in model.model["transformer"].parameters()))
            self.assertTrue(all(int(state["step"]) == 1
                                for state in model.optimizer["transformer"].state.values()))

    def test_invalid_objective_and_amp_are_rejected_before_update(self):
        model = make_model("pair_only")
        with self.assertRaises(ValueError):
            objective_step(model, make_batch(), 1, 72, "unknown")
        model.use_amp = True
        with self.assertRaises(ValueError):
            objective_step(model, make_batch(), 1, 72, "pair_only")
        self.assertFalse(model.optimizer["transformer"].state)


if __name__ == "__main__":
    unittest.main()
