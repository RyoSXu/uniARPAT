"""Supervisor gates and completion checks without running training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.eval.dos_baseline_execute import supervise
from utils.paths import file_sha256
from utils.experiment_config import ExperimentConfig
from utils.spectral_balance import BALANCE_SCHEMA, BALANCE_SCHEMA_V2
from utils.spectral_balance_experiment import legacy_balance_config


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.history = self.root / 'history.csv'
        self.source = self.root / 'source'
        self.source.write_text('frozen')
        self.report = dict(command=['unused'], environment={}, run_dir=str(self.run), configuration={'epochs': 100},
                           history=str(self.history), input_sha256={str(self.source): file_sha256(self.source)})
        (self.root / 'report.json').write_text(json.dumps(self.report))
        self.check = dict(passed=True, preparation_sha256=file_sha256(self.root / 'report.json'))
        self.save_check()

    def tearDown(self):
        self.tmp.cleanup()

    def save_check(self):
        (self.root / 'preflight.json').write_text(json.dumps(self.check))

    def test_failed_preflight_never_launches(self):
        self.check['passed'] = False
        self.save_check()
        with patch('tools.eval.dos_baseline_execute.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'preflight'):
                supervise(self.root)
            launch.assert_not_called()

    def test_changed_source_never_launches(self):
        self.source.write_text('changed')
        with patch('tools.eval.dos_baseline_execute.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                supervise(self.root)
            launch.assert_not_called()

    def test_completion_ignores_old_logs_and_records_state(self):
        execution = self.root / 'execution'
        execution.mkdir()
        (execution / 'old.log').write_text('CUDA out of memory')

        def launch(*args, **kwargs):
            self.run.mkdir()
            for name in ('checkpoint_latest.pth', 'checkpoint_best.pth'):
                (self.run / name).touch()
            self.history.write_text('epoch,balanced_score\n100,0.2\n')
            return Mock(pid=12345, poll=Mock(return_value=0))

        with patch('tools.eval.dos_baseline_execute.subprocess.Popen', side_effect=launch), \
                patch('tools.eval.dos_baseline_execute.signal.signal'):
            supervise(self.root)
        state = json.loads((execution / 'state.json').read_text())
        self.assertEqual(state['status'], 'training_complete')
        self.assertEqual(state['completed_epochs'], 100)

    def test_unrequested_resume_never_launches(self):
        self.run.mkdir()
        (self.run / 'checkpoint_latest.pth').touch()
        with patch('tools.eval.dos_baseline_execute.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'resume'):
                supervise(self.root)
            launch.assert_not_called()

    def test_exit_after_35_epochs_cannot_complete_100_epoch_protocol(self):
        def launch(*args, **kwargs):
            self.run.mkdir()
            for name in ('checkpoint_latest.pth', 'checkpoint_best.pth'):
                (self.run / name).touch()
            self.history.write_text('epoch,balanced_score\n35,0.2\n')
            return Mock(pid=12345, poll=Mock(return_value=0))
        with patch('tools.eval.dos_baseline_execute.subprocess.Popen', side_effect=launch), \
                patch('tools.eval.dos_baseline_execute.signal.signal'):
            with self.assertRaisesRegex(RuntimeError, '100 completed epochs'):
                supervise(self.root)
        self.assertEqual(json.loads((self.root/'execution/state.json').read_text())['status'], 'stopped_error')


class ComputationBudgetTests(unittest.TestCase):
    def test_v2_backbone_reference_keeps_actual_training_budget_separate(self):
        cfg = ExperimentConfig(epochs=50, schedule_epochs=100, pair_budget=51200,
                               spectral_balance_config='unit.json')
        with patch('utils.spectral_balance_experiment.load_balance_protocol',
                   return_value={'schema': BALANCE_SCHEMA_V2}):
            legacy = legacy_balance_config(cfg)
            self.assertIsNone(legacy.pair_budget)
            self.assertEqual(legacy.epochs, 35)
            self.assertIsNone(legacy.schedule_epochs)
        self.assertEqual(cfg.pair_budget, 51200)
        self.assertEqual(cfg.epochs, 50)
        self.assertEqual(cfg.schedule_epochs, 100)

    def test_v1_still_exposes_nondefault_budget_to_frozen_validator(self):
        cfg = ExperimentConfig(epochs=100, pair_budget=51200, spectral_balance_config='unit.json')
        with patch('utils.spectral_balance_experiment.load_balance_protocol',
                   return_value={'schema': BALANCE_SCHEMA}):
            legacy = legacy_balance_config(cfg)
            self.assertEqual(legacy.pair_budget, 51200)
            self.assertEqual(legacy.epochs, 100)

    def test_100_epoch_scheduler_retains_learning_rate_beyond_old_budget(self):
        import torch
        from utils.builder import build_warmup_cosine_scheduler
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=5e-5)
        scheduler = build_warmup_cosine_scheduler(optimizer, 100)
        rates = []
        for _ in range(100):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(rates[0], 1e-5)
        self.assertAlmostEqual(rates[5], 5e-5)
        self.assertGreater(rates[35], 3e-5)
        self.assertAlmostEqual(rates[47], 2.993157309113669e-05)
        self.assertGreater(rates[49], 2e-5)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-6)
        short = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=5e-5)
        short_scheduler = build_warmup_cosine_scheduler(short, 50)
        for _ in range(50):
            short.step()
            short_scheduler.step()
        self.assertAlmostEqual(short.param_groups[0]['lr'], 1e-6)


if __name__ == '__main__':
    unittest.main()
