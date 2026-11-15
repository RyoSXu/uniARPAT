"""v2 label contract: protocol fit on labelled rows, contract guards, W1 loss and per-source report."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from utils.partial_labels import FIELD, batch_labels, labelled_mean
from utils.paths import file_sha256
from utils.spectral_balance import (
    BALANCE_SCHEMA, balanced_checkpoint_eligible, fit_task_normalizer, load_balance_protocol,
    prepare_balance_protocol_v2, validate_balance_protocol,
)
from utils.source_report import grouped_report

ROOT = Path(__file__).resolve().parents[2]
S1_V2 = ROOT / 'data/spectral_store_v1/views/s1_v2'
SOURCES = {'none': 0, 'legacy': 1, 'mp': 2, 'mp_dfpt': 4, 'phonondb': 5, 'jarvis': 6}
LINKS = {'none': 0, 'hint': 1, 'hint_edge': 2, 'rehung': 3, 'own': 4}


def write_view(root, label_contract='v2', valid_has_phdos=(True, True)):
    """Tiny view: train row 2 is eDOS-only (phDOS zeros), row 3 phDOS-only; valid is dual."""
    rng = np.random.default_rng(0)
    arrays = {'train': {'index': np.array(['m0', 'm1', 'm2', 'm3']),
                        'n_atoms': np.array([2, 4, 1, 3], np.int64),
                        'has_edos': np.array([True, True, True, False]),
                        'has_phdos': np.array([True, True, False, True])},
              'valid': {'index': np.array(['v0', 'v1']), 'n_atoms': np.array([2, 2], np.int64),
                        'has_edos': np.array([True, True]), 'has_phdos': np.array(valid_has_phdos)},
              'dev': {'index': np.array(['d0', 'd1', 'd2']), 'n_atoms': np.array([1, 1, 1], np.int64),
                      'has_edos': np.array([True, True, False]),
                      'has_phdos': np.array([False, False, True])}}
    bins = {'edos': 4, 'phdos': 3}
    files = {}
    for split, values in arrays.items():
        (root / split).mkdir(parents=True)
        n = len(values['index'])
        for task, width in bins.items():
            target = rng.uniform(.1, 2., (n, width)) * values[f'has_{task}'][:, None]
            values[f'{task}_tgtdos'] = target
            values[f'source_{task}'] = np.where(values[f'has_{task}'], SOURCES['mp'], 0).astype(np.int8)
            values[f'link_{task}'] = np.where(values[f'has_{task}'], LINKS['own'], 0).astype(np.int8)
            values[f'disputed_{task}'] = np.zeros(n, bool)
            values[f'volume_flag_{task}'] = np.zeros(n, bool)
        values['is_entry'] = np.zeros(n, bool)
        for key, value in values.items():
            relative = f'{split}/{split}_index.npy' if key == 'index' else f'{split}/{key}_{split}.npy'
            np.save(root / relative, value)
            files[relative] = {'sha256': file_sha256(root / relative)}
    manifest = {'schema': 'crystal-q1-v1', 'partial_labels': True, 'splits': list(arrays),
                'grid': {'edos': {'range': [-2., 2.], 'bins': 4}, 'phdos': {'range': [0., 3.], 'bins': 3}},
                'source_codes': SOURCES, 'link_codes': LINKS, 'files': files}
    if label_contract is not None:
        manifest['label_contract'] = label_contract
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return arrays


class ProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'view'
        self.arrays = write_view(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalizer_is_fitted_on_labelled_rows_only(self):
        protocol = prepare_balance_protocol_v2(self.root, 'valid')
        train = self.arrays['train']
        self.assertEqual(protocol['fit_samples'], {'edos': 3, 'phdos': 3})
        for task, width in (('edos', 1.), ('phdos', 1.)):
            has = train[f'has_{task}']
            expected = fit_task_normalizer(train[f'{task}_tgtdos'][has], train['n_atoms'][has], width)
            self.assertAlmostEqual(protocol['tasks'][task]['energy'], expected['energy'], places=12)
        self.assertEqual(protocol['reference']['mode'], 'none')
        self.assertTrue(balanced_checkpoint_eligible(
            {'selection_samples': 2, 'blind_fail_rate_edos': 0., 'blind_fail_rate_phdos': 50.}, protocol))

    def test_saved_protocol_reloads_and_detects_changed_data(self):
        path = Path(self.tmp.name) / 'protocol.json'
        path.write_text(json.dumps(prepare_balance_protocol_v2(self.root, 'valid')))
        self.assertEqual(load_balance_protocol(path, self.root)['view']['valid_split'], 'valid')
        np.save(self.root / 'train/n_atoms_train.npy', np.array([2, 4, 1, 4], np.int64))
        with self.assertRaisesRegex(ValueError, 'payload differs'):
            load_balance_protocol(path, self.root)

    def test_contract_mismatch_is_rejected_in_both_directions(self):
        path = Path(self.tmp.name) / 'protocol.json'
        path.write_text(json.dumps(prepare_balance_protocol_v2(self.root, 'valid')))
        v1_view = Path(self.tmp.name) / 'v1'
        write_view(v1_view, label_contract=None)
        with self.assertRaisesRegex(ValueError, 'label contract mismatch'):
            load_balance_protocol(path, v1_view)
        v1 = {'schema': BALANCE_SCHEMA, 'label_contract': 'v2'}
        with self.assertRaises(ValueError):
            validate_balance_protocol(v1)

    def test_selection_split_must_be_public_and_carry_both_tasks(self):
        self.assertEqual(prepare_balance_protocol_v2(self.root, 'dev')['reference']['samples'], 3)
        single = Path(self.tmp.name) / 'single'
        write_view(single, valid_has_phdos=(False, False))
        with self.assertRaisesRegex(ValueError, 'no phdos labels'):
            prepare_balance_protocol_v2(single, 'valid')
        with self.assertRaisesRegex(ValueError, 'unknown public valid split'):
            prepare_balance_protocol_v2(self.root, 'test')
        with self.assertRaisesRegex(ValueError, 'unknown public valid split'):
            prepare_balance_protocol_v2(self.root, 'train')

    def test_fixed_reference_requires_label_and_counts(self):
        protocol = prepare_balance_protocol_v2(self.root, 'valid')
        protocol['reference'] = {**protocol['reference'], 'mode': 'fixed'}
        with self.assertRaises(ValueError):
            validate_balance_protocol(protocol)
        protocol['reference'].update(label='h2-v2', failures_edos=0, failures_phdos=1,
                                     r2_edos_median=.5, r2_phdos_median=.6)
        validate_balance_protocol(protocol)


class LabelledLossTests(unittest.TestCase):
    def test_unlabelled_rows_contribute_neither_value_nor_gradient(self):
        values = torch.tensor([1., 100., 3.], requires_grad=True)
        labels = torch.tensor([True, False, True])
        loss = labelled_mean(values, labels)
        self.assertAlmostEqual(float(loss), 2.)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([.5, 0., .5]))
        self.assertEqual(float(labelled_mean(values, torch.zeros(3, dtype=torch.bool))), 0.)

    def test_chunked_accumulation_with_batch_denominator_matches_full_batch(self):
        torch.manual_seed(0)
        values, labels = torch.rand(7), torch.rand(7) > .4
        full = labelled_mean(values, labels)
        count = int(labels.sum())
        chunks = sum(labelled_mean(values[a:b], labels[a:b], count) for a, b in ((0, 3), (3, 5), (5, 7)))
        torch.testing.assert_close(chunks, full)

    def test_w1_weights_tasks_equally_regardless_of_label_counts(self):
        edos = torch.tensor([2., 2., 2., 2.])
        phdos = torch.tensor([4., 0., 0., 0.])
        labels = torch.tensor([[True, True], [True, False], [True, False], [True, False]])
        loss = .5 * labelled_mean(edos, labels[:, 0]) + .5 * labelled_mean(phdos, labels[:, 1])
        self.assertAlmostEqual(float(loss), 3.)

    def test_batch_flags_are_optional_and_strictly_typed(self):
        batch = [torch.zeros(2, 3)] + [None] * (FIELD - 1)
        self.assertIsNone(batch_labels(batch))
        self.assertIsNone(batch_labels(batch + [None]))
        batch.append(torch.ones(2, 2, dtype=torch.bool))
        self.assertEqual(batch_labels(batch).shape, (2, 2))
        for bad in (torch.ones(2, 2), torch.ones(3, 2, dtype=torch.bool)):
            batch[FIELD] = bad
            with self.assertRaises(ValueError):
                batch_labels(batch)


class SourceReportTests(unittest.TestCase):
    def test_groups_use_labelled_rows_and_require_complete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'views' / 'v'
            write_view(root)
            np.save(root / 'dev/disputed_edos_dev.npy', np.array([True, False, False]))
            samples = pd.DataFrame({'material_id': ['d0', 'd1', 'd2'],
                                    'blind_energy_edos': [1., 3., np.nan], 'blind_r2_edos': [.5, -.1, np.nan],
                                    'blind_energy_phdos': [np.nan, np.nan, 2.], 'blind_r2_phdos': [np.nan, np.nan, .9]})
            report = grouped_report(samples, root, 'dev')
            edos = report['tasks']['edos']
            self.assertEqual(edos['all'], {'n': 2, 'energy_mean': 2., 'energy_median': 2.,
                                           'r2_median': .2, 'fail_rate': 50.})
            self.assertEqual(edos['disputed']['n'], 1)
            self.assertEqual(edos['clean']['energy_mean'], 3.)
            self.assertEqual(edos['source=mp']['n'], 2)
            self.assertEqual(report['tasks']['phdos']['all']['n'], 1)
            with self.assertRaisesRegex(ValueError, 'cover the view split'):
                grouped_report(samples.iloc[:2], root, 'dev')
            with self.assertRaisesRegex(ValueError, 'lack scores'):
                grouped_report(samples.assign(blind_r2_edos=np.nan), root, 'dev')


@unittest.skipUnless((S1_V2 / 'manifest.json').exists(), 'built s1_v2 view not present')
class S1V2DatasetTests(unittest.TestCase):
    def test_dataset_rows_carry_label_flags_through_collate(self):
        from utils.bucket_batch import crystal_collate
        from utils.dataset import Dos_Dataset
        manifest = json.loads((S1_V2 / 'manifest.json').read_text())
        data = Dos_Dataset(data_dir=str(S1_V2), split='valid', view_split='dev_indep', dos_sumnorm=True,
                           edos_edges=None, phdos_edges=None)
        self.assertTrue(data.partial_labels)
        self.assertEqual(len(data), 42)
        batch = crystal_collate([data[i] for i in range(4)])
        self.assertIsNone(batch[FIELD - 1])
        flags = batch_labels(batch)
        torch.testing.assert_close(flags, torch.tensor([[False, True]] * 4))
        self.assertEqual(manifest['label_contract'], 'v2')
        with self.assertRaises(ValueError):
            Dos_Dataset(data_dir=str(S1_V2), split='train', view_split='dev')


if __name__ == '__main__':
    unittest.main()
