"""Location, historical authenticity and live-source migration contracts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from utils.paths import (
    PROJECT_ROOT, Q1_DATA_ROOT, historical_path, input_path, protected_path,
    repository_path,
)


class TestRepositoryPaths(unittest.TestCase):
    def test_builtin_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(input_path('./data/train4ARPAT'), Q1_DATA_ROOT)
                self.assertEqual(repository_path('configs/default.yaml'), PROJECT_ROOT / 'tools/training-defaults.yaml')
                self.assertEqual(input_path('output/run/checkpoint.pth'), PROJECT_ROOT / 'results/runs/run/checkpoint.pth')
                self.assertEqual(repository_path('data/grids_c2b/grids.json'), PROJECT_ROOT / 'data/reference/grids.json')
            finally:
                os.chdir(previous)

    def test_canonical_generated_locations_are_idempotent(self):
        for name in ('results/reports/new/report.csv', 'results/runs/new/checkpoint.pth', 'results/figures/new.png'):
            path = repository_path(name)
            self.assertEqual(path, PROJECT_ROOT / name)
            self.assertEqual(repository_path(path), path)
        self.assertEqual(repository_path('results/new/report.csv'), PROJECT_ROOT / 'results/reports/new/report.csv')

    def test_second_migration_resolves_old_run_directory_and_stays_idempotent(self):
        path = repository_path('output/ablation_m1_g2edge')
        self.assertEqual(path, PROJECT_ROOT / 'archive/results/runs/ablation_m1_g2edge')
        self.assertEqual(repository_path(path), path)
        current = repository_path('output/ablation_m1_e9ctl')
        self.assertEqual(current, PROJECT_ROOT / 'results/runs/ablation_m1_e9ctl')

    def test_historical_report_prefix_keeps_its_suffixes_in_archive(self):
        prefix = repository_path('results/reports/g2_small_fit_q1')
        self.assertEqual(prefix, PROJECT_ROOT / 'archive/results/reports/g2_small_fit_q1')
        self.assertTrue(prefix.with_suffix('.json').is_file())
        self.assertEqual(prefix.with_name(prefix.name + '_samples.csv'),
                         repository_path('results/reports/g2_small_fit_q1_samples.csv'))

    def test_custom_user_paths_keep_their_meaning(self):
        self.assertEqual(input_path('custom/data'), Path('custom/data'))
        self.assertEqual(input_path('../weights/checkpoint.pth'), Path('../weights/checkpoint.pth'))
        self.assertEqual(input_path('/tmp/custom/checkpoint.pth'), Path('/tmp/custom/checkpoint.pth'))

    def fixture(self, directory):
        root = Path(directory)
        (root / 'archive/metadata').mkdir(parents=True)
        current = root / 'utils/source.py'
        current.parent.mkdir()
        current.write_bytes(b'new-path source\n')
        original = root / 'saved-source.txt'
        original.write_bytes(b'original source\n')
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        after = hashlib.sha256(current.read_bytes()).hexdigest()
        (root / 'archive/metadata/repository-migration-2026-10-04.json').write_text(json.dumps({
            'path_moves': {'datasets/source.py': 'utils/source.py'}, 'prefix_moves': {},
            'source_versions': {'datasets/source.py': {before: 'saved-source.txt'}},
            'source_before_sha256': {'datasets/source.py': before},
            'source_after_sha256': {'datasets/source.py': after},
        }))
        return root, current, original, before, after

    def test_edited_source_uses_exact_original_for_historical_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root, current, original, before, after = self.fixture(directory)
            self.assertEqual(historical_path('datasets/source.py', before, root=root), original)
            self.assertEqual(historical_path('datasets/source.py', after, root=root), current)

    def test_missing_or_tampered_historical_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, current, original, before, after = self.fixture(directory)
            with self.assertRaises(ValueError):
                historical_path('datasets/source.py', '0' * 64, root=root)
            original.write_text('tampered')
            with self.assertRaises(ValueError):
                historical_path('datasets/source.py', before, root=root)

    def test_live_guard_accepts_only_recorded_migration_and_rejects_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root, current, original, before, after = self.fixture(directory)
            self.assertEqual(protected_path('datasets/source.py', before, root=root), current)
            current.write_text('unrecorded formula change')
            # The original is still authentic; that alone cannot authorize live code.
            self.assertEqual(historical_path('datasets/source.py', before, root=root), original)
            with self.assertRaises(ValueError):
                protected_path('datasets/source.py', before, root=root)

    def test_git_backed_original_is_verified_and_cache_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, current, original, before, after = self.fixture(directory)
            subprocess.run(['git', '-C', str(root), 'init', '--quiet'], check=True)
            source = root / 'original.py'
            source.write_bytes(original.read_bytes())
            subprocess.run(['git', '-C', str(root), 'add', 'original.py'], check=True)
            tree = subprocess.check_output(['git', '-C', str(root), 'write-tree']).decode().strip()
            manifest_path = root / 'archive/metadata/repository-migration-2026-10-04.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['source_versions'] = {}
            manifest['source_git_versions'] = {
                'datasets/source.py': {before: {'commit': tree, 'path': 'original.py'}},
            }
            manifest_path.write_text(json.dumps(manifest))
            original.unlink()
            restored = historical_path('datasets/source.py', before, root=root)
            self.assertEqual(hashlib.sha256(restored.read_bytes()).hexdigest(), before)
            restored.write_text('tampered cache')
            with self.assertRaises(ValueError):
                historical_path('datasets/source.py', before, root=root)

    def test_location_import_does_not_load_dataset_or_torch(self):
        code = "import sys; import utils.paths; assert 'utils.dataset' not in sys.modules; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
