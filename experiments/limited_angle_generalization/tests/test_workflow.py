import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark as b
from datasets import generate_dataset, seal, verify_dataset, shared_directory
from run_cases import execute_job
try:
    import numpy as np
except ImportError:
    np = None


class JobTest(unittest.TestCase):
    def test_child_failure_is_recorded_and_retry_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'job'
            with self.assertRaisesRegex(ValueError, 'subprocess failed'):
                execute_job(directory, [sys.executable, '-c', 'import sys; print("diagnostic"); sys.exit(7)'], {}, lambda: {}, root=Path(tmp))
            state = json.loads((directory / 'state.json').read_text())
            self.assertEqual((state['status'], state['returncode']), ('failed', 7))
            self.assertIn('diagnostic', (directory / 'process.log').read_text())
            with self.assertRaises(FileExistsError):
                execute_job(directory, [], {}, lambda: {}, root=Path(tmp))

    def test_exit_zero_with_missing_artifacts_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'job'
            def check():
                raise ValueError('missing artifact')
            with self.assertRaisesRegex(ValueError, 'missing artifact'):
                execute_job(directory, [sys.executable, '-c', 'pass'], {}, check, root=Path(tmp))
            state = json.loads((directory / 'state.json').read_text())
            self.assertEqual((state['status'], state['returncode']), ('failed', 0))


@unittest.skipIf(np is None, 'NumPy required')
class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/development.json').read_text())
        source = self.root / config['volumes'][0]['source']
        source.mkdir(parents=True)
        np.save(source / 'vol.npy', np.arange(120, dtype=np.float32).reshape(4,5,6)/120)
        meta = {'vol': 'vol.npy', 'scanner': {'mode':'cone','DSD':7.,'DSO':5.,'nDetector':[2,3],
            'sDetector':[4.,6.],'nVoxel':[4,5,6],'sVoxel':[2.,3.,4.],'offOrigin':[0,0,0],
            'offDetector':[0,0],'accuracy':0.5,'filter':None}}
        (source / 'meta_data.json').write_text(json.dumps(meta))
        with patch.object(b, 'provenance', return_value={'source_files_sha256':{}}):
            self.manifest = b.build_manifest(config, self.root)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def projector(self, volume, scanner, angles):
        self.calls.append(len(angles))
        return np.asarray([(2 + np.cos(a))*np.ones(scanner['nDetector'])*volume.mean() for a in angles], dtype=np.float32)

    def generate(self, case):
        return generate_dataset(self.manifest, case, self.root, self.projector)

    def test_pairing_reuses_shared_projections_and_refuses_overwrite(self):
        a,bcase = self.manifest['cases'][:2]
        self.generate(a)
        count = len(self.calls)
        self.generate(bcase)
        # Second case: only one reference check and five training batches, no evaluation regeneration.
        self.assertEqual(len(self.calls)-count, 6)
        self.assertTrue(all(n <= 10 for n in self.calls))
        for case in (a,bcase):
            verify_dataset(self.manifest, case, self.root)
        with self.assertRaises(ValueError):
            self.generate(a)

    def test_shared_code_identity_change_rejected(self):
        case = self.manifest['cases'][0]
        self.generate(case)
        self.manifest['provenance']['source_files_sha256']['changed.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'shared acquisition identity'):
            self.generate(self.manifest['cases'][1])

    def test_projection_corruption_detected(self):
        case = self.manifest['cases'][0]
        self.generate(case)
        path = self.root / case['data_directory'] / 'proj_train/proj_train_0000.npy'
        path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_dataset(self.manifest, case, self.root)

    def test_metadata_radians_tampering_rejected(self):
        case = self.manifest['cases'][0]
        self.generate(case)
        path = self.root / case['data_directory'] / 'meta_data.json'
        meta = json.loads(path.read_text()); meta['proj_train'][0]['angle'] = 90
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'radian angles'):
            verify_dataset(self.manifest, case, self.root)

    def test_incomplete_shared_group_is_not_reused(self):
        case = self.manifest['cases'][0]
        shared_directory(case, self.root).mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            self.generate(case)
        self.assertFalse((self.root / case['data_directory']).exists())

    def test_resealed_wrong_reference_rejected(self):
        case = self.manifest['cases'][0]
        self.generate(case)
        directory = self.root / case['data_directory']
        np.save(directory / 'vol_gt.npy', np.zeros((4,5,6), dtype=np.float32))
        (directory / 'dataset.sha256').unlink()
        seal(directory)
        with self.assertRaisesRegex(ValueError, 'reference hash mismatch'):
            verify_dataset(self.manifest, case, self.root)


@unittest.skipIf(np is None, 'NumPy required')
class MetricsTest(unittest.TestCase):
    def test_physical_gradient_and_no_clipping(self):
        from evaluate_cases import volume_metrics
        ref = np.arange(4, dtype=np.float32)[:,None,None]*np.ones((4,5,6), dtype=np.float32)/4
        result = volume_metrics(ref, ref + .1, [.25,.5,1])
        self.assertAlmostEqual(result['psnr_peak_1'], 20, places=4)
        self.assertAlmostEqual(result['boundary_mae'], .1, places=6)
        self.assertLess(result['boundary_gradient_error'], 1e-6)
        stretched = volume_metrics(ref, ref*2, [.25,.5,1])
        self.assertAlmostEqual(stretched['boundary_gradient_axis_error'][0], 1)

    def test_constant_reference_has_no_boundary_and_perfect_psnr_is_explicit(self):
        from evaluate_cases import volume_metrics
        ref = np.zeros((3,4,5), dtype=np.float32)
        result = volume_metrics(ref, ref, [1,1,1])
        self.assertEqual(result['boundary_voxels'], 0)
        self.assertIsNone(result['boundary_mae'])
        self.assertIsNone(result['psnr_peak_1'])
        self.assertTrue(result['perfect_match'])
        json.dumps(result, allow_nan=False)
