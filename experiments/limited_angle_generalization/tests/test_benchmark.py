import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

LAYER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAYER))
import benchmark as b


class AnglesTest(unittest.TestCase):
    def test_historical_grids_and_counts(self):
        evaluation = [(i + 0.5) * 3.6 for i in range(100)]
        for start, span, last, count in ((0, 360, 352.8, 100), (0, 120, 117.6, 33), (90, 120, 207.6, 33)):
            with self.subTest(start=start, span=span):
                angles = b.train_angles(start, span, 50)
                self.assertEqual(len(angles), 50)
                self.assertEqual(angles[0], start)
                self.assertAlmostEqual(angles[-1], last)
                self.assertEqual(sum(b.inside(a, start, span) for a in evaluation), count)
                self.assertGreater(b.min_separation(angles, evaluation), 0.59)
                self.assertAlmostEqual(math.degrees(math.radians(angles[-1])), last)

    def test_wraparound_half_open_tolerance(self):
        for angle in (330, 0, 45, 89.9, -30, 690, 330 - 1e-8):
            self.assertTrue(b.inside(angle, 330, 120), angle)
        for angle in (90, 90 - 1e-8, 329.9, 180):
            self.assertFalse(b.inside(angle, 330, 120), angle)
        self.assertTrue(b.inside(360, 0, 360))

    def test_circular_overlap_and_duplicate(self):
        self.assertEqual(b.min_separation([0], [360]), 0)
        self.assertAlmostEqual(b.min_separation([359.9], [0.1]), 0.2)
        with self.assertRaises(ValueError):
            b.validate_angles([0, 360], "test")

    def test_midpoints_collide_for_ninety_degree_case(self):
        training = [b.train_angles(0, 90, 50), b.train_angles(330, 120, 50)]
        midpoint = [(i + 0.5) * 3.6 for i in range(100)]
        self.assertLess(b.min_separation(training[0], midpoint), b.TOL_DEG)
        evaluation, offset = b.common_evaluation(training, 100)
        self.assertNotEqual(offset, 0.5)
        self.assertEqual((evaluation, offset), b.common_evaluation(training, 100))
        for angles in training:
            self.assertGreater(b.min_separation(angles, evaluation), b.TOL_DEG)

    def test_joint_coordinate_rotation_sector_equivariance(self):
        evaluation = [1, 89, 90, 200, 329, 330, 359]
        expected = [b.inside(a, 330, 120) for a in evaluation]
        for rotation in (-360, -91, 45, 360):
            self.assertEqual(expected, [b.inside(a + rotation, 330 + rotation, 120) for a in evaluation])

    def test_invalid_angles(self):
        for start, span, count in ((0, 0, 50), (0, 361, 50), (float("nan"), 120, 50),
                                   (0, float("inf"), 50), (0, 120, 0), (0, 120, 2.5), (0, 120, True)):
            with self.assertRaises(ValueError):
                b.train_angles(start, span, count)


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = json.loads((LAYER / "configs/development.json").read_text())
        source = self.root / self.config["volumes"][0]["source"]
        source.mkdir(parents=True)
        self.volume = source / "vol.npy"
        self.volume.write_bytes(b"tiny source fixture; hash validation does not parse npy")
        self.metadata = source / "meta_data.json"
        self.meta = {"vol": "vol.npy", "scanner": {
            "mode": "cone", "DSD": 7., "DSO": 5., "nDetector": [16, 24], "sDetector": [4., 6.],
            "nVoxel": [8, 10, 12], "sVoxel": [2., 3., 4.], "offOrigin": [0, 0, 0],
            "offDetector": [0, 0], "accuracy": 0.5, "filter": None}}
        self.metadata.write_text(json.dumps(self.meta))
        self.mock = patch.object(b, "provenance", return_value={"source_files_sha256": {}})
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.tmp.cleanup()

    def build(self):
        return b.build_manifest(self.config, self.root)

    def test_shared_evaluation_and_physical_geometry(self):
        manifest = self.build()
        self.assertEqual(b.validate_manifest(manifest, self.root), manifest)
        cases = manifest["cases"]
        self.assertEqual(len({c["evaluation_group"] for c in cases}), 1)
        self.assertEqual(len({c["source_sha256"] for c in cases}), 1)
        self.assertEqual(len({c["geometry_sha256"] for c in cases}), 1)
        self.assertEqual(manifest["volumes"]["0_chest_cone"]["geometry"]["dVoxel"], [0.25, 0.3, 1/3])
        for c in cases:
            self.assertEqual(c["inside_count"] + c["outside_count"], 100)
            self.assertEqual(c["train_radians"], [math.radians(a) for a in c["train_degrees"]])

    def test_duplicate_case_and_acquisition(self):
        self.config["acquisitions"].append(copy.deepcopy(self.config["acquisitions"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            self.build()
        self.config["acquisitions"][-1]["case_id"] = "alias"
        with self.assertRaisesRegex(ValueError, "duplicate acquisition"):
            self.build()

    def test_patient_group_leakage(self):
        other = dict(self.config["volumes"][0], volume_id="alias", population_role="final_test")
        self.config["volumes"].append(other)
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.build()

    def test_same_source_hash_leakage_under_new_identity(self):
        other = dict(self.config["volumes"][0], volume_id="alias", independence_group="invented", population_role="final_test")
        self.config["volumes"].append(other)
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.build()

    def test_manifest_tampering_even_with_recomputed_hash(self):
        manifest = self.build()
        manifest["cases"][0]["train_radians"][1] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            b.validate_manifest(manifest, self.root)
        manifest["manifest_sha256"] = b.digest({k: v for k,v in manifest.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "manifest/source mismatch"):
            b.validate_manifest(manifest, self.root)

    def test_changed_source_is_detected(self):
        manifest = self.build()
        self.volume.write_bytes(b"modified source")
        with self.assertRaisesRegex(ValueError, "manifest/source mismatch"):
            b.validate_manifest(manifest, self.root)

    def test_inconsistent_spacing_rejected(self):
        self.meta["scanner"]["dVoxel"] = [1, 1, 1]
        self.metadata.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "inconsistent dVoxel"):
            self.build()

    def test_unsupported_seed_or_geometry_term_rejected(self):
        for key, value in (("optimization_seed", 1), ("geometry_term", True)):
            saved = self.config["method"][key]
            self.config["method"][key] = value
            with self.assertRaises(ValueError):
                self.build()
            self.config["method"][key] = saved

    def test_no_planning_writes(self):
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        manifest = self.build()
        b.ensure_new_outputs(manifest, self.root)
        self.assertEqual(before, sorted(p.relative_to(self.root) for p in self.root.rglob("*")))

    def test_exclusive_write_and_existing_output_states(self):
        manifest = self.build()
        target = self.root / "manifest.json"
        b.write_new_json(target, manifest)
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            b.write_new_json(target, {})
        self.assertEqual(target.read_bytes(), original)
        output = self.root / manifest["cases"][0]["output_directory"]
        self.assertEqual(b.output_state(output), "new")
        output.mkdir(parents=True)
        self.assertIn("incomplete", b.output_state(output))
        with self.assertRaisesRegex(ValueError, "refusing existing"):
            b.ensure_new_outputs(manifest, self.root)
        (output / "ckpt").mkdir()
        (output / "ckpt/chkpnt5000.pth").touch()
        self.assertIn("resume_forbidden", b.output_state(output))
        (output / "eval/iter_030000").mkdir(parents=True)
        (output / "eval/iter_030000/eval3d.yml").touch()
        (output / "point_cloud/iteration_30000").mkdir(parents=True)
        (output / "point_cloud/iteration_30000/vol_pred.npy").touch()
        self.assertIn("completed_artifacts", b.output_state(output))
        with self.assertRaises(ValueError):
            b.ensure_new_outputs(manifest, self.root)

    def test_path_escape_rejected(self):
        for path in ("../outside", "/tmp/outside"):
            with self.assertRaises(ValueError):
                b.local_path(self.root, path)


class ImportTest(unittest.TestCase):
    def test_no_gpu_dependencies_on_import(self):
        code = "import benchmark,sys; assert not any(x in sys.modules for x in ('torch','tigre','numpy'))"
        result = subprocess.run([sys.executable, "-c", code], cwd=LAYER, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
