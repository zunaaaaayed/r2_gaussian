import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('controls', Path(__file__).with_name('run.py'))
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)

class PairingTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((controls.REPO / 'experiments/limited_angle_generalization/manifests/development_v4.json').read_text())
        self.base = self.manifest['cases'][1]

    def test_only_tv_and_output_arguments_change(self):
        original = copy.deepcopy(self.base)
        case, cmd = controls.control_case(self.base, '0.025')
        previous = self.base['planned_commands']['train']
        differing = [i for i,(a,b) in enumerate(zip(cmd,previous)) if a != b]
        self.assertEqual(differing, [previous.index('--model_path')+1, previous.index('--lambda_tv')+1])
        self.assertEqual(case['initializer'], self.base['initializer'])
        self.assertEqual(case['data_directory'], self.base['data_directory'])
        self.assertEqual(self.base, original)

    def test_reserved_acquisition_is_blocked(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            controls.control_case(self.manifest['cases'][3], '0.1')

    def test_unplanned_weight_is_blocked(self):
        with self.assertRaises(ValueError):
            controls.control_case(self.base, '0.2')

if __name__ == '__main__':
    unittest.main()
