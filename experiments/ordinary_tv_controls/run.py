"""Paired ordinary-TV controls; baseline artifacts are immutable inputs."""
import argparse
import copy
import json
import re
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/limited_angle_generalization'))
from benchmark import validate_manifest, provenance, sha256, require, write_new_json
from datasets import verify_dataset, load_array
from run_cases import completed_stage, initializer_check, final_check, execute_job, environment
from evaluate_cases import volume_metrics


def baseline_manifest(path):
    # Added control scripts are allowed; every previously frozen source remains exact.
    m = validate_manifest(json.loads(path.read_text()), check_files=False)
    now = provenance(REPO)
    require(now['submodules'] == m['provenance']['submodules'], 'submodules changed')
    for name, value in m['provenance']['source_files_sha256'].items():
        require(sha256(REPO / name) == value, 'frozen baseline source changed: ' + name)
    return m


def control_case(base, weight, attempt=None):
    require(base['population_role'] == base['acquisition_role'] == 'development', 'reserved case')
    require(base['case_id'] in ('chest_start0_span120', 'chest_start90_span120'), 'unsupported case')
    require(weight in ('0.025', '0.1'), 'unsupported TV control')
    case = copy.deepcopy(base)
    case['method']['lambda_tv'] = float(weight)
    case['run_id'] = base['case_id'] + '_ordinary_tv_' + weight.replace('.', 'p') + '_s0_i30000'
    if attempt is not None:
        require(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', attempt) is not None, 'invalid attempt label')
        case['run_id'] += '__' + attempt
    case['output_directory'] = 'output/ordinary_tv_controls/' + case['run_id']
    command = list(base['planned_commands']['train'])
    command[command.index('--model_path') + 1] = case['output_directory']
    command[command.index('--lambda_tv') + 1] = weight
    return case, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--weight', required=True, choices=('0.025', '0.1'))
    parser.add_argument('--attempt', help='Fresh attempt label; preserves earlier outputs and starts from the original initializer')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = baseline_manifest(REPO / 'experiments/limited_angle_generalization/manifests/development_v4.json')
    base = next(c for c in manifest['cases'] if c['case_id'] == args.case)
    case, command = control_case(base, args.weight, args.attempt)
    checks = verify_dataset(manifest, base)
    init = initializer_check(manifest, base)
    require(init == completed_stage(base, 'initialize', manifest)['artifacts'], 'initializer changed')
    require(final_check(manifest, base) == completed_stage(base, 'train', manifest)['artifacts'], 'baseline changed')
    directory = REPO / 'output/ordinary_tv_control_jobs' / case['run_id']
    output = REPO / case['output_directory']
    require(not directory.exists() and not output.exists(), 'existing control; no retry/resume')
    command = [sys.executable, '-u'] + command[1:]
    print(json.dumps({'case': case, 'command': command, 'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    context = {'manifest': manifest, 'case': case, 'baseline_case_id': base['case_id'],
               'dataset_checks': checks, 'initializer': init, 'environment': environment(),
               'provenance': provenance(REPO), 'control_script_sha256': sha256(Path(__file__)),
               'protocol_sha256': sha256(Path(__file__).with_name('README.md'))}
    def postcheck():
        artifacts = final_check(manifest, case)
        require(verify_dataset(manifest, base) == checks, 'dataset changed during training')
        require(initializer_check(manifest, base) == init, 'initializer changed during training')
        geometry = manifest['volumes'][base['volume_id']]['geometry']
        ref = load_array(output / 'point_cloud/iteration_30000/vol_gt.npy', geometry['nVoxel'])
        pred = load_array(output / 'point_cloud/iteration_30000/vol_pred.npy', geometry['nVoxel'])
        metrics = volume_metrics(ref, pred, geometry['dVoxel'])
        # Native final SSIM uses the same untouched evaluator as the baseline.
        import yaml
        native = yaml.safe_load((output / 'eval/iter_030000/eval3d.yml').read_text())
        require(abs(metrics['psnr_peak_1'] - native['psnr_3d']) < 1e-3, 'PSNR mismatch')
        metrics.update(slice_aggregated_ssim=native['ssim_3d'], lambda_tv=float(args.weight),
                       case_id=base['case_id'], iteration=30000, initializer_sha256=init['initializer_sha256'],
                       reference_sha256=sha256(output / 'point_cloud/iteration_30000/vol_gt.npy'),
                       prediction_sha256=sha256(output / 'point_cloud/iteration_30000/vol_pred.npy'))
        write_new_json(directory / 'metrics.json', metrics)
        return dict(artifacts, metrics_sha256=sha256(directory / 'metrics.json'))
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(execute_job(directory, command, context, postcheck), indent=2))


if __name__ == '__main__':
    main()
