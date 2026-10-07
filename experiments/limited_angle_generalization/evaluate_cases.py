"""Fixed-final CPU volume metrics, without clipping or per-image normalization."""
import argparse
import json
from pathlib import Path
import sys

from benchmark import REPO, local_path, require, sha256, validate_manifest, write_new_json
from datasets import load_array, verify_dataset


def volume_metrics(reference, prediction, spacing):
    import numpy as np
    require(reference.shape == prediction.shape and reference.ndim == 3 and min(reference.shape) > 1,
            "volume shape mismatch or too small")
    require(np.isfinite(reference).all() and np.isfinite(prediction).all(), "nonfinite volume")
    require(len(spacing) == 3 and all(np.isfinite(h) and h > 0 for h in spacing), "invalid spacing")
    require(reference.min() >= 0 and reference.max() <= 1, "unit-range metric requires a reference in [0,1]")
    error = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(error ** 2))
    # null plus perfect_match avoids nonstandard JSON Infinity.
    result = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(mse)),
              "psnr_peak_1": 10 * np.log10(1 / mse) if mse else None,
              "perfect_match": mse == 0, "psnr_peak": 1.0,
              "normalization": "none", "clipping": "none"}
    def gradient(array):
        return np.stack([(array[1:, :-1, :-1] - array[:-1, :-1, :-1]) / spacing[0],
                         (array[:-1, 1:, :-1] - array[:-1, :-1, :-1]) / spacing[1],
                         (array[:-1, :-1, 1:] - array[:-1, :-1, :-1]) / spacing[2]], axis=0)
    ref_gradient = gradient(reference.astype(np.float64))
    magnitude = np.linalg.norm(ref_gradient, axis=0)
    positive = magnitude[magnitude > 0]
    result.update(boundary_voxels=0, boundary_threshold=None, boundary_mae=None,
                  boundary_gradient_error=None, boundary_gradient_axis_error=None)
    if positive.size:
        threshold = float(np.quantile(positive, 0.9))
        mask = magnitude >= threshold
        difference = gradient(prediction.astype(np.float64)) - ref_gradient
        result.update(boundary_voxels=int(mask.sum()), boundary_threshold=threshold,
                      boundary_mae=float(np.mean(np.abs(error[:-1, :-1, :-1][mask]))),
                      boundary_gradient_error=float(np.mean(np.linalg.norm(difference, axis=0)[mask])),
                      boundary_gradient_axis_error=[float(np.mean(np.abs(x[mask]))) for x in difference])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "report exists")
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    matches = [c for c in manifest["cases"] if c["case_id"] == args.case]
    require(len(matches) == 1, "unknown case")
    case = matches[0]
    verify_dataset(manifest, case)
    # Direct invocation has the same completion/identity gate as the runner.
    from run_cases import completed_stage, final_check
    state = completed_stage(case, "train", manifest)
    require(state["artifacts"] == final_check(manifest, case), "final reconstruction changed")
    geometry = manifest["volumes"][case["volume_id"]]["geometry"]
    source = local_path(REPO, case["data_directory"]) / "vol_gt.npy"
    prediction = local_path(REPO, case["output_directory"]) / "point_cloud/iteration_30000/vol_pred.npy"
    ref = load_array(source, geometry["nVoxel"])
    pred = load_array(prediction, geometry["nVoxel"])
    metrics = volume_metrics(ref, pred, geometry["dVoxel"])
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    from r2_gaussian.utils.image_utils import metric_vol
    torch.set_num_threads(1)
    counts = [int(np.sum(np.max(ref, axis=tuple(i for i in range(3) if i != axis)) > 0)) for axis in range(3)]
    if all(counts):
        ssim, axes = metric_vol(np.asarray(ref), np.asarray(pred), "ssim")
        require(np.isfinite(ssim) and all(np.isfinite(axes)), "nonfinite SSIM")
    else:
        ssim, axes = None, [None, None, None]
    metrics.update(ssim_normalization="none; inherited volume SSIM skips zero-reference slices",
                   slice_aggregated_ssim=ssim, ssim_axis=axes, ssim_valid_slices=counts,
                   iteration=30000, case_id=args.case, manifest_sha256=manifest["manifest_sha256"],
                   reference_sha256=sha256(source), prediction_sha256=sha256(prediction),
                   gaussian_projection_metrics="not computed", tigre_projection_metrics="not computed")
    write_new_json(args.report, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
