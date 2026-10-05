"""Save pre-sampling FDK volumes and compare them with final R2 volumes."""

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from initialize_pcd import (
    ModelParams, Scene, get_geometry_tigre, recon_volume, t2a,
)
from generate import sha256


def metrics(prediction, reference, mask):
    error = (
        prediction[mask].astype(np.float64)
        - reference[mask].astype(np.float64)
    )
    mse = float(np.mean(error ** 2))
    return {
        "voxel_count": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "signed_bias": float(np.mean(error)),
        "absolute_error_p95": float(np.percentile(np.abs(error), 95)),
        "psnr_peak_1": float(-10 * np.log10(mse)) if mse > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/limited_angle_120_diagnostics/fdk"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    hashes = {}
    records = {}
    common_reference = None

    for arc in (360, 120):
        case = Path(
            f"data/trdp2/chest_50views_noisefree/arc_{arc}/0_chest_cone"
        )
        final_folder = Path(
            f"output/chest_50views_noisefree_arc{arc}"
        ) / "point_cloud/iteration_30000"
        metadata = json.loads((case / "meta_data.json").read_text())

        # Match initialize_pcd's Scene loading and training-camera order.
        model_parser = argparse.ArgumentParser()
        model_params = ModelParams(model_parser)
        model_args = model_params.extract(model_parser.parse_args([]))
        model_args.source_path = str(case)
        model_args.eval = False  # Do not load evaluation projections.

        print(f"\nLoading {arc}-degree training case...", flush=True)
        scene = Scene(model_args, shuffle=False)
        cameras = scene.getTrainCameras()
        assert len(cameras) == 50
        projections = np.concatenate(
            [t2a(cam.original_image) for cam in cameras], axis=0
        )
        angles = np.stack([t2a(cam.angle) for cam in cameras], axis=0)
        geometry = get_geometry_tigre(scene.scanner_cfg)

        print("Reconstructing raw FDK volume...", flush=True)
        fdk = recon_volume(
            projections, angles, copy.deepcopy(geometry), "fdk"
        )
        gt = t2a(scene.vol_gt)
        final = np.load(final_folder / "vol_pred.npy", allow_pickle=False)
        saved_gt = np.load(final_folder / "vol_gt.npy", allow_pickle=False)

        assert fdk.shape == final.shape == gt.shape
        assert np.array_equal(gt, saved_gt)
        assert np.isfinite(fdk).all() and np.isfinite(final).all()
        if common_reference is None:
            common_reference = gt.copy()
        else:
            assert np.array_equal(gt, common_reference)

        destination = args.output / f"arc_{arc}"
        destination.mkdir()
        np.save(destination / "vol_fdk.npy", fdk)

        masks = {
            "whole_volume": np.ones(gt.shape, dtype=bool),
            "reference_above_0.05": gt > 0.05,
            "reference_at_or_below_0.05": gt <= 0.05,
        }
        for method, volume in (("fdk_raw", fdk), ("r2_final", final)):
            for region, mask in masks.items():
                rows.append({
                    "arc_degrees": arc,
                    "method": method,
                    "region": region,
                    **metrics(volume, gt, mask),
                })

        records[str(arc)] = {
            "scene_scale": float(scene.scene_scale),
            "training_views": len(cameras),
            "fdk_min": float(fdk.min()),
            "fdk_max": float(fdk.max()),
            "fdk_negative_fraction": float(np.mean(fdk < 0)),
            "fdk_voxels_above_0.05": int(np.count_nonzero(fdk > 0.05)),
            "volume_sha256": sha256(destination / "vol_fdk.npy"),
        }

        inputs = [
            case / "meta_data.json",
            case / metadata["vol"],
            final_folder / "vol_pred.npy",
            final_folder / "vol_gt.npy",
        ]
        inputs.extend(case / frame["file_path"]
                      for frame in metadata["proj_train"])
        for path in inputs:
            hashes[str(path)] = sha256(path)

        del scene, cameras, projections, fdk, final, gt, saved_gt, masks
        torch.cuda.empty_cache()

    csv_path = args.output / "volume_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for path in (
        Path(__file__), REPO / "initialize_pcd.py",
        REPO / "r2_gaussian/dataset/dataset_readers.py",
        REPO / "r2_gaussian/utils/ct_utils.py",
    ):
        hashes[str(path)] = sha256(path)

    protocol = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "method": "Initializer FDK before thresholding and point sampling",
        "geometry": "Same Scene scaling and camera path as initialize_pcd.py",
        "reference_usage": "Evaluation only",
        "test_projections_used": False,
        "clipping": False,
        "intensity_normalization": False,
        "density_rescale_applied": False,
        "cases": records,
        "input_sha256": hashes,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    print("\nWhole-volume comparison:")
    for row in rows:
        if row["region"] == "whole_volume":
            print(
                f'{row["arc_degrees"]:3d} degrees | '
                f'{row["method"]:8s} | '
                f'PSNR {row["psnr_peak_1"]:.4f} | '
                f'MAE {row["mae"]:.6f} | '
                f'RMSE {row["rmse"]:.6f}'
            )
    print("\nFDK ranges and sampling eligibility:")
    print(json.dumps(records, indent=2))
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
