"""Evaluate fixed final volumes using shared held-out TIGRE projections."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate import project, sha256


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/trdp2/chest_50views_noisefree"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/limited_angle_120_diagnostics/projection_angles"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    assert args.batch_size > 0
    args.output.mkdir(parents=True, exist_ok=False)

    cases = {
        arc: args.data / f"arc_{arc}" / "0_chest_cone"
        for arc in (360, 120)
    }
    metas = {
        arc: json.loads((case / "meta_data.json").read_text())
        for arc, case in cases.items()
    }

    # Acquisition span differs, but physical scanner geometry must match.
    scanners = {
        arc: {
            key: value for key, value in meta["scanner"].items()
            if key != "totalAngle"
        }
        for arc, meta in metas.items()
    }
    assert scanners[360] == scanners[120], "Scanner mismatch"
    scanner = metas[360]["scanner"]

    frames = metas[360]["proj_test"]
    angles = np.array([frame["angle"] for frame in frames], dtype=np.float64)
    angles120 = np.array(
        [frame["angle"] for frame in metas[120]["proj_test"]],
        dtype=np.float64,
    )
    assert np.array_equal(angles, angles120), "Evaluation-angle mismatch"
    assert len(angles) == 100, "Expected 100 held-out views"
    degrees = np.rad2deg(angles) % 360
    assert np.allclose(degrees, (np.arange(100) + 0.5) * 3.6)

    for meta in metas.values():
        train = np.array([frame["angle"] for frame in meta["proj_train"]])
        separation = np.abs(
            np.angle(np.exp(1j * (angles[:, None] - train[None, :])))
        )
        assert separation.min() > 1e-8, "Train/test angle overlap"

    hashes = {}

    def load_array(path):
        hashes[str(path)] = sha256(path)
        array = np.load(path, allow_pickle=False)
        assert np.isfinite(array).all(), f"Nonfinite values: {path}"
        return array

    references = {}
    for arc in (360, 120):
        references[arc] = np.stack([
            load_array(cases[arc] / frame["file_path"])
            for frame in metas[arc]["proj_test"]
        ])
    assert np.array_equal(references[360], references[120])
    reference_projections = references[360]
    del references

    gt = load_array(cases[360] / metas[360]["vol"])
    assert np.array_equal(gt, load_array(cases[120] / metas[120]["vol"]))
    assert gt.shape == tuple(scanner["nVoxel"])

    def forward(volume):
        result = []
        volume = np.ascontiguousarray(volume, dtype=np.float32)
        for start in range(0, len(angles), args.batch_size):
            stop = min(start + args.batch_size, len(angles))
            print(f"  Angles {start + 1}-{stop}/{len(angles)}", flush=True)
            result.append(project(volume, scanner, angles[start:stop]))
        return np.concatenate(result)

    print("Checking reference reprojection...", flush=True)
    calibration = forward(gt).astype(np.float64)
    calibration -= reference_projections
    calibration_mae = float(np.abs(calibration).mean())
    calibration_max = float(np.abs(calibration).max())
    tolerance = 1e-5 * max(1.0, float(np.abs(reference_projections).max()))
    print(f"Reference reprojection MAE: {calibration_mae:.8g}")
    print(f"Reference maximum error:   {calibration_max:.8g}")
    assert calibration_max <= tolerance, (
        "Reference reprojection failed. Check geometry/projector before "
        "interpreting reconstruction errors."
    )
    del calibration

    rows = []
    for arc in (360, 120):
        folder = Path(
            f"output/chest_50views_noisefree_arc{arc}"
        ) / "point_cloud/iteration_30000"
        saved_gt = load_array(folder / "vol_gt.npy")
        assert np.array_equal(saved_gt, gt), "Saved reference mismatch"
        del saved_gt

        prediction = load_array(folder / "vol_pred.npy")
        assert prediction.shape == gt.shape
        print(f"Projecting {arc}-degree reconstruction...", flush=True)
        projections = forward(prediction)
        assert projections.shape == reference_projections.shape
        del prediction

        for index, angle in enumerate(degrees):
            error = (
                projections[index].astype(np.float64)
                - reference_projections[index].astype(np.float64)
            )
            mse = float(np.mean(error ** 2))
            rows.append({
                "arc_degrees": arc,
                "view_index": index,
                "angle_degrees": float(angle),
                "sector": "inside_120" if angle < 120 else "outside_120",
                "mae": float(np.mean(np.abs(error))),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "signed_bias": float(np.mean(error)),
            })
        del projections

    write_csv(args.output / "per_angle_metrics.csv", rows)

    summaries = []
    for arc in (360, 120):
        for sector in ("all", "inside_120", "outside_120"):
            selected = [
                row for row in rows
                if row["arc_degrees"] == arc
                and (sector == "all" or row["sector"] == sector)
            ]
            summaries.append({
                "arc_degrees": arc,
                "sector": sector,
                "view_count": len(selected),
                "mae": float(np.mean([r["mae"] for r in selected])),
                "rmse": float(np.sqrt(np.mean([r["mse"] for r in selected]))),
                "signed_bias": float(np.mean([
                    r["signed_bias"] for r in selected
                ])),
            })
    write_csv(args.output / "sector_summary.csv", summaries)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for arc, color in ((360, "tab:blue"), (120, "tab:orange")):
        selected = [row for row in rows if row["arc_degrees"] == arc]
        x = [r["angle_degrees"] for r in selected]
        axes[0].plot(
            x, [r["mae"] for r in selected], color=color,
            label=f"{arc}° reconstruction",
        )
        axes[1].plot(
            x, [r["signed_bias"] for r in selected], color=color,
        )

    for ax in axes:
        ax.axvspan(0, 120, color="grey", alpha=0.12)
        ax.axvline(120, color="grey", linestyle="--", linewidth=1)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 360)
    axes[0].set_ylabel("Projection MAE")
    axes[0].legend()
    axes[0].set_title(
        "Shared held-out views; shading = acquired 120° interval\n"
        "TIGRE projections of final voxel volumes; no intensity rescaling"
    )
    axes[1].set_ylabel("Signed bias (prediction − reference)")
    axes[1].set_xlabel("Source angle (degrees)")
    axes[1].axhline(0, color="black", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(args.output / "projection_error_by_angle.png", dpi=180)
    plt.close(fig)

    for case in cases.values():
        path = case / "meta_data.json"
        hashes[str(path)] = sha256(path)
    for path in (Path(__file__), Path(__file__).with_name("generate.py"),
                 Path("r2_gaussian/utils/ct_utils.py")):
        hashes[str(path)] = sha256(path)

    protocol = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "iteration": 30000,
        "projector": "TIGRE via generate.project",
        "representation": "saved voxel volume, not direct Gaussian rendering",
        "normalization": "none",
        "clipping": "none",
        "sector_definition": "inside=[0,120); outside=[120,360)",
        "evaluation_only": True,
        "batch_size": args.batch_size,
        "reference_reprojection_mae": calibration_mae,
        "reference_reprojection_max_error": calibration_max,
        "reference_reprojection_tolerance": tolerance,
        "input_sha256": hashes,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    print("\n" + (args.output / "sector_summary.csv").read_text())
    print(f"Saved diagnostics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
