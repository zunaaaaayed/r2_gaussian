"""Compare final reconstructions in absolute and acquisition-relative angles."""

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
    out = Path("output/limited_angle_120_diagnostics/acquisition_rotation")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")

    configurations = {
        "control_360": ("arc_360", "arc360", 0, 360),
        "start_0": ("arc_120", "arc120", 0, 120),
        "start_90": ("arc_120_start_90", "arc120_start90", 90, 120),
    }
    hashes = {}
    volumes = {}
    reference = None
    targets = None
    angles = None
    scanner = None

    def load(path):
        hashes[str(path)] = sha256(path)
        value = np.load(path, allow_pickle=False)
        assert np.isfinite(value).all(), f"Nonfinite values: {path}"
        return value

    for name, (data_folder, run_suffix, start, span) in configurations.items():
        case = Path("data/trdp2/chest_50views_noisefree") / data_folder / "0_chest_cone"
        meta_path = case / "meta_data.json"
        hashes[str(meta_path)] = sha256(meta_path)
        meta = json.loads(meta_path.read_text())

        gt = load(case / meta["vol"])
        test_angles = np.array(
            [frame["angle"] for frame in meta["proj_test"]],
            dtype=np.float64,
        )
        train_angles = np.array(
            [frame["angle"] for frame in meta["proj_train"]],
            dtype=np.float64,
        )
        assert len(train_angles) == 50
        assert np.allclose(
            train_angles, np.deg2rad(start + np.arange(50) * span / 50)
        )
        separation = np.abs(np.angle(np.exp(
            1j * (test_angles[:, None] - train_angles[None, :])
        )))
        assert separation.min() > 1e-8, "Training/test overlap"

        test_projections = np.stack([
            load(case / frame["file_path"]) for frame in meta["proj_test"]
        ])
        physical_geometry = {
            key: value for key, value in meta["scanner"].items()
            if key not in ("startAngle", "totalAngle")
        }

        if reference is None:
            reference = gt
            targets = test_projections
            angles = test_angles
            scanner = meta["scanner"]
            common_geometry = physical_geometry
        else:
            assert np.array_equal(reference, gt)
            assert np.array_equal(angles, test_angles)
            assert np.array_equal(targets, test_projections)
            assert physical_geometry == common_geometry

        folder = Path(
            f"output/chest_50views_noisefree_{run_suffix}"
        ) / "point_cloud/iteration_30000"
        assert np.array_equal(reference, load(folder / "vol_gt.npy"))
        volumes[name] = load(folder / "vol_pred.npy")
        assert volumes[name].shape == reference.shape

    degrees = np.rad2deg(angles) % 360
    assert len(angles) == 100
    assert np.allclose(degrees, (np.arange(100) + 0.5) * 3.6)
    out.mkdir(parents=True)

    def forward(volume):
        result = []
        for first in range(0, len(angles), 10):
            print(f"  Views {first + 1}-{min(first + 10, len(angles))}",
                  flush=True)
            result.append(project(
                np.ascontiguousarray(volume, dtype=np.float32),
                scanner, angles[first:first + 10],
            ))
        return np.concatenate(result)

    print("Checking reference reprojection...", flush=True)
    error = forward(reference).astype(np.float64) - targets
    calibration = {
        "mae": float(np.abs(error).mean()),
        "max_error": float(np.abs(error).max()),
    }
    tolerance = 1e-5 * max(1.0, float(np.abs(targets).max()))
    assert calibration["max_error"] <= tolerance, calibration
    del error

    rows = []
    curves = {}
    for name, volume in volumes.items():
        print(f"Projecting {name}...", flush=True)
        predictions = forward(volume)
        start = configurations[name][2]
        span = configurations[name][3]
        relative = (degrees - start) % 360
        mae = []

        for index, angle in enumerate(degrees):
            error = (
                predictions[index].astype(np.float64)
                - targets[index].astype(np.float64)
            )
            view_mae = float(np.abs(error).mean())
            mae.append(view_mae)
            rows.append({
                "case": name,
                "view_index": index,
                "angle_degrees": float(angle),
                "relative_angle_degrees": float(relative[index]),
                "inside_acquired_arc": int(relative[index] < span),
                "mae": view_mae,
                "mse": float(np.mean(error ** 2)),
                "signed_bias": float(error.mean()),
            })
        curves[name] = np.array(mae)
        del predictions

    baseline = curves["control_360"]
    assert np.all(baseline > 0)
    for row in rows:
        row["mae_ratio_to_control"] = float(
            row["mae"] / baseline[row["view_index"]]
        )
    write_csv(out / "per_angle_metrics.csv", rows)

    summaries = []
    for name in configurations:
        sectors = ("all",) if name == "control_360" else ("all", "inside", "outside")
        for sector in sectors:
            selected = [
                row for row in rows
                if row["case"] == name
                and (
                    sector == "all"
                    or bool(row["inside_acquired_arc"]) == (sector == "inside")
                )
            ]
            matched_control_mae = float(np.mean([
                baseline[row["view_index"]] for row in selected
            ]))
            mean_mae = float(np.mean([row["mae"] for row in selected]))
            summaries.append({
                "case": name,
                "sector": sector,
                "view_count": len(selected),
                "mae": mean_mae,
                "rmse": float(np.sqrt(np.mean([
                    row["mse"] for row in selected
                ]))),
                "signed_bias": float(np.mean([
                    row["signed_bias"] for row in selected
                ])),
                "matched_control_mae": matched_control_mae,
                "ratio_of_mean_mae_to_control": mean_mae / matched_control_mae,
            })
    write_csv(out / "sector_summary.csv", summaries)

    # Report one maximum in each pre-specified relative half-circle.
    # These are descriptive summaries, not checkpoint-selection criteria.
    peaks = []
    for name in ("start_0", "start_90"):
        for lower, upper in ((0, 180), (180, 360)):
            selected = [
                row for row in rows if row["case"] == name
                and lower <= row["relative_angle_degrees"] < upper
            ]
            peak = max(selected, key=lambda row: row["mae"])
            peaks.append({
                "case": name,
                "relative_sector": f"[{lower},{upper})",
                "peak_absolute_degrees": peak["angle_degrees"],
                "peak_relative_degrees": peak["relative_angle_degrees"],
                "peak_mae": peak["mae"],
                "mae_ratio_to_control": peak["mae_ratio_to_control"],
            })
    write_csv(out / "peak_summary.csv", peaks)

    colors = {"start_0": "tab:orange", "start_90": "tab:purple"}
    labels = {"start_0": "120° starting at 0°", "start_90": "120° starting at 90°"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, name in zip(axes, ("start_0", "start_90")):
        start = configurations[name][2]
        ax.plot(degrees, baseline, color="tab:blue", label="360° control")
        ax.plot(degrees, curves[name], color=colors[name], label=labels[name])
        ax.axvspan(start, start + 120, color=colors[name], alpha=0.10)
        ax.set_title(labels[name])
        ax.set_xlabel("Absolute source angle (degrees)")
        ax.set_xlim(0, 360)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Raw projection MAE")
    fig.suptitle("Shared held-out directions; shading marks each acquired arc")
    fig.tight_layout()
    fig.savefig(out / "absolute_angle_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in ("start_0", "start_90"):
        relative = (degrees - configurations[name][2]) % 360
        order = np.argsort(relative)
        axes[0].plot(
            relative[order], curves[name][order],
            color=colors[name], label=labels[name],
        )
        axes[1].plot(
            relative[order], (curves[name] / baseline)[order],
            color=colors[name], label=labels[name],
        )
    for ax in axes:
        ax.axvspan(0, 120, color="grey", alpha=0.12)
        ax.axvline(120, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(0, 360)
        ax.set_xlabel("Angle relative to acquisition start (degrees)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Raw projection MAE")
    axes[1].set_ylabel("MAE / control MAE at matching absolute angle")
    axes[1].axhline(1, color="black", linestyle=":", linewidth=1)
    fig.suptitle("Acquisition-relative comparison; no angular interpolation")
    fig.tight_layout()
    fig.savefig(out / "relative_angle_comparison.png", dpi=180)
    plt.close(fig)

    for path in (
        Path(__file__), Path(__file__).with_name("generate.py"),
        Path("r2_gaussian/utils/ct_utils.py"),
    ):
        hashes[str(path)] = sha256(path)
    protocol = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "iteration": 30000,
        "reference_reprojection": calibration,
        "reference_tolerance": tolerance,
        "normalization": "none",
        "clipping": "none",
        "relative_angle": "(absolute_angle - acquisition_start) % 360",
        "ratio_baseline": "360-degree control at matching absolute angle",
        "input_sha256": hashes,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("\n" + (out / "sector_summary.csv").read_text())
    print((out / "peak_summary.csv").read_text())
    print(f"Reference check: {calibration}")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
