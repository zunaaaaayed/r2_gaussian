"""Compare saved raw FDK and final R2 volumes; no reconstruction or training."""

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
        "--output", type=Path,
        default=Path("output/limited_angle_120_diagnostics/fdk_vs_final"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    hashes = {}

    def load(path):
        hashes[str(path)] = sha256(path)
        value = np.load(path, allow_pickle=False)
        assert np.isfinite(value).all(), f"Nonfinite values: {path}"
        return value

    volumes = {}
    metadata = {}
    references = {}
    gt = None

    for arc in (360, 120):
        case = Path(
            f"data/trdp2/chest_50views_noisefree/arc_{arc}/0_chest_cone"
        )
        meta_path = case / "meta_data.json"
        hashes[str(meta_path)] = sha256(meta_path)
        meta = json.loads(meta_path.read_text())
        metadata[arc] = meta

        reference = load(case / meta["vol"])
        if gt is None:
            gt = reference
        else:
            assert np.array_equal(gt, reference)

        folder = Path(
            f"output/chest_50views_noisefree_arc{arc}"
        ) / "point_cloud/iteration_30000"
        assert np.array_equal(gt, load(folder / "vol_gt.npy"))

        volumes[arc] = {
            "FDK": load(Path(
                "output/limited_angle_120_diagnostics/fdk"
            ) / f"arc_{arc}/vol_fdk.npy"),
            "R2": load(folder / "vol_pred.npy"),
        }
        for volume in volumes[arc].values():
            assert volume.shape == gt.shape

        references[arc] = np.stack([
            load(case / frame["file_path"])
            for frame in meta["proj_test"]
        ])

    assert np.array_equal(references[360], references[120])
    reference_projections = references[360]

    angles = np.array([
        frame["angle"] for frame in metadata[360]["proj_test"]
    ], dtype=np.float64)
    assert np.array_equal(angles, np.array([
        frame["angle"] for frame in metadata[120]["proj_test"]
    ]))
    degrees = np.rad2deg(angles) % 360
    assert len(angles) == 100
    assert np.allclose(degrees, (np.arange(100) + 0.5) * 3.6)

    geometries = [
        {k: v for k, v in metadata[arc]["scanner"].items()
         if k != "totalAngle"}
        for arc in (360, 120)
    ]
    assert geometries[0] == geometries[1]
    scanner = metadata[360]["scanner"]

    # Plot identical slices with an error scale shared across acquisitions.
    error_limits = {}
    for axis in range(3):
        indices = [
            int(round(fraction * (gt.shape[axis] - 1)))
            for fraction in (0.35, 0.50, 0.65)
        ]
        displayed_errors = []
        for arc in (360, 120):
            for volume in volumes[arc].values():
                for index in indices:
                    error = np.abs(
                        np.take(volume, index, axis=axis)
                        - np.take(gt, index, axis=axis)
                    )
                    displayed_errors.append(error.ravel())
        vmax = max(
            float(np.percentile(np.concatenate(displayed_errors), 99)),
            1e-12,
        )
        error_limits[str(axis)] = vmax

        for arc in (360, 120):
            fig, axes = plt.subplots(3, 5, figsize=(17, 10))
            titles = (
                "Reference", "Raw FDK", "Final R2",
                "|FDK − reference|", "|R2 − reference|",
            )
            for row, index in enumerate(indices):
                reference_slice = np.take(gt, index, axis=axis).T
                fdk = np.take(volumes[arc]["FDK"], index, axis=axis).T
                final = np.take(volumes[arc]["R2"], index, axis=axis).T
                panels = (
                    reference_slice, fdk, final,
                    np.abs(fdk - reference_slice),
                    np.abs(final - reference_slice),
                )
                for column, panel in enumerate(panels):
                    ax = axes[row, column]
                    im = ax.imshow(
                        panel, origin="lower", interpolation="nearest",
                        cmap="gray" if column < 3 else "magma",
                        vmin=0, vmax=1 if column < 3 else vmax,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row == 0:
                        ax.set_title(titles[column])
                    if column == 0:
                        ax.set_ylabel(f"Index {index}")
                    if column == 4:
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(
                f"{arc}° acquisition | array axis {axis}\n"
                "Grayscale window [0,1]: out-of-window values display clipped; "
                "calculations use raw values\n"
                f"Errors [0,{vmax:.4f}], shared across both acquisitions; "
                "above pooled 99th percentile saturated"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            fig.savefig(
                args.output / f"arc{arc}_axis{axis}.png", dpi=150
            )
            plt.close(fig)

    def forward(volume):
        batches = []
        for start in range(0, len(angles), 10):
            print(f"  Views {start + 1}-{min(start + 10, len(angles))}",
                  flush=True)
            batches.append(project(
                np.ascontiguousarray(volume, dtype=np.float32),
                scanner, angles[start:start + 10],
            ))
        return np.concatenate(batches)

    print("Checking reference reprojection...", flush=True)
    check = forward(gt).astype(np.float64) - reference_projections
    calibration = {
        "mae": float(np.abs(check).mean()),
        "max_error": float(np.abs(check).max()),
    }
    tolerance = 1e-5 * max(1.0, float(np.abs(reference_projections).max()))
    assert calibration["max_error"] <= tolerance, calibration
    del check

    rows = []
    for arc in (360, 120):
        for method, volume in volumes[arc].items():
            print(f"Projecting {arc}° {method}...", flush=True)
            projections = forward(volume)
            for index, angle in enumerate(degrees):
                error = (
                    projections[index].astype(np.float64)
                    - reference_projections[index].astype(np.float64)
                )
                rows.append({
                    "arc_degrees": arc,
                    "method": method,
                    "angle_degrees": float(angle),
                    "sector": "inside_120" if angle < 120 else "outside_120",
                    "mae": float(np.abs(error).mean()),
                    "mse": float(np.mean(error ** 2)),
                    "signed_bias": float(error.mean()),
                })
            del projections

    write_csv(args.output / "per_angle_metrics.csv", rows)

    summary = []
    for arc in (360, 120):
        for method in ("FDK", "R2"):
            for sector in ("all", "inside_120", "outside_120"):
                selected = [
                    r for r in rows
                    if r["arc_degrees"] == arc and r["method"] == method
                    and (sector == "all" or r["sector"] == sector)
                ]
                summary.append({
                    "arc_degrees": arc,
                    "method": method,
                    "sector": sector,
                    "view_count": len(selected),
                    "mae": float(np.mean([r["mae"] for r in selected])),
                    "rmse": float(np.sqrt(np.mean([
                        r["mse"] for r in selected
                    ]))),
                    "signed_bias": float(np.mean([
                        r["signed_bias"] for r in selected
                    ])),
                })
    write_csv(args.output / "sector_summary.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arc in zip(axes, (360, 120)):
        for method, color in (("FDK", "tab:purple"), ("R2", "tab:green")):
            selected = [
                r for r in rows
                if r["arc_degrees"] == arc and r["method"] == method
            ]
            ax.plot(
                [r["angle_degrees"] for r in selected],
                [r["mae"] for r in selected],
                label=method, color=color,
            )
        ax.axvspan(0, 120, color="grey", alpha=0.12)
        ax.axvline(120, color="grey", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlim(0, 360)
        ax.set_title(f"{arc}° acquisition")
        ax.set_xlabel("Held-out source angle (degrees)")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
    axes[0].set_ylabel("Raw projection MAE (log scale)")
    fig.suptitle(
        "FDK versus final R2: shared held-out views\n"
        "Grey shading marks [0°,120°) in both panels"
    )
    fig.tight_layout()
    fig.savefig(args.output / "projection_comparison.png", dpi=180)
    plt.close(fig)

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
        "reference_reprojection": calibration,
        "reference_tolerance": tolerance,
        "projection_batch_size": 10,
        "intensity_normalization": False,
        "numerical_clipping": False,
        "grayscale_display_window": [0, 1],
        "error_display_limits_by_axis": error_limits,
        "slice_fractions": [0.35, 0.50, 0.65],
        "input_sha256": hashes,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    print("\n" + (args.output / "sector_summary.csv").read_text())
    print(f"Reference check: {calibration}")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
