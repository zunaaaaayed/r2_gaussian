"""Inspect completed reconstructions without changing training or checkpoints."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/limited_angle_120_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    arc: ROOT / f"output/chest_50views_noisefree_arc{arc}"
    for arc in (360, 120)
}
FINAL = "point_cloud/iteration_30000"
MASK_THRESHOLD = 0.05


def load_volume(path):
    array = np.load(path, mmap_mode="r")
    assert array.ndim == 3 and np.isfinite(array).all(), path
    return array


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Verify that both runs were evaluated against the same reference.
gt_paths = {arc: run / FINAL / "vol_gt.npy" for arc, run in RUNS.items()}
gt = load_volume(gt_paths[360])
assert np.array_equal(gt, load_volume(gt_paths[120])), "References differ."

pred_paths = {arc: run / FINAL / "vol_pred.npy" for arc, run in RUNS.items()}
pred = {arc: load_volume(path) for arc, path in pred_paths.items()}
assert all(array.shape == gt.shape for array in pred.values())

# Exploratory intensity mask, NOT an anatomical segmentation.
mask = gt > MASK_THRESHOLD
assert mask.any() and (~mask).any()

rows = []
for arc, volume in pred.items():
    error = np.asarray(volume, dtype=np.float64) - gt
    for region in ("whole_volume", "reference_above_0.05", "reference_at_or_below_0.05"):
        values = (
            error.ravel() if region == "whole_volume"
            else error[mask] if region == "reference_above_0.05"
            else error[~mask]
        )
        mse = float(np.mean(values**2))
        rows.append({
            "arc_degrees": arc,
            "region": region,
            "voxel_count": values.size,
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(mse)),
            "signed_bias": float(np.mean(values)),
            "absolute_error_p95": float(np.percentile(np.abs(values), 95)),
            "psnr_peak_1": float(-10 * np.log10(mse)) if mse > 0 else float("inf"),
        })
    del error

save_csv(OUT / "regional_metrics.csv", rows)

# Fixed slice positions chosen before inspecting results.
# Array axes are used because patient orientation has not been verified.
fractions = (0.35, 0.50, 0.65)
for axis in range(3):
    indices = [round(f * (gt.shape[axis] - 1)) for f in fractions]
    panels = []
    for index in indices:
        reference = np.take(gt, index, axis=axis)
        full = np.take(pred[360], index, axis=axis)
        limited = np.take(pred[120], index, axis=axis)
        panels.append((
            reference, full, limited,
            np.abs(full - reference),
            np.abs(limited - reference),
        ))

    # Shared across both methods and all displayed slices of this axis.
    errors = np.concatenate([
        row[column].ravel() for row in panels for column in (3, 4)
    ])
    error_max = max(float(np.percentile(errors, 99)), 1e-6)

    fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
    titles = (
        "Reference", "360-degree reconstruction", "120-degree reconstruction",
        "Absolute error: 360 degrees", "Absolute error: 120 degrees",
    )

    for row_index, (index, row) in enumerate(zip(indices, panels)):
        for column, array in enumerate(row):
            ax = axes[row_index, column]
            image = ax.imshow(
                array.T, origin="lower",
                cmap="gray" if column < 3 else "magma",
                vmin=0, vmax=1 if column < 3 else error_max,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(titles[column], fontsize=10)
            if column == 0:
                ax.set_ylabel(f"Axis {axis}, index {index}")
            if column == 4:
                fig.colorbar(image, ax=axes[row_index, 3:].tolist(), shrink=0.8)

    fig.suptitle(
        f"Array axis {axis}: final iteration 30000\n"
        f"Images: fixed [0, 1]; errors: shared [0, {error_max:.4f}] "
        "(above 99th percentile saturated)"
    )
    fig.savefig(OUT / f"slices_axis{axis}.png", dpi=160)
    plt.close(fig)

# Read all available recorded volume evaluations.
history = []
for arc, run in RUNS.items():
    for path in sorted((run / "eval").glob("iter_*/eval3d.yml")):
        with path.open() as stream:
            metrics = yaml.safe_load(stream)
        history.append({
            "arc_degrees": arc,
            "iteration": int(path.parent.name.split("_")[1]),
            "psnr_3d": metrics["psnr_3d"],
            "ssim_3d": metrics["ssim_3d"],
        })
assert history, "No evaluation history found."
save_csv(OUT / "optimization_history.csv", history)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
loss_rows = []

for arc, run in RUNS.items():
    data = sorted(
        (r for r in history if r["arc_degrees"] == arc),
        key=lambda r: r["iteration"],
    )
    for ax, metric in zip(axes[:2], ("psnr_3d", "ssim_3d")):
        ax.plot(
            [r["iteration"] for r in data],
            [r[metric] for r in data],
            marker="o", label=f"{arc} degrees",
        )

    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    tag = "train/loss_render"
    if tag in events.Tags().get("scalars", []):
        # Retain the latest event for each step if duplicate records exist.
        points = {event.step: event.value for event in events.Scalars(tag)}
        steps = sorted(points)
        axes[2].plot(
            steps, [points[s] for s in steps],
            linewidth=0.7, alpha=0.7, label=f"{arc} degrees",
        )
        loss_rows.extend({
            "arc_degrees": arc, "iteration": step,
            "training_projection_l1": points[step],
        } for step in steps)
    else:
        print(f"Note: no {tag} TensorBoard scalar found for {arc} degrees.")

for ax, title in zip(axes, (
    "Volume PSNR (dB)", "Slice-aggregated SSIM",
    "Training projection L1 (sampled view)",
)):
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend()
axes[2].set_yscale("log")
fig.savefig(OUT / "optimization_history.png", dpi=160)
plt.close(fig)

if loss_rows:
    save_csv(OUT / "training_projection_loss.csv", loss_rows)

provenance = {
    "script_sha256": file_hash(Path(__file__)),
    "volume_shape": list(gt.shape),
    "slice_fractions": list(fractions),
    "image_display_range": [0, 1],
    "error_display": "Pooled 99th percentile per axis; shared across methods",
    "regional_mask": "Reference > 0.05; exploratory, not anatomical segmentation",
    "intensity_units": "Dataset normalized attenuation; not HU",
    "prediction_sha256": {
        str(arc): file_hash(path) for arc, path in pred_paths.items()
    },
    "reference_sha256": file_hash(gt_paths[360]),
}
(OUT / "inspection_protocol.json").write_text(
    json.dumps(provenance, indent=2) + "\n"
)

print("\nREGIONAL METRICS")
for row in rows:
    print(
        f"{row['arc_degrees']:3d} deg | {row['region']:30s} | "
        f"MAE {row['mae']:.6f} | RMSE {row['rmse']:.6f} | "
        f"bias {row['signed_bias']:+.6f}"
    )

print("\nVOLUME METRICS OVER OPTIMIZATION")
for row in history:
    print(
        f"{row['arc_degrees']:3d} deg | iteration {row['iteration']:5d} | "
        f"PSNR {row['psnr_3d']:.4f} | SSIM {row['ssim_3d']:.5f}"
    )

print("\nDiagnostics saved to:", OUT)
