# TRDP2: limited-angle experiment checkpoint

## Completed experiment

- One synthetic chest volume; noise-free cone-beam projections.
- 50 training views for each acquisition.
- Full-circle control: [0, 360), spacing 7.2 degrees.
- Limited-angle case: [0, 120), spacing 2.4 degrees.
- Shared 100 held-out full-circle midpoint views.
- Separate FDK-based initialization for each acquisition.
- Final evaluation fixed at iteration 30000.
- View count is matched; angular spacing differs.
- The comparison includes initialization and optimization effects.

## Final volume results

| Acquisition | PSNR | Slice-aggregated SSIM |
|---|---:|---:|
| 360 degrees | 35.546806 | 0.941930 |
| 120 degrees | 30.906631 | 0.900952 |

The limited-angle result exhibits directional smearing, boundary
distortions, and loss of fine detail in the inspected slices.

Intensity-threshold regions are not anatomical segmentations.
Small whole-volume signed bias hides opposing regional biases.

## Held-out projection diagnostic

Final saved voxel volumes were forward-projected using TIGRE,
with the dataset generator's geometry and axis conventions.
No clipping or independent intensity normalization was applied.

Reference reprojection MAE: 0.
Reference reprojection maximum absolute error: 0.
Check tolerance: 1e-5.

This verifies internal consistency with dataset generation,
not independent physical validation.

Relative to the 360-degree reconstruction:
- Inside [0,120): projection MAE approximately 1.06 times larger.
- Outside [0,120): projection MAE approximately 2.63 times larger.
- Error peaks at 145.8 and 329.4 degrees are approximately
  5.00 and 4.88 times the corresponding control errors.
- At 239.4 degrees, the ratio is approximately 1.33.

Unobserved directions are not equally difficult.
These are reference-based error measurements, not uncertainty estimates.
No correction or evidential-learning method has been tested yet.

## Next task

Reconstruct and evaluate the FDK volumes BEFORE point sampling,
using only each case's training projections.

Compare FDK with the final Gaussian reconstruction using:
1. Volume metrics and matching slice/error plots.
2. Held-out projection error versus angle.

Determine which distortions exist before Gaussian optimization
and which are reduced, preserved, or introduced afterward.

Do not overwrite completed reconstructions.
Do not select checkpoints or tune methods using test-reference metrics.

## Storage

Small diagnostic outputs are versioned in results/.
Large datasets, reconstructed volumes, checkpoints, and compiled
extensions remain local and are not included in this Git snapshot.
