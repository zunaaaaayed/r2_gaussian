# TRDP2: limited-angle experiment checkpoint

Continuation reconciled 5 October 2026 on `experiment/limited-angle-120`,
commit `327a731834ad3e4f6b89a1e993df01cd3fa1bb6d`. The measurements below
are frozen pilot evidence; the previous raw-FDK next-task note is superseded.

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
| 120 degrees, start 90 degrees | 28.594484 | 0.862338 |

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

## Subsequently completed diagnostics

Raw FDK before thresholding, point sampling and density rescaling was evaluated
for the 360-degree and start-0 120-degree cases. Saved whole-volume PSNR is
24.16951069526235 and 19.256990392660512 respectively. Negative values were
retained. FDK/final projection comparisons and slice/error plots are complete.
Evidence: `output/limited_angle_120_diagnostics/fdk/` and `fdk_vs_final/`.

The start-90 acquisition [90,210) is also complete at 30,000 iterations.
Its 50 training directions end at 207.6 degrees; reference and 100 evaluation
files match the other pilots byte-for-byte. Both 120-degree cases have 33 inside
and 67 outside held-out directions. Rotated outside/control mean-MAE ratio:
3.114293888654874 (not 3.19). Error peaks shifted 97.2 degrees after rotating
acquisition by 90 degrees; this is descriptive, not an established invariant.
Evidence: `output/limited_angle_120_diagnostics/acquisition_rotation/`.
Those diagnostic CSV/protocol files exist locally under ignored output storage;
they are not all present in the tracked `results/` directory.

The interrupted start-0 attempt remains archived under
`output/chest_50views_noisefree_arc120_interrupted_20260928_161645/`.
It is a replaced attempt, not another seed or an exact resumed run.

## Next task

The first-session preparation milestone is in
[`../limited_angle_generalization/README.md`](../limited_angle_generalization/README.md):
source/environment audit, protocol, literature matrix, acquisition manifest,
CPU checks and a conditional regularizer design. Follow its exact workstation
commands for validation and the opt-in tiny backend smoke. No full benchmark
or new scientific regularizer has been launched.

The main objective is fixed-setting reconstruction generalization across
acquisition configurations and independent volumes. This chest and its inspected
orientations remain development data. Independent-volume data and a justified
geometry-to-penalty rule are unresolved; geometry conditioning is a hypothesis,
not an established improvement or verified novelty.

Do not overwrite completed reconstructions.
Do not select checkpoints or tune methods using test-reference metrics.

## Storage

Small diagnostic outputs are versioned in results/.
Large datasets, reconstructed volumes, checkpoints, and compiled
extensions remain local and are not included in this Git snapshot.
