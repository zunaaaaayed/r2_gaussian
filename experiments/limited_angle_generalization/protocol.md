# Limited-angle reconstruction generalization protocol — preparation v1

Date: 2026-10-05. Baseline: [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian),
working [repository](https://github.com/zunaaaaayed/r2_gaussian), commit
`327a731834ad3e4f6b89a1e993df01cd3fa1bb6d`. This protocol defines proposed
experiments; it contains no new reconstruction results or novelty claim.

## Question and splits

Can one reconstruction rule, with settings chosen on development data and then
frozen, improve across acquisition orientations, spans, and independent volumes?
Each scan still receives its own Gaussian optimization. No population-trained
network or patient-specific prior is introduced in this milestone.

The inspected chest volume and all three historical acquisitions are development
evidence. Only this source volume is locally available. Its actual patient identity
is unknown. Four local datasets containing the same reference are one volume,
not four independent subjects. No final-test population is currently assigned.
Before population experiments, inventory legitimately accessible volumes, verify
patient/source relationships, group all crops/slices from the same patient, and
reserve independent volumes before looking at their reconstruction results.
Duplicate reference hashes and identity groups cannot cross population splits.

The configurable six-case manifest separates population role from acquisition
role. Reserve start 330°/span 120° for orientation validation, start 0°/span 90°
for span validation, and start 45°/span 150° for combined acquisition validation.
These are validation acquisitions on a development volume, not final population
tests. If their results inform tuning, record that exposure and reserve different
conditions for final evaluation. Future independent-volume tests should separately
measure familiar geometry and combined volume/acquisition shift. With only a few
heterogeneous examples, report example transfer, not patient-population accuracy.

## Acquisition and pairing

Use noise-free circular cone-beam geometry initially, with 50 training directions
per case. Preserve supplied scanner dimensions and detector orientation. Extents
and source distances are in supplied dataset length units, with unknown physical
calibration; intensities are supplied values, not HU. Record both explicit degree
arrays and radian metadata arrays. Scanner startAngle/totalAngle remain degrees.
Training angles are `start + i * span / 50`, right endpoint excluded, including
unwrapped angles beyond 360° for wraparound cases. Classification uses modular
relative degrees; a 1e-6° tolerance snaps start to inside and end to outside.

The pilot's midpoint evaluation grid collides with the proposed 90° training
grid. New cases therefore share `(i + 0.25) * 3.6°`, selected deterministically
against the union of all six training sets. Minimum separation is 0.3°; counts
are 100/0, 34/66, 34/66, 33/67, 25/75, 41/59 in config order. These new counts
do not revise the pilot's 33/67 counts. Changing the grid requires a new frozen
manifest and re-evaluation of all paired methods at the same angles. Existing
pilot projection metrics must not be pooled with metrics on the new grid.

Generation remains a next-stage implementation: compute one shared reference and
evaluation artifact per reference/physical-geometry/evaluation-angle group, copy
or link it without modifying the originals, and verify all hashes. Use
`limited_angle_120/generate.py:project` conventions (xyz→zyx, detector-row reversal).
Check tiny reference reprojection before full generation, then verify each actual
generated dataset against the frozen manifest. Internal reprojection agreement
does not independently validate physical geometry.

The 360°/50-view case is a matched sparse-view control. Ground truth is the known
volume. Fixed count changes angular density with span. Fixed-spacing experiments
are a separate later factor. Missing source angles are not an exact global
Fourier wedge for circular cone-beam data.

## Baselines and selection

1. Raw FDK from measured training data, retaining negative values in metrics.
2. Unchanged R²-Gaussian: L1 + 0.25 DSSIM, existing TV weight 0.05, 32³ patch,
   own FDK initializer (50,000 points, threshold 0.05, density scale 0.15),
   30,000 fixed iterations, original density control.
3. The same ordinary TV fairly tuned on development data; proposed initial
   weight search `{0.025, 0.05, 0.1}` only after the first baseline is reproduced.
4. Geometry regularization only after the design gates in `regularizer_design.md`.
5. Classical iterative TV/directional-TV after validating its 3D cone-beam operator
   and comparable data scaling; CIL challenge assumptions cannot be copied blindly.

For a first comparison, use identical measured files, initializer bytes, seed,
iteration budget and evaluator. The preparer currently accepts only baseline seed
zero and fixed settings because the original entry points hardcode seed zero.
Seed plumbing is future work; do not label restarts as replicates. Initializer,
densification and geometry changes are separate experimental factors.

Select one rule on development volume metrics with detail preservation as a
constraint; freeze config/code/input hashes before validation/final tests. Do not
select checkpoints or regularization weights with final-test reference PSNR.
Current training always evaluates iteration 1 and final, with intermediate default
checks; their logging is not a stopping rule. A later final-test runner must seal
these outputs until the declared final checkpoint. Baseline completion at 30,000
is preserved, including the existing final-iteration optimizer-step behavior.

Required geometry ablations: geometry disabled, equally tuned uniform penalty,
deliberately incorrect regularizer orientation with the correct projector, and
local versus global descriptor if introduced. Extra smoothing alone must not be
credited as a geometry benefit. Report initialization-only and density-control-only
changes separately.

## Metrics fixed before new results

Whole-volume PSNR uses `10 log10(1 / MSE)` for this supplied [0,1] reference,
matching `image_utils.metric_vol(pixel_max=1.0)`. MAE and RMSE retain numerical
values without clipping or independent normalization. Record reference range per
volume; do not silently normalize a future dataset into this convention.

Retain repository slice-aggregated SSIM, named accurately: 11×11 Gaussian window,
sigma 1.5, zero padding, constants 0.01² and 0.03² (unit range); average slices
whose reference maximum is positive within each array axis, then average the
three axis means. Record each axis value and valid slice count. No valid slices
means undefined, not zero or success. This is not volumetric-window 3D SSIM.

Predefine a detail diagnostic on the common interior voxel grid: forward
differences divided by source spacing, Euclidean reference gradient magnitude;
boundary mask = upper decile of strictly positive reference gradient magnitudes
(ties included). If empty, mark undefined. Reuse this exact mask across methods.
Report boundary MAE and mean norm of the difference between predicted/reference
physical gradients on the mask, plus per-axis gradient error. This is an
operational detail mask, not anatomical segmentation. It is proposed evaluator
behavior and is not yet an implemented evaluation command.

Secondary diagnostics: TIGRE projection MAE/RMSE/bias per held-out direction,
inside/outside counts and absolute/relative angle curves; distinguish these from
Gaussian-renderer projection loss. Use one display intensity window [0,1] for
this volume and a common declared error scale across methods. Report per-case
regressions and worst cases, runtime, preprocessing, peak VRAM and exact final
Gaussian count. The 100 views are correlated measurements, not 100 subjects.

## Compute and execution gates

| Stage | Work | Budget / release condition |
|---|---|---|
| Now | CPU manifest, pilot checks, tests, source/literature audit | No full GPU runs |
| Backend smoke | One Gaussian, 16×20×24 query; two 3-view 24×32 TIGRE calls | Reserve 5 minutes including imports; no training; opt-in script |
| Development pair | One baseline and one tuned-TV setting at a single development geometry | About 38–46 minutes training; reserve 1 hour plus measured generation/init overhead |
| Validation expansion | At most the three reserved acquisition cases after freezing settings | 57–69 minutes per method, estimate only |
| Independent volumes | After inventory/split freeze and benchmark runner review | Not estimable until data exist |

Pilot logs show start-to-completion times 18m57s, 22m15s, 19m25s; budgeting
19–23 minutes/run is evidence-based but not a guarantee. Six cases × one method
would take about 114–138 minutes; six × three methods about 5.7–6.9 GPU hours,
excluding generation/init/evaluation. This is a sizing estimate, not a scheduled
sweep. Data occupy about 219 MiB/case and completed output 189–222 MiB/case;
reserve at least 0.5 GiB per baseline case or 1 GiB/case when adding method
checkpoints and diagnostics. Peak training VRAM was not recorded; measure it on
the first new run. PyTorch allocator peaks exclude TIGRE allocations.

Before a training runner is enabled: exclusive run directories; dataset and
initializer verification; source diff, untracked code hashes, config, environment,
seed, timing and subprocess exit records; no overwrites; checkpoint identity
verification. Existing checkpoints do not capture Python/NumPy/Torch RNG states
or the remaining shuffled camera stack, so exact continuation is not established.
This milestone refuses resume. Use tmux and inspect current GPU processes; never
kill unrelated workloads. When using tee, enable pipefail and save exit codes.
