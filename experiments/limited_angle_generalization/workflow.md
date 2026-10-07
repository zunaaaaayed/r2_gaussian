# Guarded workflow — 7 October 2026

Preparation/backend isolation is preserved at commit `a26f8f8` on
`experiment/limited-angle-120`. Work continues on
`experiment/limited-angle-generalization`. The completed pilot and native
initialization/training code are unchanged by this continuation.

## Next workstation commands

```bash
tmux new-session -A -s trdp2-generalization
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLBACKEND=Agg
git switch experiment/limited-angle-generalization
python -m unittest discover -s experiments/limited_angle_generalization/tests -v
python experiments/limited_angle_generalization/prepare.py \
  --manifest experiments/limited_angle_generalization/manifests/development_v4.json \
  --check-pilot --dry-run
nvidia-smi
python experiments/limited_angle_generalization/run_cases.py \
  --manifest experiments/limited_angle_generalization/manifests/development_v4.json \
  --case chest_start0_span120 --stage generate
```

The last command only prints a plan. After checking GPU availability, the next
execution step is dataset generation (no optimization):

```bash
python experiments/limited_angle_generalization/run_cases.py \
  --manifest experiments/limited_angle_generalization/manifests/development_v4.json \
  --case chest_start0_span120 --stage generate --execute
python experiments/limited_angle_generalization/run_cases.py \
  --manifest experiments/limited_angle_generalization/manifests/development_v4.json \
  --case chest_start0_span120 --stage verify
```

Inspect the generation job's `state.json`, `context.json`, and `process.log` under
`output/limited_angle_generalization_jobs/` before proceeding. Generation creates
100 shared evaluation views and 50 training views in batches of at most ten.
It copies the reference and evaluation files byte-for-byte between paired cases,
checks all checksums, shapes, radian angles, physical geometry and code identity,
and independently calls the same projector again for one reference view. This
last check establishes internal consistency, not independent physics validation.
TIGRE runs in a separate process to avoid its forward projector resetting a live
PyTorch CUDA context.

To inspect subsequent commands, replace `--stage generate` with `--stage initialize`,
`--stage train`, or `--stage evaluate`, without `--execute`. Prerequisites are
checked on execution, so a dry-run does not certify that the preceding stage exists.
Do not launch a full reconstruction until generation/initialization artifacts have
been reviewed. Training is fixed at the baseline 30,000 iterations, seed zero and
TV 0.05. This is a single-case baseline runner, not a sweep or a tuned-TV runner.

## Recorded evidence and limits

- 33 research-environment CPU tests pass: angle/split checks, source and code
  identity, shared-file reuse, corruption and partial-generation rejection,
  subprocess failures, and known numerical metrics. Dataset tests use a synthetic
  injected projector; no new GPU generation or reconstruction was run here.
- The earlier isolated backend smoke passed on the workstation and was also
  confirmed by the user's v2 report. It does not validate this entire workflow.
- Each stage refuses an existing job directory. Initialization refuses existing
  points; training reserves a fresh output and verifies the recorded initializer.
  A zero exit code alone is insufficient: expected artifacts must pass checks.
  Failed/partial artifacts are retained. There is no automatic retry or resume;
  inspect them and prepare distinct destinations before a deliberate retry.
  A forcibly killed parent can leave `running` state; this is not completion.
- Fixed-final evaluation computes unclipped unit-peak PSNR, MAE, RMSE, inherited
  slice-averaged volume SSIM and physical-spacing gradient errors. The boundary
  mask is the upper decile of positive reference-gradient magnitudes on the common
  forward-difference interior (ties included). Constant references have no boundary
  metric; exact equality reports null PSNR plus `perfect_match=true`.
- Projection metrics and peak VRAM are explicitly unmeasured. Runtime is recorded
  per stage. No speed, quality, or memory advantage is established.
- Only one independent source volume exists. Independent-volume generalization,
  a validated geometry descriptor, regularizer direction/strength, tuned-TV and
  wrong-orientation controls, the inherited half-voxel initializer convention,
  and SPARK full-text comparison remain open. Geometry regularization is a
  research hypothesis, with neither effectiveness nor novelty established.

Frozen manifests bind source code. Further code changes require a new reviewed
snapshot; do not bypass the checks or rewrite historical manifests.
