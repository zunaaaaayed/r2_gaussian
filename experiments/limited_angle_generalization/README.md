# Limited-angle generalization: first-session preparation

Prepared 5 October 2026 at baseline commit
`327a731834ad3e4f6b89a1e993df01cd3fa1bb6d`, branch
`experiment/limited-angle-120`. Files are local changes; no commit or push was made.
Research objective: one reconstruction rule with frozen settings that transfers
across limited-angle acquisition configurations and independent volumes.
Original method credit: [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian).
Working [repository](https://github.com/zunaaaaayed/r2_gaussian).

## What is ready

| File | Purpose |
|---|---|
| `audit.md` | Actual checkout, host, CUDA/Conda/data inventory, completed pilot evidence and preservation |
| `protocol.md` | Algorithmic generalization, split rules, baselines, metrics and staged compute budget |
| `literature.md` | Primary-source evidence matrix; accessible EPNet/CIL detail and explicit SPARK full-text gap |
| `regularizer_design.md` | Verified TV/query/density-control behavior, conditional design and falsification gates |
| `configs/development.json` | Six planned configurations; only the real available chest source is assigned |
| `benchmark.py`, `prepare.py` | CPU-only resolution, source/code hashes, explicit degree/radian arrays, shared disjoint grid, leakage/overwrite checks |
| `manifests/development_v3.json` | Current resolved source/config snapshot; status is planned, not generated |
| `manifests/development_v1.json` | Superseded preparation snapshot before correcting the smoke's voxel-center convention; intentionally fails current code-hash validation; never executed |
| `manifests/pilot_preservation.json`, `pilot_audit.json` | Before-edit hashes and saved diagnostic input-integrity checks |
| `manifests/checks.json`, `final_check_0.txt`, `final_check_1.txt`, `final_check_2.txt` | Final Python 3.9 verification summary and actual command outputs |
| `manifests/validation.txt` | Earlier v1 validation log, retained as superseded preparation evidence |
| `tests/test_benchmark.py`, `tests/test_baseline_cpu.py` | Synthetic CPU checks and numerical baseline TV/PSNR checks |
| `smoke_gpu.py` | Opt-in tiny baseline query/gradient/projector check; no optimization |

The old pilot continuation note has been reconciled. Its historical measurements,
data, initializers, checkpoint files, diagnostic artifacts and TIGRE patch are
preserved. No baseline training/initialization implementation was modified. No
new full GPU run was launched. No geometry regularizer was implemented because
the direction/strength rule lacks validation.

The preparation CLI deliberately has **no execution or resume path**. It identifies
new/existing outputs and distinguishes completed artifacts from unverified
checkpoints/incomplete directories; it refuses every existing destination.
Printed initialization/training command templates are review material only;
their inputs do not yet exist and they must be executed through the future guarded
runner after generation and hash verification, not copied into a sweep.
It does not infer resume safety from a checkpoint filename. A reviewed generator,
training runner and evaluator are later deliverables. This bounds the work to
Section 17 rather than silently turning a proposed manifest into a GPU sweep.

The historical grid tests pass with 100/0, 33/67, 33/67 sector counts. The new 90°
case would overlap the historical midpoint grid; the six-case snapshot uses a
shared 0.25-bin offset and records its own counts. The three paired pilot reference
volumes and 100 evaluation files match byte-for-byte. All 851 saved diagnostic
input-hash entries across four protocols match (entries include repeated files).

## Exact university workstation commands

Run inside the university host; from another machine first use
`ssh mdzislam@cheonech`. Use a persistent terminal for any GPU work:

```bash
tmux new-session -A -s trdp2-generalization
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLBACKEND=Agg
git status --short
nvidia-smi
python -m unittest discover -s experiments/limited_angle_generalization/tests -v
python experiments/limited_angle_generalization/prepare.py \
  --manifest experiments/limited_angle_generalization/manifests/development_v3.json \
  --check-pilot --dry-run
```

The test suite now has 23 tests, including worker isolation/failure checks. All passed in the existing Python 3.9 research
environment on CPU. System Python 3.13 passes the 20 standard-library tests and
skips the three PyTorch checks. No dependency installation is required.

Next, inspect the smoke plan (this command does not import GPU libraries):

```bash
python experiments/limited_angle_generalization/smoke_gpu.py \
  --report output/generalization_backend_smoke_20261005_v1.json
```

After confirming the workstation is available, the exact tiny GPU command is:

```bash
python experiments/limited_angle_generalization/smoke_gpu.py --execute \
  --report output/generalization_backend_smoke_20261005_v1.json
```

The corrected smoke passed on the workstation GPU after isolating the two backends
in separate subprocesses. See `smoke_fix.md` and `manifests/backend_smoke_isolated.json`.
It queries one Gaussian on
16×20×24 voxels, checks known peak location/finite parameter gradients, and compares
wrapped/unwrapped TIGRE projections of a tiny phantom. Reserve five minutes for
imports and execution. The report refuses overwrite; use a new suffix for a
deliberate repeat. It is backend correctness evidence only, not reconstruction
performance, an FDK-context regression or a complete regularizer gradient test.

To prepare a changed config, resolve it read-only first and then freeze to a new
filename. The output parent must already exist; existing manifests are never
overwritten:

```bash
python experiments/limited_angle_generalization/prepare.py \
  --config experiments/limited_angle_generalization/configs/development.json --dry-run
python experiments/limited_angle_generalization/prepare.py \
  --config experiments/limited_angle_generalization/configs/development.json \
  --write experiments/limited_angle_generalization/manifests/development_v4.json
```

Code or source changes invalidate a frozen manifest: review them and create a new
snapshot. Do not bypass validation or replace the working CUDA/TIGRE stack.

## Remaining scientific and execution questions

- Only one independent reference is present; obtain and assign additional source
  volumes with verified identity/provenance before a population generalization test.
- SPARK full text/supplement remains unavailable in this session. Its abstract
  already establishes close geometry-conditioned Gaussian prior art.
- A ray-direction moment is not a cone-beam visibility mask. Validate a descriptor
  against operator perturbation responses before selecting penalty direction.
- Initializer point placement omits the half-voxel offset used by query sampling;
  investigate with controlled phantoms before changing this inherited convention.
- Determine whether a gain survives tuned ordinary TV, uniform-strength and wrong-
  orientation controls while preserving boundaries at frozen settings.
- Implement shared-evaluation generation, dataset verification and a guarded
  runner/evaluator before a development pair. Preserve initializer pairing, log
  process exit status and measure peak VRAM. Exact checkpoint continuation is
  currently unverified.

The first subsequent reconstruction pair should be one baseline and one tuned-TV
setting on development data after backend/data correctness gates: roughly 38–46
minutes of training, reserve one hour plus generation/init overhead, and at least
1 GiB free for its datasets/checkpoints/diagnostics. This is an estimate from
pilot logs, not authorization or a command to launch a full sweep.

The earlier `development_v2.json` is superseded by v3 after the subprocess-isolation fix.
Historical check logs retain their original results; they do not describe the fixed smoke.
