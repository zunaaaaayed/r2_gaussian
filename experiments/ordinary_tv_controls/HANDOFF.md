# Workstation transfer handoff — 7 October 2026

User requested all training stop and continuation on another computer. The
`trdp2-tv-controls-20261007` queue was interrupted while waiting for GPU availability.
No ordinary-TV control had started and no partial TV checkpoint exists. No other
GPU workload was stopped. The unrelated interactive `trdp2-generalization` tmux
session was left untouched. Do not restart a queue on the old workstation.

## Completed this session

Two development baselines (50 views, 120° span, seed 0, TV 0.05, 30,000 iterations)
completed generation, verification, FDK initialization, training and evaluation.
Their dataset, initializer, final checkpoint, volume and metric hashes were
verified. The same reference and 100 evaluation projections are used for both.

| Start angle | PSNR | SSIM | RMSE | Boundary MAE | Training seconds |
|---|---:|---:|---:|---:|---:|
| 0° | 30.912636 | 0.901018 | 0.028469 | 0.038802 | 3391.39 |
| 90° | 28.591247 | 0.862378 | 0.037191 | 0.056537 | 2638.93 |

These reproduce the earlier pilot and show orientation sensitivity on one volume.
They establish neither independent-volume transfer nor regularizer efficacy or
novelty. GPU contention was uncontrolled; runtimes are not a speed comparison.
Boundary preservation remains a concern. Independent TIGRE projection metrics and
peak VRAM remain unmeasured. Native Gaussian-render metrics use image normalization.

Prepared `run.py`, three guard tests, the predeclared TV selection rule in README,
and the public-data shortlist in `independent_volumes.md`. Four new TV jobs remain:
weights 0.025 and 0.1, each at starts 0° and 90°. Both use existing exact initializer
bytes. Reserved acquisitions are untouched. No independent volume was downloaded.

## Saved artifacts and transfer

`session_20261007/` contains completion records, final metrics, intermediate volume
metrics, stopped-queue record, archival queue scripts, and a SHA-256 inventory.
The queue scripts capture the OLD workstation paths: do not execute them unchanged
on a different computer. Large artifacts remain outside Git, as before.

A separate uncompressed archive is saved on the old workstation:
`output/workstation_transfer_20261007/session_artifacts.tar`.
Its SHA-256 and byte count are in `session_20261007/archive.json`.
It includes source chest data needed for v4, both new datasets/initializers,
shared evaluation data, both complete run outputs/checkpoints, job provenance/logs,
smoke reports, queue logs and slice diagnostics. The historical pilot remains
unchanged on the old workstation and is not duplicated in this continuation
archive. Copy the old `data/trdp2/chest_50views_noisefree/` and historical pilot
outputs separately if you need to re-run the pilot audit on the new host.

Transfer the Git branch AND the archive using your normal secure file-transfer
route. Check the archive hash, extract only into a fresh checkout/data location,
and verify every path against `transfer_inventory.json`. Git alone does not
transfer ignored data or checkpoints. Absolute paths in historical context records
are historical evidence and should not be rewritten.

## Next computer: preflight before GPU execution

Install/activate a compatible research environment and build the native extensions
for the target GPU; do not assume old compiled binaries are portable. Preserve and
record the TIGRE context-preservation patch and its source provenance. The original
stack is Python 3.9.23, torch 2.1.2+cu118, CUDA 11.8, NumPy 1.24.4, Open3D 0.18.
TIGRE forward projection must remain isolated from live PyTorch CUDA work.

From the repository root, after transferring artifacts and activating that environment:

```bash
python -c 'import sys, numpy, torch, tigre; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())'
python -m unittest discover -s experiments/limited_angle_generalization/tests -v
python -m unittest discover -s experiments/ordinary_tv_controls -p 'test_*.py' -v
python experiments/limited_angle_generalization/smoke_gpu.py --report output/backend_smoke_new_workstation_v1.json
python experiments/limited_angle_generalization/smoke_gpu.py --execute --report output/backend_smoke_new_workstation_v1.json
python experiments/ordinary_tv_controls/run.py --case chest_start0_span120 --weight 0.025
```

Only after the new environment and smoke pass, inspect GPU availability and run a
single control in tmux by adding `--execute` to the last command. Then continue
start90/0.025, start0/0.1 and start90/0.1 serially, checking each completion record.
No automatic cross-host resume is supported. Freeze the selected TV rule before
reserved acquisitions or independent-volume testing.

The old baseline runner validates an exact source-file set: adding tracked control
Python files means it can reject v4. The new control runner instead structurally
validates v4, verifies EVERY originally frozen file and submodule identity, and
records additive code separately. It never permits edits to frozen baseline code.
Do not regenerate historical manifests or disable integrity checks to migrate.
