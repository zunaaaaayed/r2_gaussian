# Repository and workstation audit — 2026-10-05

## Checkout and preservation

Host: `cheonech`. Requested workspace path resolves to
`/autofs/unitytravail/travail/mdzislam/trdp2/r2_gaussian`.
Branch: `experiment/limited-angle-120`; HEAD
`327a731834ad3e4f6b89a1e993df01cd3fa1bb6d` (Add limited-angle rotation and
FDK comparison diagnostics). Origin:
`https://github.com/zunaaaaayed/r2_gaussian.git`.

At entry, tracked files were clean. Existing untracked user work was
`.graphifyignore`, `TRDP2_CODEX_HANDOFF.md`, and `graphify-out/`. None was discarded,
stashed, committed or pushed. No branch switch was needed for this additive
experiment layer. No applicable AGENTS.md was found in the repository or either
logical/resolved ancestor chain. `.gitignore` already excludes data, output,
logs and compiled products; no ignore/licensing/submodule configuration changed.

Submodules: simple-knn `86710c2d4b46680c02301765dd79e465819c8f19`; GLM
`33b4a621a697a305bc3a7610d290677b96beb181`. Both had normal initialized status.

Before editing the continuation note, 610 files (1,498,699,590 bytes) across the
pilot experiment directory, three controlled datasets, three completed outputs
and diagnostic output tree were SHA-256 hashed. Snapshot:
`manifests/pilot_preservation.json` (also initially `/tmp/trdp2-pilot-before.json`).
The only intended change in that snapshot is `limited_angle_120/CONTINUE_HERE.md`.
The final rehash confirmed exactly that change; the other 609 files matched.
Historical code, patch, metric files, volume arrays, projections, initializers,
checkpoints and diagnostic figures are preserved.

Graphify was used for navigation only. Its existing health record reports 583
dangling endpoint edges, 3 self-loops and 61 undirected endpoint collapses in its
extraction diagnostic; the graph may be incomplete. Findings were checked against
actual source and saved artifacts. Query/reflection generated local graph cache/
reflection files; no graph rebuild or scientific inference from graph edges.

## Environment

| Item | Observed |
|---|---|
| Default Python | `/usr/bin/python3`, 3.13.5; no PyTorch |
| Research interpreter | `~/espaces/travail/trdp2/tools/miniforge3/envs/trdp2-r2/bin/python`, 3.9.23 |
| Packages | PyTorch 2.1.2+cu118; NumPy 1.24.4; Open3D 0.18.0; PyYAML 6.0.2 |
| CUDA compiler / Torch build | nvcc 11.8.89 / CUDA 11.8 |
| Driver / device | Read-only unsandboxed nvidia-smi: driver 615.71.09, RTX 4000 Ada, 20,475 MiB |
| GPU state at inspection | 515 MiB used, 0% utilization; graphical processes only. Snapshot, not a reservation |
| Sandbox limitation | nvidia-smi failed inside the sandbox; succeeded with approved outside-sandbox read-only check. Do not diagnose this as a broken CUDA installation |
| TIGRE | Package located in trdp2-r2 site-packages; version distribution metadata absent. Source `../tools/TIGRE-v2.3`, commit `e33ee191454a096e40d4ab92f2fa42f5a1f4a74e` |
| Extensions | Rasterizer Python package resolves to local submodule; simple_knn is a namespace package. Actual CUDA execution remains a separate smoke check |

TIGRE source has the context-reset removal at
`Common/CUDA/voxel_backprojection.cu:614`. Ignoring line endings, its diff is
exactly replacement of `cudaDeviceReset()` with preservation comments. Ordinary
diff counts many changed lines due to line endings. The tracked patch and working
installation were not reapplied, rebuilt or upgraded. The saved context-check
initializer and original control initializer have the identical SHA-256
`eff857f30a32c4e7d9ed2fa10799167d73be0ec69afb1f837dc6f06941759376`.
Historical log/README evidence reports initializer exit 0; no new initializer run
or proof of installed binary/source equivalence is claimed here.

## Data and completed evidence

Only `data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone` and its three
controlled pilot derivatives were found via metadata inventory. All four reference
arrays are finite float32, 256³, range [0,1]. Patient identity is unknown. Original
metadata has noise=true; controlled pilot metadata has noise=false. New planned
data derive from the reference, not the original noisy measurements.

Scanner: cone, DSO=5, DSD=7; 512² detector of extent 4×4; volume extent 2×2×2;
zero offsets; projector accuracy 0.5. Length calibration is unspecified. Pilot
reference SHA-256:
`c895f1f7127a2ba7b25330240e16cd28b75286e037d4bcd9419a63282bd5dedd`.
The three references and all shared held-out projection files agree byte-for-byte.

| Acquisition | Saved final PSNR | Saved slice-aggregated SSIM | Log start to completion | Output disk |
|---|---:|---:|---:|---:|
| 360°, start 0° | 35.54680633544922 | 0.94193035364151 | 18m57s | 189 MiB |
| 120°, start 0° | 30.906631469726562 | 0.900951623916626 | 22m15s | 222 MiB |
| 120°, start 90° | 28.594484329223633 | 0.8623384634653727 | 19m25s | 220 MiB |

All three have iteration-30000 volume/eval files, completion logs and saved
checkpoints. FDK and rotation diagnostics are completed in ignored local output
storage; the tracked `results/` tree is less complete. Saved raw-FDK PSNR values
24.16951069526235 / 19.256990392660512 and rotated outside/control MAE ratio
3.114293888654874 agree with the handoff. Pilot sector counts are 100/0, 33/67,
33/67. Saved FDK-vs-final and rotation protocols both report reference reprojection
MAE=max error=0 with tolerance 1e-5. These are historical internal checks, not new
GPU reprojections or physical ground truth validation.

Metric precision differences are expected: diagnostic float64-derived PSNR
35.54680512313456 versus repository float32 PSNR 35.54680633544922, for example.
They are not substituted for one another. No missing rotated MAE/RMSE is invented.

Control/original/rotated initializers are 50,000×4. The final number of Gaussians
differs from the initializer; do not infer an exact count from rounded progress
log `pts` fields. Pilot datasets total 656 MiB and diagnostics 147 MiB. Peak VRAM
was not recorded. Logs document the earlier interrupted/replaced attempt; it is
not independent evidence. Resume reproducibility is unverified because checkpoints
do not save full RNG/camera-stack state.

## Verification scope

The standard-library benchmark tests exercise angle units, half-open boundaries,
wraparound, circular overlap, common-grid selection, joint sector rotation,
duplicate cases, patient/hash leakage, input/config tampering, physical spacing,
unsupported seed/feature rejection, dry-run nonmutation and overwrite refusal.
CPU PyTorch checks verify baseline TV values and gradients and fixed-range PSNR
without clipping. Actual pilot inputs are independently checked by
`prepare.py --check-pilot`. The opt-in backend smoke is prepared separately;
no full reconstruction or geometry regularizer experiment is authorized by a
successful preparation check alone.

Final verification: all 21 tests passed in Python 3.9.23; all new Python compiled
and parsed for Python 3.9; the current `development_v2.json` and actual pilot checks
passed; the earlier v1 snapshot was rejected after a source change; exclusive
manifest overwrite refusal and GPU-smoke dry-run nonmutation passed. `git diff
--check` passed and core training/initialization code has no diff. All 851 saved
input-hash entries from four diagnostic protocols matched. Logs and a structured
summary are under `manifests/final_check_*.txt` and `manifests/checks.json`.
