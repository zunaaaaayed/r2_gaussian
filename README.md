# TRDP2 — Limited-angle CT reconstruction generalization

TRDP2 is [Md Zunayedul Islam](https://github.com/zunaaaaayed)'s research project
on Gaussian-based CT reconstruction across limited-angle acquisition configurations
and independent volumes. This repository is maintained for TRDP2 research and
experiments; it is not the official R²-Gaussian project or an upstream contribution.

The reconstruction implementation originates from
[R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) by Ruyi Zha and collaborators.
TRDP2 adds acquisition-controlled experiments, reproducibility checks, guarded
execution, evaluation, and research protocols. Original authorship, commit history,
and license notices are retained.

## Research objective

Develop and evaluate a reconstruction rule whose settings can be selected on
development data and then frozen across acquisition orientations, angular spans,
and independent volumes. Each scan has its own reconstruction optimization.

Geometry-dependent regularization is a **research hypothesis**. Neither its
benefit nor its novelty has been established. Ordinary-TV controls, boundary
preservation, independent-volume evaluation, and relevant prior work are essential
to assessing that hypothesis.

## Current results and status

Two development baselines completed 30,000 iterations with 50 views, a 120° span,
TV weight 0.05, seed zero, and the same chest reference and evaluation projections:

| Acquisition start | Volume PSNR | Slice-averaged SSIM | RMSE | Boundary MAE |
|---|---:|---:|---:|---:|
| 0° | 30.913 dB | 0.9010 | 0.02847 | 0.03880 |
| 90° | 28.591 dB | 0.8624 | 0.03719 | 0.05654 |

These reproduce the completed pilot and demonstrate orientation sensitivity on
**one development volume**. They do not establish independent-volume generalization.
Metrics use the supplied unit-range reference without clipping predicted values.
See the protocol for the SSIM and boundary definitions.

As of 7 October 2026, GPU work is stopped for a workstation transfer. Four
ordinary-TV controls (weights 0.025 and 0.1 at both orientations) are prepared but
have not started. Reserved acquisitions are untouched. No additional independent
volumes have been downloaded. Do not restart the archived queue scripts on the
old workstation.

## Project guide

- [Workstation handoff and exact continuation steps](experiments/ordinary_tv_controls/HANDOFF.md)
- [Generalization protocol and split rules](experiments/limited_angle_generalization/protocol.md)
- [Benchmark preparation and verification](experiments/limited_angle_generalization/README.md)
- [Guarded single-case workflow](experiments/limited_angle_generalization/workflow.md)
- [Ordinary-TV controls and selection rule](experiments/ordinary_tv_controls/README.md)
- [Independent-volume data research and intake plan](experiments/ordinary_tv_controls/independent_volumes.md)
- [Saved results and transfer inventory](experiments/ordinary_tv_controls/session_20261007/)
- [Historical limited-angle pilot](experiments/limited_angle_120/README.md)
- [Initial baseline reproduction](experiments/baseline_chest_50views/README.md)
- [Research handoff specification](TRDP2_CODEX_HANDOFF.md)

## Setup and safe continuation

```bash
git clone --recursive https://github.com/zunaaaaayed/trdp2.git
cd trdp2
```

The verified university stack used Python 3.9.23, PyTorch 2.1.2+cu118, CUDA 11.8,
NumPy 1.24.4, Open3D 0.18, and an RTX 4000 Ada GPU. The local Conda environment is
named `trdp2-r2`. Native extensions and TIGRE must be checked on a new computer;
copied compiled binaries are not assumed portable. Follow the
[handoff](experiments/ordinary_tv_controls/HANDOFF.md) before GPU execution.
The Python package remains named `r2_gaussian` to preserve the verified baseline.

After activating the research environment:

```bash
python -m unittest discover -s experiments/limited_angle_generalization/tests -v
python -m unittest discover -s experiments/ordinary_tv_controls -p 'test_*.py' -v
# Read-only paired-control preflight; transferred data is required:
python experiments/ordinary_tv_controls/run.py --case chest_start0_span120 --weight 0.025
```

Execution requires an explicit `--execute`. Existing destinations are rejected;
exact checkpoint continuation is not verified. TIGRE forward projection is
isolated from live PyTorch CUDA tensors because its context reset caused a
segmentation fault in the original combined smoke process.

Data, checkpoints, logs, and compiled extensions are excluded from Git. Transfer
the separately checksummed artifact archive described in the handoff as well as
this repository. Frozen source/data hashes must remain valid. Historical records
retain their original repository name and workstation paths for provenance.

Original installation options, data-generation instructions, and CLI reference
are preserved in [README_UPSTREAM.md](README_UPSTREAM.md); these are inherited
upstream documentation, not claims about TRDP2 results.

## Attribution and license

TRDP2 research extensions are maintained by Md Zunayedul Islam. The original
R²-Gaussian implementation and its dependencies retain their authorship and
applicable terms. See [LICENSE.md](LICENSE.md) and the license files in each
submodule. Renaming this project does not replace those licenses or transfer
ownership of the original implementation.

The inherited implementation builds on
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF),
[NAF](https://github.com/Ruyi-Zha/naf_cbct), and
[TIGRE](https://github.com/CERN/TIGRE).

When using the original reconstruction method, cite its paper:

```bibtex
@inproceedings{r2_gaussian,
  title={R$^2$-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction},
  author={Ruyi Zha and Tao Jun Lin and Yuanhao Cai and Jiwen Cao and Yanhao Zhang and Hongdong Li},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

For TRDP2 experiments, additionally reference this repository and the exact commit
and protocol used. No separate TRDP2 method publication is claimed here.
