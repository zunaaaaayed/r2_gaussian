# Baseline: chest, 50 views, full-angle cone-beam CT

## Purpose

Reproduce the original R2-Gaussian pipeline before introducing limited-angle
acquisition or a new reconstruction method.

## Data

Official synthetic dataset:
https://drive.google.com/drive/folders/1K3-8RECm-MwUpphIXEeqWznHpTdv4sHz

Archive: cone_ntrain_50_angle_360.zip
Case: 0_chest_cone

Expected local directory:
data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone

The reference volume is used for evaluation. The supplied initialization is
used for this baseline. Training uses 50 projections distributed around the
full circular trajectory. This is sparse-view CT, not limited-angle CT.

dataset_metadata.json records the acquisition.
dataset.sha256 identifies the exact input files.

Verify inputs from the repository root:

    (cd data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone &&
     sha256sum -c ../../../../experiments/baseline_chest_50views/dataset.sha256)

## Working environment

- Linux
- NVIDIA RTX 4000 Ada Generation, approximately 20 GB VRAM
- Python 3.9.23
- PyTorch 2.1.2+cu118
- NumPy 1.24.4
- CUDA toolkit 11.8
- GCC/G++ 11
- TIGRE source checked out at tag v2.3
- R2-Gaussian CUDA extensions compiled locally

Environment snapshots and source revisions are stored alongside this file.
Some snapshots contain absolute local paths and require adaptation elsewhere.

## Baseline command

From the repository root, with the trdp2-r2 environment active:

    export MPLBACKEND=Agg
    python train.py \
      -s data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone \
      -m output/chest_50views_baseline \
      --eval \
      --iterations 30000 \
      --test_iterations 5000 10000 20000 30000 \
      --save_iterations 30000 \
      --checkpoint_iterations 10000 20000 30000

Use a new output directory for each repeated run.
All other reconstruction parameters use the checked-out implementation's defaults.
Random-number initialization follows the original training script.

## Results and interpretation

metrics.yml contains the final 30,000-iteration evaluation.
The checkpoint was fixed in advance, not selected by best reference-volume PSNR.

Reported psnr_3d: 35.17658996582031 dB.
Reported ssim_3d: 0.9353867769241333.

The SSIM aggregate comes from slice-based evaluation across three axes.
This is a single-case, single-run reproduction, not evidence of a new method.
Exact numerical equality across machines is not guaranteed.

## Planned limited-angle experiment

Start with 50 projections over a 120-degree acquisition arc.
Generate a matched full-angle control using the same projection/noise protocol.
Specify the arc start angle and endpoint convention explicitly.
Regenerate initialization using only each case's available training projections.
Keep reference volumes and unavailable projections out of optimization and
initialization. Use them only for evaluation.
