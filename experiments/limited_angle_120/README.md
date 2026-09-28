# Controlled limited-angle experiment: 120 degrees

## Question

How does restricting angular coverage affect the original R2-Gaussian
reconstruction pipeline when the training projection count stays fixed?

## Protocol

- Reference anatomy and scanner geometry: supplied 50-view chest case.
- Regenerate both cases with TIGRE, without added noise.
- Control: 50 views over [0, 360) degrees, spacing 7.2 degrees.
- Limited-angle: 50 views over [0, 120) degrees, spacing 2.4 degrees.
- Start angle: 0 degrees. Right endpoint excluded.
- Last sampled angles: 352.8 degrees and 117.6 degrees.
- Shared evaluation: 100 full-circle midpoint angles.
- Reference volume and evaluation projections are evaluation-only inputs
  during reconstruction; they generate the simulated dataset beforehand.
- Generate initialization separately from each case's training projections.
- Initialization seed: original initialize_pcd.py seed, 0.
- Reconstruction random seeds: original train.py/safe_state implementation.
- Original optimizer, regularization, and density-control defaults.
- Final checkpoint fixed at 30,000 iterations.
- No per-case tuning using reference-volume scores.

## Interpretation

This is a single-anatomy feasibility experiment.
It tests angular coverage at fixed view count, not fixed angular spacing.
It includes the effect of coverage on FDK-based initialization.
Limited-angle FDK is an incomplete-data initializer, not an exact inversion.

Compare these two regenerated cases with each other.
The earlier supplied-data reproduction is a separate experiment.

Generated-data hashes and provenance are stored within each data directory.
Small floating-point differences across hardware/software may remain.
