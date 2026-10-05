# TRDP2 Codex handoff: reconstruction generalization under limited-angle acquisition

Prepared: 5 October 2026.
Owner: Zunayed (Md Zunayedul Islam), Erasmus Mundus Joint Master in Artificial Intelligence for Image Processing and Computer Vision, Université de Bordeaux.
Supervisors: Dr. Pascal Desbarats and Ms. Anna-Louise Brun, LaBRI / Université de Bordeaux.
Working repository: https://github.com/zunaaaaayed/r2_gaussian.git
Previously inspected experiment branch: `experiment/limited-angle-120`.

## 1. Your assignment

Act as a careful research software collaborator. Continue this project from the completed pilot toward improved reconstruction generalization under limited angular acquisition. Read this entire handoff, inspect the actual repository and its applicable AGENTS.md instructions, and reconcile the current state before making changes.

The supervisors recommended that reconstruction generalization become the main objective now that the project has narrowed to missing-angle / missing-wedge reconstruction. Earlier ideas about uncertainty estimation, evidential deep learning, adaptive acquisition, and projection reliability are background, not the immediate implementation target.

Main question:

> Can an acquisition-dependent reconstruction strategy improve R²-Gaussian reconstruction across angular spans, acquisition orientations, and independent volumes, with settings selected on development cases and then held fixed?

Initial candidate:

> Geometry-dependent regularization for limited-angle 3D Gaussian cone-beam CT reconstruction, without requiring a patient-specific prior CT or a large population training dataset.

This candidate is a hypothesis, not a verified novel method. Do not invent a novelty claim or implement a large architecture before checking the closest literature and validating the geometric reasoning.

Proceed autonomously with reversible repository inspection, documentation, protocol design, and benchmark infrastructure. Do not stop after merely outlining a plan. The first work session has a deliberately bounded deliverable in Section 17: complete that deliverable, then report what is ready and what still requires university GPU execution. Do not interpret this handoff as authorization to launch a large GPU sweep, overwrite results, or publish unreviewed changes.

## 2. Research background and scope

TRDP0 was too broad. TRDP1 mainly involved 2D Gaussian reconstruction and classical CT baselines. TRDP2 now uses an existing 3D Gaussian CT implementation and should produce a focused, reproducible investigation.

R²-Gaussian represents attenuation with trainable 3D Gaussian primitives. Optimization adjusts their positions, density, scale, and orientation using measured projections, with adaptive Gaussian density control and regularization. Its rectified radiative projection formulation is designed for tomography.

The present implementation optimizes a new Gaussian representation for each scan. It is not a pretrained cross-patient reconstruction network. Therefore:

- Per-scan optimization at evaluation is expected.
- Generalization initially means algorithmic robustness: the same rule, hyperparameters, and evaluation protocol work on new cases without ground-truth-based retuning.
- If a learned shared prior is introduced later, explicitly define population training/validation/test splits and distinguish them from projection splits within a scan.
- Improved average PSNR on previously inspected pilot cases alone is insufficient evidence of generalization.

Priority order:

1. Different acquisition starting angles / orientations.
2. Different acquired angular spans.
3. Independent volumes or anatomies.
4. Noise and scanner geometry changes as later extensions.

Do not expand this into simultaneous motion compensation, multimodal learning, uncertainty estimation, diffusion training, and adaptive scanning.

## 3. Terminology and physics

Our actual setup is 3D cone-beam CT on a circular source trajectory. Use “limited-angle cone-beam CT” for precise claims.

The standard Fourier missing-wedge picture is most directly associated with parallel-beam tomography. Do not equate a missing source-angle interval with an exact global Fourier mask in cone-beam CT. Do not label the 120° acquisition as having a “240° Fourier missing wedge.” Cone-beam redundancy, local ray directions, detector coverage, and trajectory completeness matter. Even full circular cone-beam coverage is not an exact complete acquisition for arbitrary 3D objects.

The 360° reconstruction is our matched sparse-view control, not ground truth and not a dense-view ideal reconstruction. The known reference volume is the evaluation ground truth.

Measured-data consistency cannot guarantee anatomical correctness when data are incomplete. Avoid claims that a fidelity term eliminates all hallucinations or uniquely determines invisible structures.

Distinguish two different rotation tests:

- Rotate acquisition around a fixed object: information content can genuinely change. Quality need not be invariant. This is what the pilot tested.
- Jointly rotate object and scanner coordinates: a coordinate-equivariance check; apart from discretization and numerical effects, this should describe an equivalent problem.

Do not confuse these experiments or use coordinate invariance as a reason to expect equal pilot PSNR.

## 4. Execution environment

Historical university environment, to verify before use:

- SSH username/host used in previous commands: `mdzislam@cheonech`.
- Repository path: `~/espaces/travail/trdp2/r2_gaussian`.
- Conda environment: `trdp2-r2`.
- Conda initialization: `~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh`.
- GPU: NVIDIA RTX 4000 Ada, approximately 20 GB VRAM.
- Python 3.9.23; PyTorch 2.1.2 with CUDA 11.8; TIGRE v2.3.
- tmux session: `trdp2`.
- Headless plots: `MPLBACKEND=Agg`.

Typical activation on the university machine:

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLBACKEND=Agg
```

Do not assume the Codex workspace is this university machine. Determine hostname, current checkout, available data, Python, CUDA, and GPU access. Do not install Linux/CUDA dependencies on the user's Mac as if it were the university server. If GPU/data are unavailable, complete CPU-side implementation and checks and provide exact remote commands.

A TIGRE initialization crash was traced to a CUDA-context reset. The experiment directory contains `tigre-v2.3-preserve-cuda-context.patch`. Removing the reset and rebuilding gave initializer exit code 0 and an exactly identical saved initialization in the verification comparison. Preserve and document the patch; do not blindly reapply it or replace the working TIGRE installation.

A historical run was interrupted after SSH disconnected, around iteration 8,120, with no resumable checkpoint. It was archived and replaced. Use actual logs to distinguish interrupted/restarted/resumed runs. Do not describe that interruption as a new independent seed replicate.

## 5. Repository audit and preservation rules

Start with read-only checks such as:

```bash
git status --short
git branch --show-current
git remote -v
git log -5 --oneline
rg --files -g 'AGENTS.md' -g 'README*' -g '*CONTINUE*' -g '*protocol*'
```

Read applicable instructions, including parent-directory instructions when present. Inspect tracked and untracked changes before changing branches. Do not reset, clean, stash, or discard user work automatically. Create an isolated branch/worktree if necessary; a possible new branch is `experiment/limited-angle-generalization`, but do not assume it already exists or overwrite it.

Known inspected files on the prior experiment branch:

- `train.py`
- `initialize_pcd.py`
- `experiments/limited_angle_120/README.md`
- `experiments/limited_angle_120/CONTINUE_HERE.md`
- `experiments/limited_angle_120/generate.py`
- `experiments/limited_angle_120/generate_rotated.py`
- `experiments/limited_angle_120/inspect_failure.py`
- `experiments/limited_angle_120/inspect_projection_angles.py`
- `experiments/limited_angle_120/inspect_fdk.py`
- `experiments/limited_angle_120/compare_fdk_final.py`
- `experiments/limited_angle_120/compare_acquisition_rotation.py`
- `experiments/limited_angle_120/results/`
- Small metric, initialization, and dataset provenance files in the same directory.

The prior inspection found `CONTINUE_HERE.md` stale: it still described the raw-FDK comparison as the next task even though that comparison and the rotated acquisition were subsequently completed. Reconcile with saved evidence. Update the note to reflect the completed pilot and the new objective without rewriting historical measurements.

Large datasets, volume arrays, checkpoints, and compiled extensions may exist only on the university workstation. Their absence from Git is expected. Do not commit them. Preserve existing licensing and submodule arrangements. Inspect .gitignore before producing outputs.

This handoff provides reported measurements and a design brief. For actual files and current state, fresh repository inspection and saved protocols are authoritative. If results disagree, record the discrepancy and investigate rather than silently replacing one value.

## 6. Completed pilot: data and acquisition

One chest case: `0_chest_cone`, supplied with the R²-Gaussian dataset under the original `cone_ntrain_50_angle_360` setup.

- Reference volume: 256 × 256 × 256 voxels.
- Intensities: supplied dataset values, not calibrated HU.
- Physical scanner geometry: retained from the supplied case.
- Controlled measurements: regenerated from the reference using TIGRE.
- No added projection noise.
- 50 training projections per reconstruction.
- 100 shared full-circle held-out evaluation projections.

| Case key | Data subfolder | Acquired interval | Step | Actual last training angle |
|---|---|---|---|---|
| control_360 | arc_360 | [0°, 360°) | 7.2° | 352.8° |
| start_0 | arc_120 | [0°, 120°) | 2.4° | 117.6° |
| start_90 | arc_120_start_90 | [90°, 210°) | 2.4° | 207.6° |

Dataset base: `data/trdp2/chest_50views_noisefree/`.
Each subfolder contains `0_chest_cone/`.

Training formula: start + arange(50) × span / 50; right endpoint excluded.
Evaluation angles: `(arange(100) + 0.5) × 3.6°`, i.e. 1.8° through 358.2°.
Metadata projection angles are in radians. Scanner startAngle/totalAngle fields in the existing generator are in degrees. Inspect actual conventions rather than assuming every field uses the same units.

The three configurations share reference volume and evaluation projections. For the rotated case, reference and test files were copied and hash-verified. Training/test angular overlap checks used circular angular separation.

Both 120° cases contain 33 held-out directions inside the acquired interval and 67 outside. A previous conversational statement of 34 inside views for the rotated case was corrected; do not repeat it.

Fixed count does not mean fixed angular sampling density: 360° and 120° cases have different angular spacing. This comparison measures coverage at a matched view count.

## 7. Completed pilot: reconstruction and evaluation

Each acquisition had its own FDK initialization from its own training measurements:

- 50,000 initial Gaussian locations.
- FDK density threshold: 0.05.
- Initial density multiplier: 0.15.
- 30,000 training iterations.
- Original optimization/density-control settings and TV regularization.
- Fixed final checkpoint evaluation, not best test-reference checkpoint selection.

These values do not imply the final Gaussian count stayed at 50,000. Adaptive density control may change it.

Known output directories:

```text
output/chest_50views_noisefree_arc360/
output/chest_50views_noisefree_arc120/
output/chest_50views_noisefree_arc120_start90/
```

Final volume files were read from `point_cloud/iteration_30000/vol_pred.npy` and `vol_gt.npy`. Repository metrics were read from `eval/iter_030000/eval3d.yml`.

Diagnostics:

```text
output/limited_angle_120_diagnostics/
output/limited_angle_120_diagnostics/fdk/
output/limited_angle_120_diagnostics/fdk_vs_final/
output/limited_angle_120_diagnostics/acquisition_rotation/
```

Raw FDK was evaluated before thresholding, Gaussian point sampling, and density rescaling. Negative values were retained in numerical metrics. Raw-FDK versus final-Gaussian comparisons therefore include initialization conversion, representation, and optimization effects.

The angular diagnostic projects saved voxel volumes with TIGRE. It is distinct from the Gaussian rasterizer's own projection metrics/training loss. Do not conflate them.

No independent intensity normalization or numerical clipping was applied to these diagnostics. Visual display windows may clip for presentation; metric calculations must not silently do so.

Initial reference reprojection check: MAE 0, maximum absolute error 0. This is internal consistency with generation, not an independent physical validation. Inspect subsequent protocol.json records for their own checks.

SSIM is the repository's slice-wise aggregation across array axes, not a true volumetric sliding-window 3D SSIM. Verify and record the exact metric implementation and data range in all new experiments.

## 8. Frozen pilot measurements

### Volume quality

| Acquisition | Method | PSNR dB | Slice-aggregated SSIM | MAE | RMSE |
|---|---|---:|---:|---:|---:|
| 360° | Raw FDK | 24.17 | not reported | 0.04700 | 0.06188 |
| 360° | Final R² | 35.546806 | 0.941930 | 0.00817 | 0.01670 |
| 120°, start 0° | Raw FDK | 19.26 | not reported | 0.08610 | 0.10893 |
| 120°, start 0° | Final R² | 30.906631 | 0.900952 | 0.01370 | 0.02849 |
| 120°, start 90° | Final R² | 28.594484329223633 | 0.8623384634653727 | not reported | not reported |

Rotated SSIM by array axis:

```yaml
ssim_3d_x: 0.8704048991203308
ssim_3d_y: 0.8545371294021606
ssim_3d_z: 0.8620733618736267
```

Some tabulated values are rounded report values. Use original files for numerical analysis. Do not invent missing measurements; label any mathematically derived quantities explicitly.

### Shared held-out voxel-volume projection results

```csv
case,sector,view_count,mae,rmse,signed_bias,matched_control_mae,ratio_of_mean_mae_to_control
control_360,all,100,0.0011696022446350568,0.0021165218260576725,0.0002927343442449698,0.0011696022446350568,1.0
start_0,all,100,0.0024525842642721457,0.005599465454714629,0.00013807908380047406,0.0011696022446350568,2.0969387460755167
start_0,inside,33,0.0012722255895469552,0.0023503500624370986,0.0001707805643946821,0.001199024446425977,1.0610505843639588
start_0,outside,67,0.003033954954808434,0.006638989919493705,0.00012197238440332681,0.0011551107124096781,2.6265490590761607
start_90,all,100,0.002867667656436987,0.0072000248243381144,7.997498027528993e-06,0.0011696022446350568,2.4518315261371324
start_90,inside,33,0.0011946878626777214,0.0023269151136890973,3.5277097589021616e-05,0.001137535410932592,1.050242349553122
start_90,outside,67,0.0036916726294825963,0.00864331000623502,-5.438722652012148e-06,0.0011853963567571664,3.114293888654874
```

Use 3.1143× for the rotated outside/control ratio. A secondary memory summary contained 3.19×; the explicit CSV above is the reported evidence.

Peak summary:

```csv
case,relative_sector,peak_absolute_degrees,peak_relative_degrees,peak_mae,mae_ratio_to_control
start_0,"[0,180)",145.8,145.8,0.00558223492233261,4.999653598298216
start_0,"[180,360)",329.40000000000003,329.40000000000003,0.005788624159990612,4.878548788832201
start_90,"[0,180)",243.0,153.0,0.006173545742323125,5.473550453406034
start_90,"[180,360)",66.60000000000001,336.6,0.006349016756320557,5.270388995926395
```

Relative angle = `(absolute_angle - acquisition_start) % 360`.
Each dominant peak shifted by 97.2° after a 90° acquisition rotation. Residual difference in relative coordinates: 7.2°, or two evaluation-grid steps. This is descriptive approximate alignment, not a statistically established invariant.

Other reported observations:

- Full-circle versus original limited-angle final PSNR difference: about 4.64 dB.
- Original versus rotated limited-angle final PSNR difference: about 2.31 dB, approximately 1.70× voxel MSE.
- FDK-to-final mean projection MAE: control approximately 0.01254 to 0.001170; original limited-angle approximately 0.04330 to 0.002453.
- Raw FDK negative fractions: approximately 0.3014 and 0.3523 for control/original limited case.
- FDK voxels above 0.05: 8,756,882 and 9,360,026 respectively. Eligible does not mean accurate anatomy.
- Original limited-angle final volume signed error: approximately -0.00489 where reference intensity > 0.05, +0.00367 elsewhere. These are exploratory threshold regions, not anatomical segmentations.

## 9. What the pilot establishes and does not establish

Supported observations for this one volume:

- The Gaussian pipeline substantially improves raw FDK.
- Limited-angle reconstruction retains directional smearing and boundary errors.
- Similar acquired-angle agreement can coexist with substantially different volume quality.
- Dominant held-out projection error peaks approximately move with acquisition orientation.
- Peak magnitude and shape also change; this is not a pure angular shift.

Not established:

- Generalization to other patients, anatomies, spans, noise conditions, or scanners.
- Statistical repeatability across seeds.
- Separation of initialization effects from optimization/acquisition effects.
- A validated uncertainty estimator or error predictor.
- A projection-completion or reconstruction-correction method.
- Superiority of evidential learning, diffusion, geometry conditioning, or any proposed new regularizer.

Treat the existing chest/orientation results as development evidence because we have already inspected them repeatedly. Never relabel these exact results as untouched final test cases.

## 10. Literature handoff and evidence status

These are leads reviewed during the discussion, not a systematic review proving an empty research gap. Read current primary papers and supplements; record the exact version inspected. Never infer a limitation solely from a title or an abstract. Distinguish an author-acknowledged limitation, a boundary of the reported experiments, and our hypothesis.

### A. R²-Gaussian — NeurIPS 2024

Ruyi Zha et al., “R²-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction.”

- Paper: https://arxiv.org/abs/2405.20693
- Official code: https://github.com/Ruyi-Zha/r2_gaussian
- Role: current baseline and source of data/implementation.
- Task: check full method, regularization, initialization, density control, projection conventions, and stated limitations.

### B. EPNet — MICCAI 2021

Ce Wang et al., “Improving Generalizability in Limited-Angle CT Reconstruction with Sinogram Extrapolation.”

- Paper: https://arxiv.org/abs/2103.05255
- Code: https://github.com/mars11121/EPNet
- Full-text version inspected in discussion: arXiv v4.
- Uses sinogram extrapolation and dual-domain reconstruction in an unrolled framework.
- Tests AAPM-to-COVID/LIDC transfer.
- The inspected final text says fan-beam geometry. An older review snippet described parallel beam; do not propagate that obsolete characterization.
- Models are trained/tested for corresponding acquisition settings; this differs from proving a single model handles unseen acquisition spans and orientations.
- Ablation: extending extrapolation from 30 to larger numbers of angles reduced cross-dataset performance. Finetuning the extrapolation module also reduced generalization in that comparison.
- Useful lesson: more completion is not automatically better; account for geometry changes and the trustworthiness of inferred measurements.

### C. DOLCE — ICCV 2023

Jiaming Liu et al., “DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction.”

- Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Liu_DOLCE_A_Model-Based_Probabilistic_Diffusion_Framework_for_Limited-Angle_CT_Reconstruction_ICCV_2023_paper.html
- Preprint: https://arxiv.org/abs/2211.12340
- Code: https://github.com/wustl-cig/DOLCE
- Alternates conditional diffusion sampling and data-fidelity updates; reports transfer across image types and coherent volumes from 2D pretrained models.
- Do not call it CVPR 2023; the primary conference page is ICCV 2023.
- Transfer evidence exists. Do not claim it inherently cannot generalize. Inspect exact geometry, conditioning, runtime, and splits before comparison.
- Diffusion plus physical consistency is established prior art, not our novelty.

### D. SPARK — IEEE Transactions on Medical Imaging, 2026

Haowei Zhou et al., “Structurally Informed 3-D Gaussian Splatting for Limited-Angle CBCT.”

- DOI: https://doi.org/10.1109/TMI.2026.3703335
- Abstract record: https://pubmed.ncbi.nlm.nih.gov/42284162/
- Closest competing concept found.
- Authors' abstract: a geometry-conditioned network predicts complete Gaussian parameters from sparse projections, followed by physics-based Gaussian refinement; reported setting is simulated LA-CBCT.
- Only the abstract was accessible in the prior discussion. Exact splits, unseen-orientation evaluations, acquisition ranges, implementation, and code availability remain unverified.
- Obtain the full text through legitimate accessible sources or ask the user to supply it if needed for a final novelty decision. Continue independent infrastructure work if unavailable.
- Do not propose geometry-conditioned Gaussian initialization plus refinement as novel without addressing this paper.

### E. GR-Gaussian — 2025 preprint

“GR-Gaussian: Graph-Based Radiative Gaussian Splatting for Sparse-View CT Reconstruction.”

- Paper: https://arxiv.org/abs/2508.02408
- Latest arXiv version found during discussion: v2, 6 August 2025. Recheck current status.
- Denoised initialization and pixel/graph-aware gradient strategy for Gaussian density control.
- Inspected experiments use 25 views over 360°, including X-3D and real datasets. That evidence is not a test of contiguous missing-angle robustness.
- Describes PSNR-based stopping every 500 iterations. Verify what reference/metric it uses; never use final-test reference volume PSNR for our stopping decisions.
- Graph regularization, improved splitting, and denoised Gaussian initialization are existing ideas.

### F. Directional regularization / CIL — 2023

“A directional regularization method for the limited-angle Helsinki Tomography Challenge using the Core Imaging Library (CIL).”

- DOI: https://doi.org/10.3934/ammc.2023011
- Combines single-sided directional TV, isotropic TV, preprocessing, and tailored bounds.
- Strong challenge results; task-specific assumptions matter.
- Essential classical reference and potential baseline. Do not call directional TV itself new.
- Inspect assumptions before transferring it to medical cone-beam data; challenge-specific bounds and binary-object structure may not apply.

### G. Learning the Invisible — Inverse Problems 2019

Tatiana A. Bubba et al., “Learning the Invisible: A Hybrid Deep Learning–Shearlet Framework for Limited Angle Tomography.”

- Paper: https://arxiv.org/abs/1811.04602
- DOI: https://doi.org/10.1088/1361-6420/ab10ca
- Separates visible and invisible directional information using shearlets; applies learned inference selectively.
- Useful principle: preserve recoverable information and identify where prior assumptions enter.
- Its theory/representation cannot be assumed to transfer unchanged to our 3D cone-beam Gaussian model.

### H. Clinical Metadata Guided Limited-Angle CT Image Reconstruction

Yu Shi et al.; first preprint 2025, inspected revision v2 dated 27 March 2026.

- Paper: https://arxiv.org/abs/2509.01752
- Uses metadata-guided diffusion with data consistency in a 2.5D slab setting.
- Explicit tests include unseen 75° fan-beam and 45° parallel-beam acquisitions.
- The 75° test interpolates between 60° and 90° training spans; authors identify it as a mild shift.
- Requires clinical metadata and learned priors; differs from our intended no-population-training first approach.
- This prevents claiming that unseen-angle or unseen-geometry generalization has never been studied.

### I. Other relevant work

- PFITRE: “Limited-angle x-ray nano-tomography with machine-learning enabled iterative reconstruction engine,” npj Computational Materials 2025. https://doi.org/10.1038/s41524-025-01724-0 . Reports transfer to unseen samples/modalities; explicitly acknowledges possible incorrect features under severe missing information despite data consistency.
- TEMDiff: “Limited-Angle Tomography Reconstruction via Projector Guided 3D Diffusion,” 2025 preprint. https://arxiv.org/abs/2510.06516 . Electron tomography; simulator mismatch and severe misalignment are acknowledged limitations. Do not directly transfer its physics or extremely narrow-angle results to medical CBCT.
- TG-Field: “Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction,” 2026. https://arxiv.org/abs/2602.11705 . Geometry-aware Gaussian deformation, spatial priors, and dynamic reconstruction. Inspect before claiming geometry-aware Gaussian representations are new.
- PFGDM: “Prior Frequency Guided Diffusion Model for Limited Angle (LA)-CBCT Reconstruction,” 2024. https://arxiv.org/abs/2404.01448 . Uses patient-specific prior CT information. Relevant contrasting assumption; our first method should not require a prior scan of the evaluation patient.

For a literature matrix, record: citation/version, geometry, 2D/2.5D/3D, training data, inference inputs, learned prior, initialization, tested spans, tested orientations, anatomy split, scanner shift, stopping, compute, code availability, author-stated limitations, and untested settings. Mark unknown entries as unknown.

## 11. Candidate method: exploratory design requirements

Do not implement a made-up support formula merely to have a geometry-aware loss. First formulate and validate what the geometry statistic measures.

A conceptual objective is:

    projection_fidelity(G; measured_data, geometry)
    + lambda_tv * existing_TV(volume(G))
    + lambda_geo * proposed_geometry_term(G or volume(G); geometry)

This is a design sketch, not a final equation or confirmed improvement.

Initial approach:

1. Retain original measured-projection losses and baseline defaults.
2. Compute a reproducible directional-support descriptor from the acquired geometry.
3. Compare ordinary TV with a simple geometry-oriented directional regularizer.
4. Evaluate whether this improves unseen conditions at fixed settings.
5. Only if supported, explore direct Gaussian covariance/shape or density-control changes as a separate ablation.

Potential implementation entry point: train.py already queries differentiable 3D patches for tv_3d_loss. Inspect query gradients, array/world axes, physical voxel spacing, patch boundaries, and cost before adding a patch-based term.

The descriptor must depend on measured acquisition information, not reference errors, full-circle reference measurements, or a full-circle control reconstruction unavailable at inference.

A global start/span descriptor is a simple approximation, not an exact cone-beam visibility model. Investigate whether a local ray-direction or operator-sensitivity descriptor is needed. Distinguish model sensitivity from true recoverability; low sensitivity is not a calibrated uncertainty measure.

Important failure modes:

- Penalizing weakly constrained directions can suppress artifacts but erase real boundaries.
- Gaussian elongation is not always an artifact; blanket isotropy or aspect-ratio limits can remove legitimate anisotropic structure.
- Reducing the visible training loss does not guarantee better missing-direction or volume accuracy.
- Rotating volume data into a canonical orientation can introduce interpolation artifacts; this requires a controlled resampling baseline.
- Extra smoothing or a larger regularization weight can explain gains without geometry being useful.

Preserve an exact baseline mode: new features disabled must retain the original loss, initialization, and density-control behavior. Do not silently change random-number consumption in disabled mode.

## 12. Benchmark design

Prepare a machine-readable manifest before generating expensive data. Actual volume assignments depend on an inventory of accessible datasets. Do not invent patient IDs or assume that all original dataset volumes are locally present.

Each case should contain at least:

- Unique case_id and independent volume/patient identity.
- Source path and source hash.
- Development/validation/final-test role.
- Geometry identity, physical parameters, geometry hash, units.
- Start angle, acquired span, view count, endpoint convention.
- Explicit training and evaluation angle lists.
- Noise model and seed.
- Method configuration, initialization settings, optimization seed, iteration budget.
- Output directory, expected inputs, status, and provenance.

Suggested initial acquisition grid, not a completed experiment:

- Spans: 90°, 120°, 150°.
- Starts: several values across the circle; reserve distinct starts before tuning.
- View count: 50 for continuity with the pilot.
- Same physical scanner geometry initially.
- Noise-free initially to isolate the proposed change.
- Multiple independent volumes when available.

Stage the grid so full factorial growth does not exhaust GPU time. First use a tiny development subset; expand only after correctness checks. Estimate total runs × observed runtime, disk use, and peak VRAM before launching a batch.

Separate evaluation axes:

1. New orientation, otherwise familiar acquisition conditions.
2. New span, otherwise familiar conditions.
3. New volume at familiar conditions.
4. Combined volume and acquisition shift.

If only a few heterogeneous source volumes are available, describe transfer across those examples, not population-level patient generalization. Multiple slices from one patient are not independent patients. Keep all slices/crops of one patient within one population split.

For the population-training-free method, development data still matter: choosing hyperparameters on a volume makes it development data. Holdout rules remain necessary.

## 13. Projection split and geometry correctness

Do not blindly reuse the pilot's 100-angle grid for every new start/span. A new training grid can overlap it. Generate explicit angle arrays and validate circular separation for every case.

Where cross-case comparison requires a shared evaluation set, construct a common grid disjoint from the union of all relevant training sets. If impossible at a chosen spacing, design another deterministic grid and record it. Do not change test angles separately per method.

Inside/outside classification must handle wraparound intervals, e.g. start 330°, span 120°. Use modular relative angles with a documented numerical tolerance and endpoint convention. Derive sector counts rather than hardcoding 33/67.

For 360° controls, every direction is inside the acquired range, although held-out angles remain unsampled training directions.

Maintain the existing coordinate and detector conventions. Validate reference reprojection and compare the generalized generator with old angle lists before relying on it. Do not introduce an axis transpose based on visual assumptions.

For fixed-count experiments, state that angular spacing varies with span. Later add a separate fixed-spacing experiment if needed. These are different questions, not interchangeable controls.

## 14. Baselines, ablations, and metrics

Minimum practical baselines:

1. Raw FDK as a diagnostic reference, not the only competitor.
2. Original R²-Gaussian with its current default TV.
3. R²-Gaussian with ordinary TV fairly tuned on development data.
4. Proposed geometry-dependent regularization.
5. A classical iterative TV or directional-TV method when feasible in the verified environment.

GR-Gaussian and a learned/diffusion baseline may be added after code availability and adaptation costs are established. Do not compare numbers from unrelated papers as if they were runs on our data.

Key ablations:

- Disable the geometry term.
- Replace geometry-dependent weighting with a uniform counterpart of comparable strength.
- Supply deliberately incorrect orientation to the regularizer while keeping the actual forward operator correct.
- Compare true geometry versus simple global start/span approximation if a richer descriptor is proposed.
- Separate initialization changes from regularization changes.
- Separate regularization changes from densification/splitting changes.

Matched comparisons should use the same measured data, seed, initializer artifact, iteration budget, and metric implementation wherever the method does not intentionally change that component. When an initializer is intentionally changed, label it as an independent factor.

Primary evaluation:

- Whole-volume PSNR with a documented reference range.
- Existing slice-aggregated SSIM, labeled accurately.
- Volume MAE and RMSE.
- Predefined boundary/detail metrics to detect oversmoothing; define masks and thresholds before final evaluation and use the same reference-based masks across methods.
- Runtime, peak VRAM, final Gaussian count, and any added preprocessing cost.

Secondary diagnostics:

- Voxel-volume TIGRE projection error at shared held-out angles.
- Separate Gaussian-renderer projection metrics when useful.
- Acquired-range versus missing-range errors with counts.
- Absolute and acquisition-relative angular curves.
- Fixed-scale slice/error figures without independent image normalization.
- Per-volume and per-geometry results, including regressions and worst cases.

Avoid treating the 100 correlated views from one volume as 100 independent patients. Report repeats and uncertainty at an appropriate level; do not manufacture significance from correlated slices/views.

Do not select checkpoints, stopping rules, regularizer strength, geometry orientation, or Gaussian constraints using final-test reference metrics. Fix the budget or choose a rule on development data. Measured-view validation is useful but does not by itself validate missing-direction fidelity.

Success means a repeatable improvement over fairly tuned baselines on unseen cases, with preserved detail and acceptable computational cost. A gain confined to the old chest pilot does not meet this criterion.

## 15. Engineering requirements

Prefer a small experiment layer over a broad refactor. Proposed layout, adaptable to current repository conventions:

```text
experiments/limited_angle_generalization/
    README.md
    protocol.md
    literature.md
    configs/
    manifests/
    generate_cases.py
    run_cases.py
    evaluate_cases.py
    summarize_results.py
```

These are proposed deliverables, not claims that files already exist. Add only what the staged work requires. Shared pure helpers may be placed elsewhere if that prevents importing GPU libraries for CPU-only tests.

Requirements:

- Unique run IDs and explicit refusal to overwrite existing completed runs.
- Dry-run mode resolving planned cases and commands without GPU work.
- Clear distinction between new run, resumable checkpoint, failed run, and completed run.
- Validate checkpoint identity/configuration before resume.
- Capture subprocess failures even when logging through tee; use pipefail and record pipeline exit codes.
- Log source commit, dirty diff or patch, config, seeds, environment, input hashes, and initialization provenance.
- Avoid large arrays in Git and avoid storing credentials or SSH secrets.
- Python 3.9 compatibility unless a separately justified environment change is required.
- No automatic upgrade of the working CUDA/TIGRE stack.
- On workstation execution, use tmux or the institution's actual scheduler rules; inspect existing GPU processes before claiming resources.
- Never kill unknown processes or overwrite another experiment's logs.

Meaningful correctness checks:

- Angle generation, radians/degrees, wraparound, half-open intervals, and overlap detection.
- Identical reference and evaluation inputs for paired cases.
- Reproduction of existing pilot angle arrays by generalized configuration.
- Descriptor behavior under controlled coordinate rotation.
- Regularizer finite outputs/gradients and physical axis/spacing handling.
- Original behavior when the proposed term is disabled.
- Output overwrite protection and dry-run behavior.

Use tiny synthetic phantoms and small volumes for GPU smoke checks. A smoke test verifies code execution and conventions, not scientific performance. Do not run full 256³ reconstructions merely as unit tests.

## 16. Communication and research integrity

The user prefers clear explanations and practical commands. Explain each change in terms of its research purpose. Distinguish:

- Implemented versus proposed.
- Tested locally versus requiring GPU verification.
- Observed results versus interpretation.
- Known literature evidence versus unverified claims.

Maintain a short continuation note after each meaningful milestone: current branch/commit, completed work, exact commands, output locations, unresolved risks, and next task.

Do not claim that a file is committed, pushed, archived, or a run is complete unless verified. Do not merge or push without session authorization. Local reversible development should continue without repeated permission requests. When a genuine blocker exists, complete independent work first and ask one concrete question with the reason.

If preparing reports, cite primary methods and include the working repository link. Do not omit the original R²-Gaussian credit. Do not describe synthetic-volume performance as demonstrated clinical safety or efficacy.

## 17. First Codex session: concrete deliverable

Complete this initial milestone before launching new full-scale experiments:

1. Audit the current repository, branch, user changes, applicable instructions, environment, data availability, and pilot evidence.
2. Reconcile the stale continuation note with the completed FDK and rotation diagnostics; preserve historical artifacts.
3. Write a concise research protocol defining algorithmic generalization, development/test separation, proposed baselines, metrics, and staged compute budget.
4. Create a literature matrix with source links and evidence status. Resolve accessible details, especially SPARK, EPNet, and directional TV. Explicitly record unavailable full text rather than inventing it.
5. Prepare a small configurable acquisition manifest and a dry-run/validation path that generalizes the existing fixed-angle assumptions, preserves units/conventions, and detects overlap and wraparound errors.
6. Run meaningful CPU-side checks, including reproducing the three historical angle lists and inside/outside counts.
7. Inspect the TV/query/density-control implementation and write a precise candidate regularizer design with assumptions, dimensional units, expected cost, and a falsification test. Do not guess that stronger smoothing in any particular direction is correct without analysis.
8. If the geometry design is justified and execution is available, prepare an opt-in minimal prototype and tiny correctness smoke test. Keep the feature disabled by default. If the design remains unresolved, complete the benchmark milestone and state exactly what must be resolved before a scientific implementation.
9. Report changed files, tests, outstanding literature/geometry issues, estimated next-run budget, and the exact next university commands.

Do not launch a full factorial benchmark in this session. Do not treat the user as having chosen an unverified method simply because this handoff lists a candidate. The immediate goal is a trustworthy, reviewable foundation for the next reconstruction experiment.

## 18. Suggested opening instruction from the user

“Read TRDP2_CODEX_HANDOFF.md completely and inspect this repository. Follow the bounded first-session deliverable in Section 17. Preserve the completed pilot, verify the current state, and implement the benchmark preparation and checks before any full GPU experiments. Our primary objective is reconstruction generalization across missing-angle configurations. Treat the geometry-dependent regularizer as a research hypothesis, check relevant prior work, and explain what you changed and validated.”
