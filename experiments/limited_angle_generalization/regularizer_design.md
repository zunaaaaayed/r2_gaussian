# Candidate design review — unresolved, no regularizer enabled

Inspected baseline commit `327a731834ad3e4f6b89a1e993df01cd3fa1bb6d`.
The implementation entry point exists, but geometry-to-penalty direction is not
yet justified. This milestone changes no training, loss, query or density-control
code. Geometry-dependent regularization remains a research hypothesis.

## Verified implementation

| Source | Finding and consequence |
|---|---|
| `train.py:85–146` | For positive lambda_tv, sample one patch center with CPU `torch.rand(3)`, query 32³ voxels with physical extent `dVoxel * 32`, add TV to L1 + weighted DSSIM. Reuse this patch for any added penalty; an extra random draw changes the baseline trajectory. |
| `r2_gaussian/utils/loss_utils.py:19` | TV is the sum of absolute forward differences in array axes 0/1/2, divided by the total number of edges for mean reduction. No voxel-spacing division. This is axiswise anisotropic L1 TV, despite generic descriptions of ordinary TV. |
| `r2_gaussian/gaussian/render_query.py:27` | Query passes x/y/z extents, centers and voxel counts directly into the CUDA voxelizer. It returns a differentiable volume from Gaussian positions/density/scales/rotations. Center/extents become Python numbers, not differentiable inputs. |
| `cuda_voxelizer/forward.cu:90,145,204–206` | Spacing is extent/count; Gaussian world position maps to `(position-center+extent/2)/spacing`; flat index is `nz*ny*x+nz*y+z`. The kernel evaluates at voxel index **+0.5**. The smoke's off-center mean is chosen at an exact voxel center. |
| `r2_gaussian/dataset/dataset_readers.py:65–80,143,156` | Scene lengths and projection intensities scale by `2/max(sVoxel)`. Source center at angle theta is `(DSO*cos(theta), DSO*sin(theta),0)`. Metadata angles are radians. |
| `r2_gaussian/utils/ct_utils.py:29` and pilot `generate.py:28` | TIGRE sees zyx voxel coordinates and reversed detector rows; detector dimensions remain `[v,u]`. Do not change these based on image appearance. |
| `train.py:153–191`, `gaussian_model.py:439–583` | Clone/split decisions use accumulated visible screen-position gradient norms and a scale threshold; pruning also uses density and bounding box. TV influences learned parameters and future projections, hence density control indirectly. Keep its policy fixed for a regularization ablation. |
| `initialize_pcd.py:25,75`, `general_utils.safe_state` | Initial sampling uses NumPy seed zero; training resets Python, NumPy and Torch seeds to zero. Candidate-disabled execution must consume exactly the old random draws. |

The patch's sampled center stays inside the reconstruction box if each patch
extent fits. Generalized anisotropic or very small volumes need an explicit
extent check before sampling. The baseline TV acts only on internal patch edges;
it does not impose periodic boundaries or zero-padding differences. A 32³ cube
contains 95,232 axis edges. Physical spacing is 2/256 = 0.0078125 supplied length
units in the current chest, and scene_scale is 1. Noncubic spacing changes the
meaning of equal raw-difference penalties.

There is an existing convention to investigate: initializer point placement uses
`sampled_indices * dVoxel - sVoxel/2 + offOrigin` without +0.5, while the query
kernel samples at voxel centers with +0.5. This is a potential half-voxel offset,
not proof that either complete pipeline should be changed. Preserve the baseline;
test it independently with known point/voxel phantoms before changing placement.

## What a descriptor could measure

A ray-direction second moment `C(x) = mean_i(r_i(x) r_i(x)^T)` is dimensionless
and transforms as `R C R^T` under a joint coordinate rotation. Those properties
alone do not make it a recoverability or Fourier-support tensor. For a full
circular orbit at its center, every central ray has zero z component, so C has
a zero z eigenvalue even when useful axial structure is measured by off-center
cone rays. Mapping its smallest eigenvector directly to stronger smoothing is
therefore unjustified. Detector coverage, spatial position, spatial frequency,
redundancy and the null space of the full operator all matter.

Investigate operator response on fixed, normalized, localized sinusoidal test
functions `phi_(x,q,k)` (center x, wave-normal direction q, spatial frequency k).
For the measured-geometry linear projector A, record
`s = ||A phi||_2² / ||phi||_2²`, with declared voxel/detector quadrature and
identical support/amplitude across directions. Its numerical units depend on
projector length scaling and quadrature; normalize by a geometry-local median
only for relative comparisons, retaining the unnormalized values in records.
This is perturbation sensitivity, not uncertainty or an identifiability proof:
correlated responses can still conceal near-null combinations. Validate against
small-system singular vectors or pairs of indistinguishable perturbations.
Use only acquisition geometry, never reference errors or held-out measurements,
to construct a future inference descriptor.

## Conditional patch penalty, not an implementation specification

If the descriptor evidence supports a direction rule, a testable family is
`R_M(v) = mean_x(sqrt(g(x)^T M(x) g(x) + epsilon²) - epsilon)` on common interior
patch voxels. Here `g_j = (v[x+e_j]-v[x])/h_j`, in attenuation/length; M is a
dimensionless positive-definite tensor with fixed trace 3 and bounded eigenvalue
ratio. Epsilon has the same units as g. R_M has attenuation/length units; its
coefficient must account for scene length scaling and projection fidelity units.
Equivalently multiply by one declared fixed reference length, never a per-case
intensity-dependent rescaling. This functional is an isotropic tensor-gradient
family and is **not** equal to existing L1 TV when M is identity: its uniform
counterpart must be compared explicitly, while the original TV remains untouched.

The mapping from sensitivity to M, sign of anisotropy, epsilon and coefficient
are unresolved. Both extra suppression in low-sensitivity directions and
preservation of their boundaries are plausible; neither follows automatically
from a ray moment. Do not encode either choice until the controlled tests below.
A lambda_geo=0 branch must bypass descriptor evaluation, patch sampling and all
extra loss computation, preserving baseline arithmetic and RNG state exactly.

Reuse the existing query/autograd graph. Added gradient/tensor operations scale
as O(32³) for a constant tensor (about 0.375 MiB for three float32 gradient
channels; autograd adds memory). A dense 3×3 tensor per voxel costs 1.125 MiB,
before gradients. Computing 50 local ray moments across the patch costs
O(50*32³) vectors per iteration; cache geometry if ever justified. Operator-probe
calibration is more costly: e.g. 8 centers × 12 directions × 3 frequencies = 288
forward projections, so start on tiny volumes and measure preprocessing time.
No runtime or VRAM benefit is currently measured.

## Falsification and release gates

1. On tiny anisotropic grids, verify query indexing with off-center Gaussians,
   forward-projection conventions, ramps with known physical derivatives,
   constant-zero loss, finite nonzero gradients and finite-difference gradient
   agreement away from absolute-value kinks/culling boundaries. `smoke_gpu.py`
   checks baseline backend shape/position/finite gradients, not all these gates.
2. Jointly rotate coordinates of object, detector and source: the descriptor
   must obey its tensor transformation and the penalty must agree within
   controlled discretization tolerance. Use 90° axis permutations first to avoid
   interpolation. Arbitrary resampling needs its own control. Merely changing
   start angle around a fixed object is a different experiment.
3. Across several positions/frequencies, compare descriptor rankings with the
   actual operator responses and near-null perturbations. Reject a global rule
   whose ranking reverses or misses important local behavior. Passing this test
   still does not determine the best regularizer.
4. On development phantoms with thin boundaries at multiple normals, compare
   both anisotropy signs with tuned ordinary TV and equal-strength uniform R_M.
   Reject if gains are fully explained by uniform smoothing, wrong orientation
   works equally well, or boundary/gradient errors worsen despite higher PSNR.
5. Freeze the surviving rule and strength, then test reserved acquisitions and
   independent volumes. Failure there falsifies the generalization claim even
   if the inspected chest improves. Verify identical disabled-mode outputs and
   RNG states before any implementation enters the baseline runner.

Literature constraint: directional TV already exists; SPARK already describes
geometry-conditioned Gaussian initialization/refinement. See `literature.md`.
The current unresolved mapping and missing independent volumes are reasons to
finish infrastructure first, not evidence of novelty or of method failure.
