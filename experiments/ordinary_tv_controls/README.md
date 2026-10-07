# Ordinary-TV development controls — 7 October 2026

Original method: R²-Gaussian (https://github.com/Ruyi-Zha/r2_gaussian).
Working repository: https://github.com/zunaaaaayed/r2_gaussian.

Run the predeclared TV grid {0.025, 0.05, 0.1} at development starts 0° and 90°,
both span 120°, 50 views, 30,000 iterations and seed zero. Reuse completed 0.05
runs. The four new runs change only the native --lambda_tv argument and output
path. Reuse the exact dataset and FDK initializer bytes within each acquisition;
do not rerun initialization. No geometry term or reserved acquisition is used.
Allow about four hours for four serial runs, depending on contention.

Selection is fixed before new control results: a non-default weight is eligible
only if neither development orientation has worse boundary MAE or SSIM than its
0.05 baseline. Among eligible weights and the default, select highest mean final
volume PSNR across the two orientations; ties within 0.01 dB retain 0.05 if tied,
otherwise choose the smaller weight. Report every result and each regression.
This conservative detail-preservation rule is a development heuristic, not a
statistical test. Freeze the selected weight before reserved acquisitions or
independent-volume evaluation; never choose separate weights per test volume.

run.py defaults to a read-only preflight. --execute runs one new control, checks
final artifacts, and computes unclipped volume and boundary metrics. Native final
SSIM is retained; independently computed PSNR must agree within 0.001 dB. Runtime
includes postchecks; peak VRAM is not measured. A failed run is retained, not
resumed. Outputs live under output/ordinary_tv_controls and job records under
output/ordinary_tv_control_jobs.

The archival baseline manifest is structurally revalidated against real source
data. Every source file frozen in v4 and the submodule identities must remain
unchanged. Additional control files are allowed, with current provenance and this
script/protocol hashes recorded per run. Baseline source code and historical
manifests are not edited or regenerated. Adding tracked Python files may make the
old runner's exact-file-set validation reject v4; this control runner explicitly
supports additive files without allowing changes to any frozen baseline file.

Commands (research environment):

```bash
python experiments/ordinary_tv_controls/run.py --case chest_start0_span120 --weight 0.025
python experiments/ordinary_tv_controls/run.py --case chest_start0_span120 --weight 0.025 --execute
```

The queued batch uses the same call for both weights and both orientations,
waiting for an idle GPU before each run. It stops at the first failure. No claims
about independent-volume transfer follow from this chest-only tuning experiment.
