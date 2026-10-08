# Frozen TV reference and independent-volume preparation — 8 October 2026

Original method: [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian).
Research repository: [TRDP2](https://github.com/zunaaaaayed/trdp2).

## Completed TV selection

All four additional controls reached 30,000 iterations and completed evaluation.
The accepted start90/TV0.1 result is the fresh `__chani_20261008` attempt; the
unfinished cheonech attempt must not enter aggregation. Dataset, initializer,
final-volume/checkpoint and metric hashes were verified in the completion review.

| TV | PSNR start0 | PSNR start90 | Mean PSNR | Eligible |
|---|---:|---:|---:|---|
| 0.025 | 30.766012 | 28.537963 | 29.651987 | No: SSIM declines |
| **0.05** | **30.912636** | **28.591247** | **29.751942** | **Yes** |
| 0.1 | 30.782231 | 28.448343 | 29.615287 | No: SSIM and boundary MAE decline |

`tv_reference_frozen.json` preserves all six metric records, hashes, and the
selection-protocol hash. **0.05 is frozen as the ordinary-TV reference**, with seed
zero, 30,000 iterations and no geometry penalty. This selection applies to the
existing supplied intensity/geometry convention. It is not evidence of calibrated
cross-volume transfer. Any future preprocessing change is an explicit factor;
do not silently retune the weight on validation or final-test volumes.

## Candidate subjects, assigned before reconstruction

`cohort_v1.json` binds official NBIA metadata snapshots, series identifiers and
license/collection DOI for six distinct LIDC-IDRI subject IDs:

| Role | Subject IDs |
|---|---|
| Intake/development | LIDC-IDRI-0002, LIDC-IDRI-0003 |
| Validation | LIDC-IDRI-0004, LIDC-IDRI-0005 |
| Final test candidates | LIDC-IDRI-0006, LIDC-IDRI-0007 |

This is a convenience cohort selected by ascending ID, not representative sampling.
Only one CT series was returned per subject. All scans/crops of a subject must stay
in that subject's role. Eligibility remains conditional on DICOM geometry and
source identity checks. Freeze exclusions and replacements before looking at
reconstruction metrics; record failures rather than silently replacing subjects.

A newly inspected provenance lead in `data_generator/synthetic_dataset/raw_metadata.py`
maps `0_chest` to **LIDC-IDRI-0001**. That subject is excluded. This mapping does not
prove the actual supplied `vol_gt.npy` came from that scan: independent-patient
claims relative to the pilot remain unverified until source reproduction or
external provenance evidence establishes the mapping.

## Actual intake evidence

Official metadata for all six subjects was retrieved on 8 October 2026. Only the
first development volume's raw ZIP has been downloaded:
`data/trdp2/independent_volume_raw_v1/LIDC-IDRI-0002/series.zip` (67,386,877 bytes).
Its `download.json` records URL and SHA-256. ZIP CRC verification and DICOM header
audit pass: 261 slices, 512×512 pixels, spacing 0.681641×0.681641×1.25 mm,
consistent direction cosines/positions, unique SOP identifiers and matching
subject/study/series IDs. The included license is preserved in `LIDC-IDRI_LICENSE.txt`.
See `lidc_0002_header_audit.json` for the full record.

The audit reads headers without decoding pixels. It is not a validated processed
reference volume. The other five raw series are not downloaded; no new projection
generation, initialization, reconstruction or final-test metrics have run.

## Preprocessing gate and next work

The inherited DICOM conversion sorts filenames, reverses slices, clips HU, then
uses each volume's min/max and resizes directly to a cube. That recipe is not yet
a physically calibrated cross-volume benchmark. Before GPU experiments:

1. Verify the pilot identity against subject 0001 source/preprocessing evidence.
2. On development subjects only, decode pixels and validate rescale and padding;
   order slices by physical position, not filename. Preserve orientation/spacing.
3. Freeze an explicit intensity mapping (candidate: fixed clipped HU [-1000,2000]
   mapped by `(HU+1000)/3000`, a normalized image surrogate rather than calibrated
   attenuation) and physical resampling/crop/FOV policy. Validate their consequences
   for scanner units, projection scale and fixed TV strength with phantoms. Do not
   conflate a preprocessing shift with regularizer transfer.
4. Audit and preprocess the remaining subjects under that same frozen policy.
   Keep validation/final reconstruction outcomes sealed, and report this as a small
   transfer study rather than population accuracy. Start with familiar geometry.

CPU-only audit command, after research-environment activation:

```bash
python -m unittest discover -s experiments/independent_volume_evaluation -p 'test_*.py' -v
python experiments/independent_volume_evaluation/audit_dicom.py \
  --cohort experiments/independent_volume_evaluation/cohort_v1.json \
  --subject LIDC-IDRI-0002 \
  --zip data/trdp2/independent_volume_raw_v1/LIDC-IDRI-0002/series.zip \
  --report output/lidc_0002_header_audit_repeat_v1.json
```

Reports refuse overwrite. No GPU sweep is queued by this preparation.

Sources: [TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)
(CC BY 3.0; collection DOI retained per subject), and
[official NBIA API guide](https://wiki.cancerimagingarchive.net/display/Public/NBIA%2BSearch%2BREST%2BAPI%2BGuide).
Use the collection's required attribution and citations for any publication.
