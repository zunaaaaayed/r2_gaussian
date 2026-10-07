# Independent-volume acquisition plan — 7 October 2026

No new raw volumes have been downloaded or assigned to splits. Local data still
represent one chest reference and its derived acquisitions. The pilot's patient
identity is unknown; distinct hashes cannot establish patient independence.

## Public-data shortlist and access evidence

1. **LIDC-IDRI**: preferred candidate for same-anatomy transfer with subject and
   series identifiers. TCIA lists 1,010 subjects, DICOM images, approximately
   133.15 GB for the complete image collection, CC BY 3.0, and Data Retriever
   access. Use a small selected subset, not the full collection. The project's
   synthetic-data documentation already names LIDC-IDRI as a source, so overlap
   with the anonymous pilot must be resolved before claiming independence.
   Source: https://www.cancerimagingarchive.net/collection/lidc-idri/
2. **Pancreas-CT**: candidate for a separate anatomy-transfer experiment, not an
   interchangeable chest test. TCIA lists 82 subjects and CC BY 3.0; its legacy
   access page lists 9.3 GB of DICOM images and NBIA Data Retriever access.
   Sources: https://www.cancerimagingarchive.net/browse-data/
   and https://wiki.cancerimagingarchive.net/display/public/pancreas-ct

These facts were checked through indexed official TCIA collection/policy pages.
Direct current collection-page fetches timed out; no download endpoint was tested.
Confirm the current per-collection access page when acquiring a subset. Retain
collection DOI/citation, license URL, version/date, and applicable TCIA attribution.
TCIA's policy requests collection and repository citations; do not redistribute a
mirror without checking its separate downstream requirements.
Policy: https://wiki.cancerimagingarchive.net/display/Public/Data%2BUsage%2BPolicies%2Band%2BRestrictions

## Proposed intake, before reconstruction

- Obtain a small subject-indexed subset (initial target: six chest subjects),
  one eligible CT series per subject. Record collection subject ID, study and
  series UID, source URL/version, raw checksum, spacing, orientation, dimensions,
  and preprocessing version. Keep repeated scans/crops in one subject group.
- Trace the supplied chest reference to its original subject if possible. If
  this cannot be resolved, label independence from the pilot unverified. Do not
  count transformed or resampled copies as new subjects. A separate documented
  cohort is preferable for a later definitive population test.
- Before examining reconstruction outcomes, freeze an auditable subject split:
  two intake/development subjects, two validation, two final-test candidates.
  Six subjects support a small transfer study, not a population accuracy claim.
  IDs remain unassigned until real metadata is available; do not invent them.
- Audit DICOM orientation, physical spacing, slice order and rescale parameters.
  Predeclare attenuation conversion, crop/resampling and intensity calibration;
  per-volume min/max normalization would change the meaning of fixed TV strength.
  Keep raw data immutable and record every preprocessing transform.
- Start with the familiar 120° acquisition; evaluate combined volume/acquisition
  shift only after freezing the chosen development rule. Keep final outcomes
  sealed until fixed-final evaluation. No validation volume may tune TV.

Current next data step is acquiring metadata/series selection and resolving source
identity, then implementing and testing the intake conversion. Dataset research
alone is not evidence that any independent-volume benchmark has been run.
