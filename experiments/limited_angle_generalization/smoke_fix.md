# Smoke-test segmentation fault: diagnosis and fix

Reproduced the reported crash on cheonech with Python fault tracing. The original
combined-process smoke reached `float(loss)` after both TIGRE projections, then
exited with segmentation fault (139). The query/gradient checks themselves had
already completed.

The installed TIGRE Python wrapper defaults to Siddon projection. The local
TIGRE-v2.3 source `Common/CUDA/Siddon_projection.cu:582` still calls
`cudaDeviceReset()`. The earlier preserved patch only removes the reset from
`voxel_backprojection.cu`. Thus the smoke kept live PyTorch CUDA tensors across
a projector call that resets their context. This was a smoke-test integration bug;
it does not establish a problem in the completed pilot.

The corrected script launches sequential, separate query and projector workers.
Each worker must exit successfully before its JSON is accepted. The parent holds
no GPU tensors and writes the final report only after both workers succeed.
Fault tracing and flushed stage messages make native failures diagnosable.
Merely extracting the loss earlier would leave tensor destruction/context hazards;
process isolation avoids sharing the context altogether.

Actual GPU verification passed: query shape 16×20×24, peak (10,8,13), finite
parameter gradients, TV 0.003996930085122585; wrapped/unwrapped projection maximum
error 1.7136335372924805e-6 against tolerance 1e-5. Total wall time including worker
imports was about 3.8 seconds. Saved report: `manifests/backend_smoke_isolated.json`.
No full reconstruction, dependency rebuild, TIGRE modification or pilot edit was
performed for this fix.

Two CPU regression tests check separate worker execution and failure propagation,
including rejection of a partial worker report followed by a simulated SIGSEGV.
The suite now contains 23 tests. The current source snapshot is
`manifests/development_v3.json`; v1/v2 remain superseded, never-executed preparation
snapshots. Original check logs and audit statements describe the earlier session.

Rerun in the activated research environment:

```bash
python experiments/limited_angle_generalization/prepare.py \
  --manifest experiments/limited_angle_generalization/manifests/development_v3.json --dry-run
python experiments/limited_angle_generalization/smoke_gpu.py --execute \
  --report output/generalization_backend_smoke_20261005_v2.json
```

Keep this process boundary in future generation/evaluation tools if TIGRE Siddon
projection is used alongside PyTorch CUDA work. The FDK context patch does not
make every TIGRE operation safe in a shared PyTorch context.
