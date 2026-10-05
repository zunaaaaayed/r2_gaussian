"""Opt-in, tiny baseline backend check. No optimization or scientific evaluation."""
import argparse
import faulthandler
import subprocess
import tempfile
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from benchmark import REPO, require, write_new_json


def query_check():
    # Heavy dependencies deliberately stay out of the CPU preparation path.
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    from r2_gaussian.gaussian import GaussianModel, query
    from r2_gaussian.utils.loss_utils import tv_3d_loss

    require(torch.cuda.is_available(), "CUDA unavailable; do not modify the installed stack")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = GaussianModel()
    # CUDA voxel samples use index + 0.5, unlike initializer point placement.
    model._xyz = torch.nn.Parameter(torch.tensor([[0.3125, -0.15, 0.1875]], device="cuda"))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[0.2, 0.3, 0.25]], device="cuda")))
    model._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device="cuda"))
    model._density = torch.nn.Parameter(model.density_inverse_activation(torch.tensor([[0.5]], device="cuda")))
    volume = query(model, [0, 0, 0], [16, 20, 24], [2., 2., 3.],
                   SimpleNamespace(compute_cov3D_python=False, debug=False))["vol"]
    require(tuple(volume.shape) == (16, 20, 24), "query axis/shape mismatch")
    require(bool(torch.isfinite(volume).all()), "nonfinite queried volume")
    peak = tuple(int(x) for x in np.unravel_index(int(volume.argmax()), volume.shape))
    require(peak == (10, 8, 13), "query position/spacing convention differs: " + str(peak))
    loss = tv_3d_loss(volume, "mean")
    loss.backward()
    for name in ("_xyz", "_scaling", "_rotation", "_density"):
        grad = getattr(model, name).grad
        require(grad is not None and bool(torch.isfinite(grad).all()), "invalid gradient " + name)
    require(float(model._density.grad.abs().sum()) > 0, "density gradient vanished")

    torch.cuda.synchronize()
    return {"query_shape": list(volume.shape), "query_peak_index": list(peak),
            "tv_loss": float(loss), "finite_parameter_gradients": True,
            "query_elapsed_seconds": time.perf_counter() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "memory_note": "PyTorch allocator only; excludes TIGRE allocations",
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}


def projector_check():
    import numpy as np
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments/limited_angle_120"))
    from generate import project
    started = time.perf_counter()
    scanner = dict(mode="cone", DSD=7., DSO=5., nDetector=[24, 32], sDetector=[4., 4.],
                   nVoxel=[16, 20, 24], sVoxel=[2., 2., 3.], offOrigin=[0, 0, 0],
                   offDetector=[0, 0], accuracy=0.5, filter=None, noise=False,
                   startAngle=330., totalAngle=120.)
    phantom = np.zeros((16, 20, 24), dtype=np.float32)
    phantom[7:11, 5:12, 9:16] = 0.5
    angles = np.deg2rad(np.array([330., 370., 410.]))
    a = project(phantom, scanner, angles)
    b = project(phantom, scanner, angles % (2 * np.pi))
    require(a.shape == (3, 24, 32), "projection shape mismatch")
    require(float(np.max(a)) > 0, "projection is empty")
    error = float(np.max(np.abs(a - b)))
    require(error <= 1e-5, "wrapped/unwrapped projection mismatch")
    return {"wrapped_projection_max_error": error, "projection_tolerance": 1e-5,
            "projector_elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="opt in to isolated tiny CUDA checks")
    parser.add_argument("--report", type=Path, required=True, help="new JSON report; never overwritten")
    parser.add_argument("--worker", choices=("query", "projector"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(not args.report.exists(), "report already exists")
    require(args.report.parent.is_dir(), "report parent must exist")
    if not args.execute:
        require(args.worker is None, "workers require --execute")
        print("Dry run: isolated Gaussian query and TIGRE projector workers; no GPU imports or writes.")
        return
    faulthandler.enable()
    if args.worker:
        result = query_check() if args.worker == "query" else projector_check()
        write_new_json(args.report, result)
        return

    # TIGRE 2.3 Siddon projection resets the CUDA context. Never share its
    # process with live PyTorch CUDA tensors, including during destruction.
    report = {"kind": "tiny_baseline_backend_smoke", "scientific_performance_evidence": False,
              "backend_process_isolation": True}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="trdp2-smoke-") as directory:
        for stage in ("query", "projector"):
            output = Path(directory) / (stage + ".json")
            print("Starting isolated " + stage + " check...", flush=True)
            command = [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
                       "--execute", "--worker", stage, "--report", str(output)]
            completed = subprocess.run(command)
            if completed.returncode != 0:
                raise RuntimeError("{} worker failed with return code {}; no success report written".format(
                    stage, completed.returncode))
            report.update(json.loads(output.read_text()))
            print(stage + " check passed", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    write_new_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
