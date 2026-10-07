"""Guarded single-case workflow. Default is read-only; execution requires --execute."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from benchmark import REPO, local_path, provenance, require, sha256, validate_manifest, write_new_json
from datasets import generate_dataset, load_array, verify_dataset


def set_state(directory, record):
    temporary = directory / "state.next.json"
    write_new_json(temporary, record)
    temporary.replace(directory / "state.json")


def environment():
    versions = {}
    for name in ("numpy", "torch", "open3d", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def execute_job(directory, command, context, postcheck, root=REPO):
    directory.mkdir(parents=True, exist_ok=False)
    write_new_json(directory / "context.json", context)
    record = {"status": "running", "command": command, "returncode": None,
              "started_unix_seconds": time.time(), "peak_vram_bytes": None,
              "peak_vram_note": "not measured by this runner"}
    set_state(directory, record)
    started = time.perf_counter()
    try:
        with (directory / "process.log").open("x") as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, MPLBACKEND="Agg", PYTHONFAULTHANDLER="1"))
        record["returncode"] = completed.returncode
        require(completed.returncode == 0, "subprocess failed; inspect " + str(directory / "process.log"))
        record["artifacts"] = postcheck()
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = str(error)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        set_state(directory, record)
    return record


def job_directory(case, stage, root=REPO):
    return local_path(root, "output/limited_angle_generalization_jobs/" + case["run_id"] + "__" + stage)


def completed_stage(case, stage, manifest, root=REPO):
    directory = job_directory(case, stage, root)
    state = json.loads((directory / "state.json").read_text())
    context = json.loads((directory / "context.json").read_text())
    require(state["status"] == "completed" and state["returncode"] == 0, "previous stage not complete")
    require(context["manifest"]["manifest_sha256"] == manifest["manifest_sha256"] and
            context["case_id"] == case["case_id"], "previous stage identity mismatch")
    return state


def initializer_check(manifest, case, root=REPO):
    import numpy as np
    path = local_path(root, case["initializer"])
    array = np.load(path, allow_pickle=False)
    require(array.shape == (case["method"]["n_points"], 4) and bool(np.isfinite(array).all()), "invalid initializer")
    require(bool((array[:, 3] > 0).all()), "nonpositive initializer densities")
    return {"initializer_sha256": sha256(path)}


def final_check(manifest, case, root=REPO):
    directory = local_path(root, case["output_directory"])
    final = directory / "point_cloud/iteration_30000"
    shape = manifest["volumes"][case["volume_id"]]["geometry"]["nVoxel"]
    load_array(final / "vol_pred.npy", shape)
    load_array(final / "vol_gt.npy", shape)
    require(sha256(final / "vol_gt.npy") == case["source_sha256"], "training reference changed")
    files = [final / "vol_pred.npy", final / "vol_gt.npy", final / "point_cloud.pickle",
             directory / "eval/iter_030000/eval3d.yml", directory / "ckpt/chkpnt30000.pth"]
    return {p.relative_to(directory).as_posix(): sha256(p) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--stage", choices=("generate", "verify", "initialize", "train", "evaluate"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    cases = [c for c in manifest["cases"] if c["case_id"] == args.case]
    require(len(cases) == 1, "select exactly one known case")
    case = cases[0]
    if args.execute:
        require(case["population_role"] == "development" and case["acquisition_role"] == "development",
                "execution is currently limited to development cases; reserved cases stay sealed")
    if args.worker:
        require(args.execute and args.stage == "generate", "invalid worker request")
        generate_dataset(manifest, case)
        return
    if args.stage == "verify":
        print(json.dumps(verify_dataset(manifest, case), indent=2))
        return
    directory = job_directory(case, args.stage)
    require(not directory.exists(), "existing stage directory; no automatic retry/resume: " + str(directory))
    manifest_path = str(args.manifest.resolve())
    if args.stage == "generate":
        require(not local_path(REPO, case["data_directory"]).exists(), "refusing existing dataset")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--manifest", manifest_path,
                   "--case", args.case, "--stage", "generate", "--execute", "--worker"]
        postcheck = lambda: verify_dataset(manifest, case)
    elif args.stage in ("initialize", "train"):
        command = [sys.executable, "-u"] + case["planned_commands"][args.stage][1:]
        postcheck = (lambda: initializer_check(manifest, case)) if args.stage == "initialize" else (lambda: final_check(manifest, case))
    else:
        command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate_cases.py")),
                   "--manifest", manifest_path, "--case", args.case,
                   "--report", str(directory / "metrics.json")]
        postcheck = lambda: {"metrics_sha256": sha256(directory / "metrics.json")}
    print(json.dumps({"stage": args.stage, "case": args.case, "command": command,
                      "job_directory": str(directory), "execute": args.execute}, indent=2), flush=True)
    if not args.execute:
        print("Dry run: no GPU work or files created; prerequisites checked on execution.")
        return
    require(case["population_role"] == "development" and case["acquisition_role"] == "development",
            "execution is currently limited to development cases; reserved cases stay sealed")
    checks = {}
    if args.stage != "generate":
        checks.update(verify_dataset(manifest, case))
    if args.stage == "initialize":
        require(not local_path(REPO, case["initializer"]).exists(), "initializer exists")
    if args.stage == "train":
        init_state = completed_stage(case, "initialize", manifest)
        checks.update(initializer_check(manifest, case))
        require(checks["initializer_sha256"] == init_state["artifacts"]["initializer_sha256"], "initializer changed")
        # Reserve output before native train.py (whose logger permits existing directories).
        local_path(REPO, case["output_directory"]).mkdir(parents=True, exist_ok=False)
    if args.stage == "evaluate":
        state = completed_stage(case, "train", manifest)
        require(state["artifacts"] == final_check(manifest, case), "completed reconstruction changed")
    context = {"manifest": manifest, "case_id": args.case, "stage": args.stage,
               "environment": environment(), "execution_provenance": provenance(REPO), "input_checks": checks}
    result = execute_job(directory, command, context, postcheck)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
