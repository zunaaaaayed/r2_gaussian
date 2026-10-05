"""CPU-only acquisition planning. No training, CUDA imports, or dataset writes."""

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
TOL_DEG = 1e-6
PHYSICAL_KEYS = (
    "mode", "DSD", "DSO", "nDetector", "sDetector", "nVoxel", "sVoxel",
    "offOrigin", "offDetector", "accuracy", "filter",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def train_angles(start, span, count):
    require(finite(start), "start must be finite degrees")
    require(finite(span) and 0 < span <= 360, "span must be in (0, 360] degrees")
    require(type(count) is int and count > 0, "view count must be a positive integer")
    # Unwrapped degrees deliberately retain the historical generator convention.
    return [start + index * span / count for index in range(count)]


def circular_distance(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def min_separation(a, b):
    require(bool(a) and bool(b), "angle lists must be nonempty")
    return min(circular_distance(x, y) for x in a for y in b)


def inside(angle, start, span, tolerance=TOL_DEG):
    train_angles(start, span, 1)
    require(finite(angle), "angle must be finite")
    if span == 360:
        return True
    relative = (angle - start) % 360.0
    # Values within tolerance of start belong to start; near end belong to end.
    if relative <= tolerance or 360 - relative <= tolerance:
        relative = 0.0
    if abs(relative - span) <= tolerance:
        relative = span
    return relative < span


def validate_angles(values, label):
    require(bool(values) and all(finite(x) for x in values), label + " must be finite and nonempty")
    for index, angle in enumerate(values):
        require(all(circular_distance(angle, other) > TOL_DEG for other in values[:index]),
                label + " contains duplicate circular directions")


def common_evaluation(training_sets, count):
    require(type(count) is int and count > 0, "evaluation count must be a positive integer")
    training = [angle for angles in training_sets for angle in angles]
    # Prefer the pilot midpoint grid, then deterministic offsets of the same grid.
    for fraction in [0.5, 0.25, 0.75] + [index / 64 for index in range(1, 64)]:
        values = [(index + fraction) * 360.0 / count for index in range(count)]
        if min_separation(training, values) > TOL_DEG:
            return values, fraction
    raise ValueError("No disjoint common grid among candidate offsets; change the acquisition design")


def local_path(root, value):
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "paths must be repository relative")
    result = (root / path).resolve()
    require(root.resolve() in result.parents, "path escapes repository")
    return result


def source_record(root, entry):
    case_dir = local_path(root, entry["source"])
    metadata_path = case_dir / "meta_data.json"
    meta = json.loads(metadata_path.read_text())
    scanner = meta["scanner"]
    require(scanner["mode"] == "cone", "only verified cone geometry is supported")
    geometry = {key: scanner[key] for key in PHYSICAL_KEYS}
    for key, size in (("nVoxel", 3), ("sVoxel", 3), ("nDetector", 2), ("sDetector", 2)):
        require(len(geometry[key]) == size and all(finite(x) and x > 0 for x in geometry[key]),
                "invalid geometry " + key)
    for key in ("nVoxel", "nDetector"):
        require(all(type(x) is int for x in geometry[key]), key + " must be integer")
    require(finite(scanner["DSO"]) and finite(scanner["DSD"]) and
            scanner["DSD"] > scanner["DSO"] > 0, "invalid source distances")
    for key, size in (("offOrigin", 3), ("offDetector", 2)):
        require(len(scanner[key]) == size and all(finite(x) for x in scanner[key]), "invalid " + key)
    require(finite(scanner["accuracy"]) and scanner["accuracy"] > 0, "invalid projector accuracy")
    for spacing, extent, counts in (("dVoxel", "sVoxel", "nVoxel"),
                                    ("dDetector", "sDetector", "nDetector")):
        expected = [s / n for s, n in zip(scanner[extent], scanner[counts])]
        if spacing in scanner:
            require(len(scanner[spacing]) == len(expected) and
                    all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(scanner[spacing], expected)),
                    "inconsistent " + spacing)
        geometry[spacing] = expected
    volume = local_path(root, (Path(entry["source"]) / meta["vol"]).as_posix())
    return dict(entry, source_volume=volume.relative_to(root.resolve()).as_posix(),
                source_sha256=sha256(volume), metadata_sha256=sha256(metadata_path),
                geometry=geometry, geometry_sha256=digest(geometry),
                units={"length": "supplied dataset length unit; physical calibration unknown",
                       "intensity": "supplied attenuation values, not HU",
                       "scanner_angles": "degrees", "projection_angles": "radians"})


def provenance(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    files = git("ls-files", "*.py", "*.cu", "*.cpp", "*.h", "*.hpp", "*.yml", ".gitmodules").splitlines()
    files += [p.relative_to(root).as_posix() for p in
              sorted((root / "experiments/limited_angle_generalization").rglob("*.py"))]
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "git_status": git("status", "--short"), "tracked_diff": git("diff", "HEAD", "--"),
            "submodules": git("submodule", "status"), "python": sys.version,
            "source_files_sha256": {p: sha256(root / p) for p in files}}


def build_manifest(config, root=REPO):
    root = root.resolve()
    require(config["schema_version"] == 1, "unsupported config version")
    require(config["noise"] == {"model": "none", "seed": 0}, "only noise-free preparation is supported")
    require(config["method"] == {"name": "r2_default", "optimization_seed": 0,
            "initialization_seed": 0, "iterations": 30000, "lambda_tv": 0.05,
            "n_points": 50000, "density_thresh": 0.05, "density_rescale": 0.15,
            "geometry_term": False}, "this planner supports only the unchanged baseline, seed zero")
    volumes = {}
    identities = {}
    hashes = {}
    for entry in config["volumes"]:
        require(entry["volume_id"] not in volumes, "duplicate volume ID")
        require(entry["population_role"] in ("development", "validation", "final_test"), "invalid population role")
        require(bool(entry["independence_group"]), "independence group required")
        record = source_record(root, entry)
        for key, table in ((record["independence_group"], identities), (record["source_sha256"], hashes)):
            require(key not in table or table[key] == entry["population_role"], "volume/patient split leakage")
            table[key] = entry["population_role"]
        volumes[entry["volume_id"]] = record
    specs = config["acquisitions"]
    require(bool(specs), "empty acquisition list")
    training = [train_angles(s["start_degrees"], s["span_degrees"], s["view_count"]) for s in specs]
    evaluation, offset = common_evaluation(training, config["evaluation_count"])
    cases, ids, destinations, acquisitions = [], set(), set(), set()
    for spec, angles in zip(specs, training):
        case_id = spec["case_id"]
        require(isinstance(case_id, str) and re.fullmatch(r"[a-z0-9_]+", case_id), "unsafe case ID")
        require(case_id not in ids, "duplicate case ID")
        ids.add(case_id)
        require(spec["volume_id"] in volumes, "unknown volume ID")
        require(spec["acquisition_role"] in ("development", "validation", "final_test"), "invalid acquisition role")
        volume = volumes[spec["volume_id"]]
        acquisition_key = (volume["source_sha256"], volume["geometry_sha256"],
                           spec["start_degrees"] % 360, spec["span_degrees"], spec["view_count"])
        require(acquisition_key not in acquisitions, "duplicate acquisition under different IDs")
        acquisitions.add(acquisition_key)
        run_id = case_id + "_r2_default_s0_i30000"
        data_dir = "data/trdp2/limited_angle_generalization_v1/" + case_id
        output_dir = "output/limited_angle_generalization_v1/" + run_id
        initializer = data_dir + "/init_" + case_id + ".npy"
        for destination in (data_dir, output_dir):
            require(destination not in destinations, "duplicate output destination")
            destinations.add(destination)
        count = sum(inside(a, spec["start_degrees"], spec["span_degrees"]) for a in evaluation)
        case = dict(spec, population_role=volume["population_role"],
                    independence_group=volume["independence_group"], run_id=run_id,
                    source=volume["source"], source_sha256=volume["source_sha256"],
                    geometry_sha256=volume["geometry_sha256"], endpoint="half_open",
                    train_degrees=angles, train_radians=[math.radians(a) for a in angles],
                    evaluation_degrees=evaluation, evaluation_radians=[math.radians(a) for a in evaluation],
                    evaluation_group=digest([volume["source_sha256"], volume["geometry_sha256"], evaluation]),
                    inside_count=count, outside_count=len(evaluation) - count,
                    min_train_test_separation_degrees=min_separation(angles, evaluation),
                    noise=config["noise"], method=config["method"], data_directory=data_dir,
                    output_directory=output_dir, initializer=initializer,
                    initializer_sha256=None,
                    planned_commands={
                        "initialize": ["python", "initialize_pcd.py", "--data", data_dir,
                                       "--output", initializer, "--recon_method", "fdk",
                                       "--n_points", "50000", "--density_thresh", "0.05",
                                       "--density_rescale", "0.15"],
                        "train": ["python", "train.py", "--source_path", data_dir,
                                  "--model_path", output_dir, "--ply_path", initializer,
                                  "--iterations", "30000", "--lambda_tv", "0.05",
                                  "--checkpoint_iterations", "5000", "10000", "15000",
                                  "20000", "25000", "30000"]},
                    expected_inputs=[data_dir + "/meta_data.json", data_dir + "/vol_gt.npy",
                                     data_dir + "/dataset.sha256"],
                    status="planned_not_generated", provenance="source inventory; no experiment executed")
        cases.append(case)
    result = {"schema_version": 1, "config": copy.deepcopy(config), "volumes": volumes,
              "evaluation_offset_bins": offset, "angle_tolerance_degrees": TOL_DEG,
              "cases": cases, "provenance": provenance(root),
              "execution_gate": "preparation only; dataset generation and run lifecycle review pending; verify backend smoke separately"}
    result["manifest_sha256"] = digest(result)
    return result


def validate_manifest(manifest, root=REPO, check_files=True):
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(digest(payload) == manifest["manifest_sha256"], "manifest content hash mismatch")
    # Re-resolve every derived field and source identity, never trust self-consistent edits.
    rebuilt = build_manifest(manifest["config"], root)
    for key in ("schema_version", "volumes", "cases", "evaluation_offset_bins", "angle_tolerance_degrees", "execution_gate"):
        require(manifest[key] == rebuilt[key], "manifest/source mismatch: " + key)
    if check_files:
        require(manifest["provenance"].get("submodules") == rebuilt["provenance"].get("submodules"),
                "submodule identity changed since preparation")
        require(manifest["provenance"]["source_files_sha256"] == rebuilt["provenance"]["source_files_sha256"],
                "source file set or contents changed since preparation")
        for path, expected in manifest["provenance"]["source_files_sha256"].items():
            require(sha256(local_path(root, path)) == expected, "code changed since preparation: " + path)
    for case in manifest["cases"]:
        validate_angles(case["train_degrees"], "training")
        validate_angles(case["evaluation_degrees"], "evaluation")
        require(case["min_train_test_separation_degrees"] > TOL_DEG, "training/evaluation overlap")
    return manifest


def output_state(path):
    path = Path(path)
    if not path.exists():
        return "new"
    if (path / "eval/iter_030000/eval3d.yml").is_file() and (path / "point_cloud/iteration_30000/vol_pred.npy").is_file():
        return "completed_artifacts_present_unverified"
    if list(path.glob("ckpt/*.pth")):
        return "checkpoint_present_identity_unverified_resume_forbidden"
    return "existing_incomplete_or_failed_inspect_logs"


def ensure_new_outputs(manifest, root=REPO):
    for case in manifest["cases"]:
        for key in ("data_directory", "output_directory"):
            path = local_path(root, case[key])
            require(not path.exists(), "refusing existing destination: " + str(path) + " (" + output_state(path) + ")")


def write_new_json(path, value):
    # Exclusive create: an existing artifact is never truncated, even on a race.
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_pilot(root=REPO):
    summary, paired, geometries = [], [], []
    for folder, start, span in (("arc_360", 0, 360), ("arc_120", 0, 120), ("arc_120_start_90", 90, 120)):
        directory = root / "data/trdp2/chest_50views_noisefree" / folder / "0_chest_cone"
        meta = json.loads((directory / "meta_data.json").read_text())
        require(meta["scanner"]["startAngle"] == start and meta["scanner"]["totalAngle"] == span,
                "pilot scanner angles differ (expected degrees)")
        require(meta["scanner"]["noise"] is False, "pilot is not noise-free")
        geometries.append({key: meta["scanner"][key] for key in PHYSICAL_KEYS})
        train = [frame["angle"] for frame in meta["proj_train"]]
        test = [frame["angle"] for frame in meta["proj_test"]]
        expected_train = [math.radians(a) for a in train_angles(start, span, 50)]
        expected_test = [math.radians((i + 0.5) * 3.6) for i in range(100)]
        require(len(train) == 50 and len(test) == 100, "pilot projection counts differ")
        require(all(abs(a-b) < 1e-12 for a,b in zip(train, expected_train)), "pilot train angles differ")
        require(all(abs(a-b) < 1e-12 for a,b in zip(test, expected_test)), "pilot evaluation angles differ")
        degrees = [math.degrees(a) for a in test]
        require(min_separation(train_angles(start, span, 50), degrees) > TOL_DEG, "pilot overlap")
        fingerprint = [sha256(directory / meta["vol"])] + [sha256(directory / f["file_path"]) for f in meta["proj_test"]]
        paired.append(fingerprint)
        count = sum(inside(a, start, span) for a in degrees)
        summary.append({"case": folder, "inside": count, "outside": 100-count,
                        "reference_sha256": fingerprint[0], "evaluation_files_sha256": digest(fingerprint[1:])})
    require(paired[0] == paired[1] == paired[2], "pilot paired reference/evaluation files differ")
    require(geometries[0] == geometries[1] == geometries[2], "pilot physical geometries differ")
    require([s["inside"] for s in summary] == [100, 33, 33], "unexpected pilot sectors")
    return summary
