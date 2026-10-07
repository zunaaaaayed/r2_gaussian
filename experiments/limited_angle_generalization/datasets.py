"""Dataset generation and integrity checks; GPU imports occur only in a worker."""
import copy
import json
from pathlib import Path
import shutil
import sys

from benchmark import REPO, PHYSICAL_KEYS, local_path, require, sha256, write_new_json


def load_array(path, shape):
    import numpy as np
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    require(array.shape == tuple(shape), "array shape mismatch: " + str(path))
    require(array.dtype == np.float32 and bool(np.isfinite(array).all()),
            "expected finite float32: " + str(path))
    return array


def file_hashes(directory):
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "dataset.sha256"}


def seal(directory):
    hashes = file_hashes(directory)
    with (directory / "dataset.sha256").open("x") as stream:
        for name, value in hashes.items():
            stream.write(value + "  " + name + "\n")
    return hashes


def verify_seal(directory, required):
    hashes = {}
    for line in (directory / "dataset.sha256").read_text().splitlines():
        value, name = line.split("  ", 1)
        require(name not in hashes, "duplicate checksum path")
        path = local_path(directory, name)
        require(path.is_file() and sha256(path) == value, "dataset checksum mismatch: " + name)
        hashes[name] = value
    require(set(hashes) == set(required), "checksum file set differs from expected inputs")
    return hashes


def shared_directory(case, root=REPO):
    return local_path(root, "data/trdp2/limited_angle_generalization_v1/_shared/" + case["evaluation_group"])


def expected_frames(split, angles):
    return [{"file_path": "{0}/{0}_{1:04d}.npy".format(split, i), "angle": angle}
            for i, angle in enumerate(angles)]


def shared_identity(manifest, case):
    return {"evaluation_group": case["evaluation_group"], "source_sha256": case["source_sha256"],
            "geometry_sha256": case["geometry_sha256"], "evaluation_radians": case["evaluation_radians"],
            "source_files_sha256": manifest["provenance"]["source_files_sha256"]}


def verify_shared(manifest, case, root=REPO):
    shared = shared_directory(case, root)
    record = json.loads((shared / "shared.json").read_text())
    expected = shared_identity(manifest, case)
    require(record == expected, "shared acquisition identity mismatch")
    frames = expected_frames("proj_test", case["evaluation_radians"])
    hashes = verify_seal(shared, ["shared.json", "vol_gt.npy"] + [f["file_path"] for f in frames])
    require(hashes["vol_gt.npy"] == case["source_sha256"], "shared reference mismatch")
    scanner = manifest["volumes"][case["volume_id"]]["geometry"]
    load_array(shared / "vol_gt.npy", scanner["nVoxel"])
    for frame in frames:
        load_array(shared / frame["file_path"], scanner["nDetector"])
    return shared, hashes


def verify_dataset(manifest, case, root=REPO):
    directory = local_path(root, case["data_directory"])
    meta = json.loads((directory / "meta_data.json").read_text())
    generation = json.loads((directory / "generation.json").read_text())
    require(generation["manifest_sha256"] == manifest["manifest_sha256"] and
            generation["case_id"] == case["case_id"], "dataset/manifest identity mismatch")
    require(generation["reference_reprojection_max_error"] <= 1e-5 and
            generation["reference_reprojection_max_error"] >= 0, "reference reprojection failed")
    scanner = manifest["volumes"][case["volume_id"]]["geometry"]
    for key in PHYSICAL_KEYS:
        require(meta["scanner"][key] == scanner[key], "physical geometry mismatch: " + key)
    require(meta["scanner"]["startAngle"] == case["start_degrees"] and
            meta["scanner"]["totalAngle"] == case["span_degrees"] and
            meta["scanner"]["noise"] is False, "acquisition metadata mismatch")
    require(meta["vol"] == "vol_gt.npy", "unexpected reference path")
    files = ["meta_data.json", "generation.json", "vol_gt.npy"]
    for split, angles in (("proj_train", case["train_radians"]), ("proj_test", case["evaluation_radians"])):
        frames = expected_frames(split, angles)
        require(meta[split] == frames, "frame paths/radian angles differ: " + split)
        files.extend(frame["file_path"] for frame in frames)
    hashes = verify_seal(directory, files)
    require(hashes["vol_gt.npy"] == case["source_sha256"], "reference hash mismatch")
    shared, shared_hashes = verify_shared(manifest, case, root)
    for name, value in shared_hashes.items():
        if name != "shared.json":
            require(hashes[name] == value, "paired reference/evaluation mismatch: " + name)
    load_array(directory / "vol_gt.npy", scanner["nVoxel"])
    for split in ("proj_train", "proj_test"):
        for frame in meta[split]:
            load_array(directory / frame["file_path"], scanner["nDetector"])
    return {"dataset_sha256": sha256(directory / "dataset.sha256"), "files_verified": len(hashes),
            "shared_checksum_sha256": sha256(shared / "dataset.sha256")}


def generate_dataset(manifest, case, root=REPO, projector=None):
    """Worker entry point; projector injection is reserved for CPU correctness tests."""
    import numpy as np
    directory = local_path(root, case["data_directory"])
    require(not directory.exists(), "refusing existing dataset")
    source = manifest["volumes"][case["volume_id"]]
    volume_path = local_path(root, source["source_volume"])
    require(sha256(volume_path) == source["source_sha256"], "source changed")
    volume = load_array(volume_path, source["geometry"]["nVoxel"])
    scanner = copy.deepcopy(source["geometry"])
    scanner.update(noise=False, startAngle=case["start_degrees"], totalAngle=case["span_degrees"])
    if projector is None:
        sys.path.insert(0, str(REPO / "experiments/limited_angle_120"))
        from generate import project
        projector = project

    def save_projections(target, split, angles):
        (target / split).mkdir()
        frames = expected_frames(split, angles)
        for offset in range(0, len(angles), 10):
            batch = np.asarray(projector(volume, scanner, np.array(angles[offset:offset+10], dtype=np.float64)))
            require(batch.shape == (min(10, len(angles)-offset), *scanner["nDetector"]), "projector shape mismatch")
            require(batch.dtype == np.float32 and bool(np.isfinite(batch).all()), "invalid projections")
            for i, projection in enumerate(batch):
                np.save(target / frames[offset+i]["file_path"], projection, allow_pickle=False)
        return frames

    shared = shared_directory(case, root)
    if not shared.exists():
        shared.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(volume_path, shared / "vol_gt.npy")
        save_projections(shared, "proj_test", case["evaluation_radians"])
        write_new_json(shared / "shared.json", shared_identity(manifest, case))
        seal(shared)
    verify_shared(manifest, case, root)  # Also rejects partial/corrupt shared groups.
    reference = projector(volume, scanner, np.array(case["evaluation_radians"][:1], dtype=np.float64))
    expected = np.load(shared / "proj_test/proj_test_0000.npy", allow_pickle=False)
    error = float(np.max(np.abs(reference[0] - expected)))
    require(np.isfinite(error) and error <= 1e-5, "reference reprojection mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(shared / "vol_gt.npy", directory / "vol_gt.npy")
    shutil.copytree(shared / "proj_test", directory / "proj_test")
    frames = save_projections(directory, "proj_train", case["train_radians"])
    source_meta = json.loads((local_path(root, case["source"]) / "meta_data.json").read_text())
    meta = {"scanner": scanner, "vol": "vol_gt.npy", "proj_train": frames,
            "proj_test": expected_frames("proj_test", case["evaluation_radians"])}
    if "bbox" in source_meta:
        meta["bbox"] = source_meta["bbox"]
    write_new_json(directory / "meta_data.json", meta)
    write_new_json(directory / "generation.json", {
        "manifest_sha256": manifest["manifest_sha256"], "case_id": case["case_id"],
        "reference_reprojection_max_error": error,
        "note": "One evaluation reference view reprojected; internal consistency, not independent physics validation"})
    seal(directory)
    return verify_dataset(manifest, case, root)
