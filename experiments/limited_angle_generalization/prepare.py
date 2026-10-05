"""Resolve, freeze and validate benchmark inputs without launching GPU work."""

import argparse
import json
from pathlib import Path
import shlex
import sys

from benchmark import (REPO, build_manifest, check_pilot, ensure_new_outputs,
                       output_state, validate_manifest, write_new_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, default=None)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--write", type=Path, help="exclusive-create a resolved manifest; parent must exist")
    parser.add_argument("--check-pilot", action="store_true", help="read and hash historical arrays; requires pilot data")
    parser.add_argument("--dry-run", action="store_true", help="explicit read-only mode (also the default)")
    args = parser.parse_args()
    if args.write and args.dry_run:
        parser.error("--write and --dry-run are mutually exclusive")
    try:
        if args.check_pilot:
            print(json.dumps({"pilot_checks": check_pilot()}, indent=2))
        if args.manifest:
            manifest = validate_manifest(json.loads(args.manifest.read_text()))
        else:
            config = args.config or Path(__file__).with_name("configs") / "development.json"
            manifest = build_manifest(json.loads(config.read_text()))
            validate_manifest(manifest)
        ensure_new_outputs(manifest)
        print("Validated: {} cases, common evaluation offset {} bins".format(
            len(manifest["cases"]), manifest["evaluation_offset_bins"]))
        for case in manifest["cases"]:
            print("{case_id}: {view_count} views, [{start_degrees}, start+{span_degrees}), "
                  "inside/outside {inside_count}/{outside_count}, separation "
                  "{min_train_test_separation_degrees:.6f} deg; {acquisition_role}".format(**case))
            print("  planned output: {} ({})".format(case["output_directory"], output_state(REPO / case["output_directory"])))
            for stage, command in case["planned_commands"].items():
                print("  blocked {} template: {}".format(stage, shlex.join(command)))
        if args.write:
            write_new_json(args.write, manifest)
            print("Prepared manifest: " + str(args.write))
        print("PREPARATION ONLY: no generation, initialization, training, resume, or evaluation launched.")
        print(manifest["execution_gate"])
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("Validation failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
