"""比较两个 legacy baseline manifest 的结构性差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    paths = ("git.commit", "git.diff_sha256", "effective_configs",
             "checkpoint.files", "dataset.files", "inference_outputs")

    def lookup(value, path):
        for component in path.split("."):
            value = value[component]
        return value

    differences = [path for path in paths
                   if lookup(reference, path) != lookup(candidate, path)]
    result = {"equal": not differences, "differences": differences}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if not differences else 1)


if __name__ == "__main__":
    main()
