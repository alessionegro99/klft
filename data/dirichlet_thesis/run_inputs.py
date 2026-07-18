#!/usr/bin/env python3
"""Run generated thesis inputs, preserving each chain in its own directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path, help="3D SU(2) heatbath binary")
    parser.add_argument("root", type=Path, help="directory made by generate_inputs.py")
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    inputs = sorted(args.root.rglob("input.yaml"))
    if not inputs:
        raise FileNotFoundError(f"no */input.yaml files below {args.root}")
    for input_path in inputs:
        print(f"running {input_path.parent}", flush=True)
        subprocess.run(
            [str(binary), "-f", "input.yaml"],
            cwd=input_path.parent,
            check=True,
        )


if __name__ == "__main__":
    main()
