#!/usr/bin/env python3
"""Prepare immutable Nt=1 matched-beta inputs after the P100 build is available.

This command reads compact metadata only and submits no jobs. The c directory
labels identify the source Nt=2 tuning, not the mass measured at Nt=1.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


REFERENCE_C = (0.5, 1.0, 2.0)
MASTER_SEED = 20260724


def sha256(path: Path) -> str:
    """Return a SHA256 digest of a compact provenance file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token(value: str | int | float) -> str:
    """Return the canonical decimal directory token without rounding."""
    return format(Decimal(str(value)).normalize(), "f").replace(".", "p")


def selected_tunings(path: Path) -> dict[float, dict[str, str]]:
    """Read the unique included Nt=2 tuning for each reference c."""
    selected = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (int(row["physical_Nt"]) != 2 or
                    any(row[key].lower() != "true" for key in
                        ("available", "selected", "include"))):
                continue
            c = float(row["target_c"])
            if c not in REFERENCE_C:
                continue
            beta = float(row["beta_star"])
            if c in selected or not math.isfinite(beta) or beta <= 0:
                raise ValueError(f"Ambiguous or undefined Nt=2 tuning for c={c}")
            if int(row["bootstrap_samples"]) != 500:
                raise ValueError("Nt=1 inputs require the current 500-replica tuning")
            selected[c] = row
    if set(selected) != set(REFERENCE_C):
        raise ValueError("Need included Nt=2 tuning results for c=0.5, 1 and 2")
    return selected


def unique_seeds(reserved: set[int], count: int) -> list[int]:
    """Draw distinct positive int32 seeds, matching KLFT's input seed type."""
    rng = np.random.default_rng(MASTER_SEED)
    used = set(reserved)
    seeds = []
    while len(seeds) < count:
        seed = int(rng.integers(1, np.iinfo(np.int32).max))
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


def render_input(run: dict) -> str:
    """Render pure-Wilson, open-slab input with the production cadences."""
    return f'''# Nt=1 matched-beta diagnostic; source Nt=2 reference c={run['reference_c']}.
# The actual Nt=1 mG*Nt is to be measured; it is not fixed to reference_c.
HeatbathParams:
  L0: {run['Ns']}
  L1: {run['Ns']}
  L2: 2
  L3: 1
  nSweep: {run['sweeps']}
  nOverrelax: 5
  seed: {run['seed']}
  start: "hot"
  temporal_dirichlet: true
  beta: {run['beta_text']}
  epsilon1: 0.0
  epsilon2: 0.0
  configuration_output: "configuration_final.klft"
GaugeObservableParams:
  measurement_interval: 1
  dirichlet_suite_measurement_interval: 5
  measure_plaquette: false
  measure_dirichlet_holonomy: true
  measure_dirichlet_plaquette_profiles: true
  measure_boundary_wilson_loops: false
  measure_dirichlet_bulk_polyakov: false
  dirichlet_correlator_max_r: 0
  dirichlet_holonomy_correlator_filename: "holonomy_correlator.out"
  dirichlet_holonomy_summary_filename: "holonomy_summary.out"
  dirichlet_plaquette_profile_filename: "plaquette_profile.out"
  write_to_file: true
GradientFlowParams:
  enabled: false
PartitioningParams:
  enabled: false
'''


def generate(output: Path, tuning: Path, build: Path, reserved: Path,
             sizes: tuple[int, int, int]) -> dict:
    """Write one smoke, three timing and twelve production inputs without overwrite.

    Parameters
    ----------
    sizes
        Candidate Ns values in reference-c order (0.5, 1, 2). Physical volume
        adequacy is still unvalidated. Timing and GPU validation are separate
        mandatory steps before any manual production submission.
    """
    if len(sizes) != 3 or any(ns < 16 or ns * ns > np.iinfo(np.int32).max
                              for ns in sizes):
        raise ValueError("Supply three valid Ns values >=16 in c=(0.5,1,2) order")
    manifest_path = output / "nt1_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    tunings = selected_tunings(tuning)
    fields = {}
    for line in build.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            fields[parts[0]] = parts[1]
    commit = fields.get("klft_commit", "")
    executable_hash = fields.get("heatbath_sha256", "")
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or not re.fullmatch(
            r"[a-f0-9]{64}", executable_hash):
        raise ValueError("Build provenance must identify the commit and executable")
    if fields.get("build_path") != "/qbigwork2/negro/dirichlet/build/klft":
        raise ValueError("Use the sole project P100 build")
    if fields.get("slab_boundary_flag") != "3":
        raise ValueError("Use a physical open-slab build with checkpoint flag 3")
    options = fields.get("cmake_options", "").split(";")
    if not {"KLFT_NDIM=3", "KLFT_NC=2", "CUDA=ON", "PASCAL60=ON"} <= set(options):
        raise ValueError("Build provenance must identify a 3D SU(2) P100 build")
    inventory = json.loads(reserved.read_text())
    if inventory.get("source_root") != "/qbigwork2/negro/dirichlet/data":
        raise ValueError("Seed inventory must cover the qbig project data tree")
    reserved_values = inventory["seeds"]
    if not all(type(seed) is int and seed > 0 for seed in reserved_values):
        raise ValueError("Invalid seed inventory")
    seeds = iter(unique_seeds(set(reserved_values), 16))
    runs = []
    for c, ns in zip(REFERENCE_C, sizes):
        row = tunings[c]
        purposes = [("timing", 100, ns, 0)] + [
            ("production", 8000, ns, i) for i in range(4)]
        if c == 0.5:
            purposes.insert(0, ("smoke", 8, 4, 0))
        for purpose, sweeps, size, chain in purposes:
            ensemble = f"c_{token(c)}/Nt_1/Ns_{size}/beta_{token(row['beta_star'])}"
            seed = next(seeds)
            run = {
                "purpose": purpose, "chain_index": chain,
                "campaign_type": "nt1_matched_beta", "reference_c": c,
                "target_c": None, "source_tuning_Nt": 2,
                "fixed_c_continuum_input": False,
                "physical_Nt": 1, "stored_L2": 2, "Ns": size,
                "beta": float(row["beta_star"]), "beta_text": row["beta_star"],
                "seed": seed, "sweeps": sweeps, "n_overrelax": 5,
                "measurement_interval": 1, "dirichlet_suite_measurement_interval": 5,
                "boundary_wilson_enabled": False, "ensemble": ensemble,
                "input": f"{ensemble}/seed_{seed}/input.yaml",
                "klft_commit": commit, "heatbath_sha256": executable_hash,
                "reference_tuning_flags": [flag for flag in row["flags"].split(";") if flag],
                "volume_status": "unvalidated",
            }
            run["raw_path"] = str(Path("/qbigwork2/negro/dirichlet/data/mg_nt_fixed")
                                  / Path(run["input"]).parent)
            run["input_sha256"] = hashlib.sha256(render_input(run).encode()).hexdigest()
            if (output / run["input"]).parent.exists():
                raise FileExistsError((output / run["input"]).parent)
            runs.append(run)
    manifest = {
        "campaign_type": "nt1_matched_beta", "master_seed": MASTER_SEED,
        "analysis_bootstrap_samples": 500,
        "boundary_conditions": "spatial periodic; open temporal slab; fixed spatial boundary links",
        "coupling_convention": "3D SU(2) Wilson beta=4/(a*g3_squared)",
        "reference_c_definition": "target c of the source Nt=2 tuning, not an Nt=1 target",
        "required_before_production": ["P100 correctness validation", "100-sweep production-cadence timing",
                                       "projected <=2 GPU-hours per chain and <=12 GPU-hours total"],
        "tuning_sha256": sha256(tuning), "build_provenance_sha256": sha256(build),
        "seed_inventory_sha256": sha256(reserved), "runs": runs,
    }
    for run in runs:
        path = output / run["input"]
        path.parent.mkdir(parents=True, exist_ok=False)
        with path.open("x") as stream:
            stream.write(render_input(run))
        with path.with_name("run_metadata.json").open("x") as stream:
            json.dump(run, stream, indent=2, allow_nan=False)
            stream.write("\n")
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main() -> None:
    """Generate inputs from verified compact provenance; never submit jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--build-provenance", type=Path, required=True)
    parser.add_argument("--reserved-seeds", type=Path, required=True)
    parser.add_argument("--ns", type=int, nargs=3, required=True,
                        metavar=("NS_C05", "NS_C1", "NS_C2"))
    args = parser.parse_args()
    result = generate(args.output, args.tuning, args.build_provenance,
                      args.reserved_seeds, tuple(args.ns))
    print(f"Prepared {len(result['runs'])} immutable inputs; no jobs submitted.")


if __name__ == "__main__":
    main()
