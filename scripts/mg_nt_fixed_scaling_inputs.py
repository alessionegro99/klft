#!/usr/bin/env python3
"""Prepare the approved four-ensemble matched-y diagnostic, without submission.

The c=0.5 tree label identifies the source coarse-resolution mass-grid family.
Every new target_c is null: these simulations fix beta/Nt, not a measured mass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from mg_nt_fixed_nt1_inputs import MASTER_SEED, token, unique_seeds
from mg_nt_fixed_nt1_run import ROOT, sha256, write_json


GRID = ((3, 96, "4.5"), (4, 160, "6.0"), (6, 320, "9.0"), (4, 160, "6.2"))
NT1_GRID = tuple((1, 128, beta) for beta in ("1.45", "1.50", "1.5333333333333333", "1.55"))
COMMIT = "f47421d895469b72d4af8a9b8479807f2b09a692"


def render_input(run: dict) -> str:
    """Render the open Dirichlet slab and its approved measurement cadences."""
    return f'''# Matched beta/Nt diagnostic; c=0.5 is a reference family only.
# target_c=null; exclude from fixed-c continuum inputs until explicitly retuned.
HeatbathParams:
  L0: {run['Ns']}
  L1: {run['Ns']}
  L2: {run['stored_L2']}
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


def generate(output: Path, build: Path, validation: Path, reserved: Path,
             *, nt1: bool = False) -> dict:
    """Write one smoke, four timings and sixteen production inputs exclusively."""
    manifest_path = output / ("nt1_scaling_manifest.json" if nt1 else "scaling_manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    fields = dict(line.split(maxsplit=1) for line in build.read_text().splitlines()
                  if len(line.split(maxsplit=1)) == 2)
    backend = "a100" if fields.get("build_path") == str(ROOT / "build/klft-a100") else "p100"
    if nt1 and backend != "p100":
        raise ValueError("The Nt=1 matched-y extension uses the approved P100 build")
    expected_path = ROOT / ("build/klft-a100" if backend == "a100" else "build/klft")
    options = set(fields.get("cmake_options", "").split(";"))
    architecture = "AMPERE80=ON" if backend == "a100" else "PASCAL60=ON"
    if (fields.get("klft_commit") != COMMIT or fields.get("slab_boundary_flag") != "3"
            or fields.get("build_path") != str(expected_path)
            or not re.fullmatch(r"[a-f0-9]{64}", fields.get("heatbath_sha256", ""))
            or not {"KLFT_NDIM=3", "KLFT_NC=2", "CUDA=ON", architecture} <= options):
        raise ValueError("Need the validated, pinned native 3D SU(2) open-slab build")
    required = {f"klft_commit {COMMIT}", f"heatbath_sha256 {fields['heatbath_sha256']}",
                "ctest passed", "cuda_memcheck passed"}
    if not required <= set(validation.read_text().splitlines()):
        raise ValueError("Build and successful GPU validation do not match")
    inventory = json.loads(reserved.read_text())
    if inventory["source_root"] != str(ROOT / "data"):
        raise ValueError("Seed inventory must cover the remote project")
    used = set(inventory["seeds"])
    # Local unsubmitted inputs also reserve their seeds.
    for path in output.parent.rglob("input.yaml"):
        match = re.search(r"^\s*seed:\s*(\d+)\s*$", path.read_text(), re.M)
        if match:
            used.add(int(match.group(1)))
    seeds = iter(unique_seeds(used, 21))
    grid = NT1_GRID if nt1 else GRID
    settings = [("smoke", 1, 4, "1.55", 10, 0)] if nt1 else [("smoke", 6, 4, "9.0", 10, 0)]
    settings += [("timing", nt, ns, beta, 100, 0) for nt, ns, beta in grid]
    settings += [("production", nt, ns, beta, 8000, chain)
                 for nt, ns, beta in grid for chain in range(4)]
    runs = []
    for purpose, nt, ns, beta, sweeps, chain in settings:
        seed = next(seeds)
        ensemble = f"c_0p5/Nt_{nt}/Ns_{ns}/beta_{token(beta)}"
        run = dict(purpose=purpose, chain_index=chain, campaign_type="matched_y_scaling",
                   reference_c=0.5, target_c=None, fixed_c_continuum_input=False,
                   beta_over_nt=float(beta) / nt, physical_Nt=nt, stored_L2=nt + 1,
                   Ns=ns, beta=float(beta), beta_text=beta, seed=seed, sweeps=sweeps,
                   n_overrelax=5, measurement_interval=1,
                   dirichlet_suite_measurement_interval=5, boundary_wilson_enabled=False,
                   ensemble=ensemble, input=f"{ensemble}/seed_{seed}/input.yaml",
                   klft_commit=COMMIT, heatbath_sha256=fields["heatbath_sha256"],
                   volume_status="unvalidated", start="hot")
        run["raw_path"] = str(ROOT / "data/mg_nt_fixed" / Path(run["input"]).parent)
        run["input_sha256"] = hashlib.sha256(render_input(run).encode()).hexdigest()
        if (output / run["input"]).parent.exists():
            raise FileExistsError((output / run["input"]).parent)
        runs.append(run)
    manifest = dict(campaign_type="matched_y_scaling", master_seed=MASTER_SEED,
                    analysis_bootstrap_samples=500, target_c=None,
                    fixed_c_continuum_input=False, backend=backend,
                    reference_c_definition="source c=0.5 coarse-resolution grid; no imposed mass",
                    boundary_conditions="spatial periodic; open temporal slab; fixed spatial surface links",
                    coupling_convention="3D SU(2) Wilson beta=4/(a*g3_squared)",
                    executable=str(expected_path / "binaries/heatbath"),
                    build_provenance_file=build.name, build_provenance_sha256=sha256(build),
                    validation_file=validation.name, validation_sha256=sha256(validation),
                    seed_inventory_sha256=sha256(reserved), reserved_seeds=sorted(used),
                    required_before_production=["new smoke and four 100-sweep timings",
                                                "manual timing gate: <=12 GPU-hours total; <=2 per chain"],
                    runs=runs)
    if nt1:
        manifest['source_sha256'] = {name:sha256(Path(__file__).parent/name) for name in
            ('mg_nt_fixed_scaling_inputs.py', 'mg_nt_fixed_nt1_inputs.py', 'mg_nt_fixed_nt1_run.py')}
        manifest['submission_sha256'] = sha256(output/'nt1_scaling_run.slurm')
        manifest['uv_lock_sha256'] = sha256(Path(__file__).resolve().parents[1]/'uv.lock')
        manifest['beta_ratio_note'] = '1.5333333333333333 represents 4.6/3 at float64 precision'
    for run in runs:
        path = output / run["input"]
        path.parent.mkdir(parents=True, exist_ok=False)
        with path.open("x") as stream:
            stream.write(render_input(run))
        write_json(path.with_name("run_metadata.json"), run)
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--nt1", action="store_true", help="prepare the approved Nt=1, Ns=128 matched-y extension")
    args = parser.parse_args()
    result = generate(args.output, args.build, args.validation, args.reserved, nt1=args.nt1)
    print(f"Prepared {len(result['runs'])} immutable inputs; no jobs submitted.")
