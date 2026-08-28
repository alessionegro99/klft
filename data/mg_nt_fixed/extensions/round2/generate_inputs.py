#!/usr/bin/env python3
"""Generate the second fixed-``m_G N_t`` bracket extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from data.mg_nt_fixed.pilot import generate_inputs as pilot  # noqa: E402


# Draws 113--118 (zero based) from the documented campaign RNG stream.
PRODUCTION_SEEDS = (767348994, 1941201624, 190362085, 1526132929)
SMOKE_SEED = 922096603
TIMING_SEED = 1947777267
ENSEMBLE = pilot.Ensemble("0.5", 3, "4.290", 80)


def _failed_groups(summary: dict[str, object]) -> dict[tuple[float, int], list[str]]:
    """Return failed tuning groups and their reasons."""
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise ValueError("tuning summary has no group list")
    failed: dict[tuple[float, int], list[str]] = {}
    for group in groups:
        if not isinstance(group, dict) or bool(group.get("accepted")):
            continue
        reasons = group.get("failure_reasons")
        if not isinstance(reasons, list):
            raise ValueError("failed tuning group has no reason list")
        failed[(float(group["target_c"]), int(group["physical_Nt"]))] = [
            str(reason) for reason in reasons
        ]
    return failed


def _existing_seeds(excluded: set[Path]) -> set[int]:
    """Return explicit seeds in every other repository manifest."""
    seeds: set[int] = set()
    for manifest in (ROOT / "data").rglob("*manifest.tsv"):
        if manifest.resolve() in excluded:
            continue
        with manifest.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None or "seed" not in reader.fieldnames:
                continue
            seeds.update(int(row["seed"]) for row in reader)
    return seeds


def _row(
    array_index: int,
    seed: int,
    sweeps: int,
    run: str,
    input_path: Path,
    purpose: str,
    text: str,
    uv_lock_sha256: str,
) -> dict[str, object]:
    """Return one manifest row."""
    return {
        "array_index": array_index,
        "campaign_type": pilot.CAMPAIGN_TYPE,
        "target_c": ENSEMBLE.target_c,
        "run": run,
        "input": input_path.as_posix(),
        "purpose": purpose,
        "physical_Nt": ENSEMBLE.nt,
        "stored_L2": ENSEMBLE.nt + 1,
        "Ns": ENSEMBLE.ns,
        "beta": ENSEMBLE.beta,
        "chain_index": array_index,
        "seed": seed,
        "sweeps": sweeps,
        "measurement_interval": pilot.MEASUREMENT_INTERVAL,
        "dirichlet_suite_measurement_interval": pilot.SUITE_MEASUREMENT_INTERVAL,
        "input_sha256": pilot.sha256_text(text),
        "heatbath_sha256": pilot.HEATBATH_SHA256,
        "klft_commit": pilot.KLFT_COMMIT,
        "uv_lock_sha256": uv_lock_sha256,
    }


def generate(tuning_summary: Path, output: Path) -> list[Path]:
    """Write the immutable production and validation inputs."""
    summary_text = tuning_summary.read_text(encoding="utf-8")
    failures = _failed_groups(json.loads(summary_text))
    if set(failures) != {(0.5, 3)}:
        raise ValueError(f"unexpected failed tuning groups: {sorted(failures)}")
    if "log-linear interpolation has p < 0.05" not in failures[(0.5, 3)]:
        raise ValueError("the round-2 point is not justified by the tuning gate")

    all_seeds = (*PRODUCTION_SEEDS, SMOKE_SEED, TIMING_SEED)
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("round-2 seeds are not unique")
    manifests = {output / "manifest.tsv", output / "validation_manifest.tsv"}
    collision = set(all_seeds) & _existing_seeds(manifests)
    if collision:
        raise RuntimeError(f"round-2 seed is already used: {min(collision)}")

    uv_lock_sha256 = pilot.sha256_file(ROOT / "uv.lock")
    name = pilot.ensemble_name(ENSEMBLE)
    pending: list[tuple[Path, str]] = []
    rows: list[dict[str, object]] = []
    for chain_index, seed in enumerate(PRODUCTION_SEEDS):
        relative = Path(name) / f"seed_{seed}" / "input.yaml"
        text = pilot.input_text(ENSEMBLE, seed)
        pending.append((output / relative, text))
        rows.append(
            _row(
                chain_index,
                seed,
                pilot.SWEEPS,
                f"mg_nt_fixed/extensions/round2/{name}/seed_{seed}",
                relative,
                "failed_log_linear_gate_bracket_extension",
                text,
                uv_lock_sha256,
            )
        )

    validation_rows: list[dict[str, object]] = []
    for index, (label, seed, sweeps) in enumerate(
        (("smoke", SMOKE_SEED, 25), ("timing", TIMING_SEED, 100))
    ):
        relative = Path(label) / "input.yaml"
        text = pilot.input_text(ENSEMBLE, seed, sweeps)
        pending.append((output / relative, text))
        validation_rows.append(
            _row(
                index,
                seed,
                sweeps,
                f"mg_nt_fixed/extensions/round2/validation/{label}",
                relative,
                f"round2_{label}_validation",
                text,
                uv_lock_sha256,
            )
        )

    seed_rows = "purpose\tchain_index\tseed\tmaster_seed_draw_index\n"
    for index, seed in enumerate(PRODUCTION_SEEDS):
        seed_rows += f"production\t{index}\t{seed}\t{113 + index}\n"
    seed_rows += f"smoke\t0\t{SMOKE_SEED}\t117\n"
    seed_rows += f"timing\t0\t{TIMING_SEED}\t118\n"
    pending.extend(
        (
            (output / "manifest.tsv", pilot.manifest_text(rows)),
            (output / "validation_manifest.tsv", pilot.manifest_text(validation_rows)),
            (output / "seed_provenance.tsv", seed_rows),
            (
                output / "trigger_tuning_summary_sha256.txt",
                f"{pilot.sha256_text(summary_text)}\n",
            ),
            (output / "master_seed.txt", f"{pilot.MASTER_SEED}\n"),
            (output / "klft_commit.txt", f"{pilot.KLFT_COMMIT}\n"),
            (output / "heatbath_sha256.txt", f"{pilot.HEATBATH_SHA256}\n"),
            (output / "uv_lock_sha256.txt", f"{uv_lock_sha256}\n"),
        )
    )
    existing = [path for path, _ in pending if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    for path, text in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    return [path for path, _ in pending]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    paths = generate(args.tuning_summary, args.output)
    print("wrote four production chains and two validation inputs")
    print(f"files created: {len(paths)}")


if __name__ == "__main__":
    main()
