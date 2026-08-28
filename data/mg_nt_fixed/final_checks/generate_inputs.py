#!/usr/bin/env python3
"""Generate the fixed-mass statistics and finite-volume checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.mg_nt_fixed.pilot import generate_inputs as pilot  # noqa: E402


# Draws 119--132 (zero based) from the documented campaign RNG stream.
PRODUCTION_SEEDS = (
    1648598430,
    106863780,
    124551628,
    490027902,
    2063971169,
    990456212,
    607905645,
    1827265464,
    1145773020,
    1542276537,
    1235269047,
    1139224139,
)
SMOKE_SEED = 1066829265
TIMING_SEED = 1690028198
ENSEMBLES = (
    (pilot.Ensemble("0.5", 3, "4.40", 80), 80000, "statistics_replacement"),
    (pilot.Ensemble("1.0", 4, "4.95", 96), 8000, "finite_volume_replacement"),
    (pilot.Ensemble("1.0", 4, "5.30", 96), 8000, "finite_volume_replacement"),
)
REANALYSIS_BETAS = {"3.70", "4.00", "4.30"}


def _existing_seeds(excluded: set[Path]) -> set[int]:
    seeds: set[int] = set()
    for manifest in (ROOT / "data").rglob("*manifest.tsv"):
        if manifest.resolve() in excluded:
            continue
        with manifest.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is not None and "seed" in reader.fieldnames:
                seeds.update(int(row["seed"]) for row in reader)
    return seeds


def _row(
    array_index: int,
    chain_index: int,
    seed: int,
    ensemble: pilot.Ensemble,
    sweeps: int,
    purpose: str,
    relative: Path,
    text: str,
    uv_lock_sha256: str,
) -> dict[str, object]:
    name = pilot.ensemble_name(ensemble)
    return {
        "array_index": array_index,
        "campaign_type": pilot.CAMPAIGN_TYPE,
        "target_c": ensemble.target_c,
        "run": f"mg_nt_fixed/final_checks/{name}/seed_{seed}",
        "input": relative.as_posix(),
        "purpose": purpose,
        "physical_Nt": ensemble.nt,
        "stored_L2": ensemble.nt + 1,
        "Ns": ensemble.ns,
        "beta": ensemble.beta,
        "chain_index": chain_index,
        "seed": seed,
        "sweeps": sweeps,
        "measurement_interval": pilot.MEASUREMENT_INTERVAL,
        "dirichlet_suite_measurement_interval": pilot.SUITE_MEASUREMENT_INTERVAL,
        "input_sha256": pilot.sha256_text(text),
        "heatbath_sha256": pilot.HEATBATH_SHA256,
        "klft_commit": pilot.KLFT_COMMIT,
        "uv_lock_sha256": uv_lock_sha256,
    }


def _validate_trigger(summary: dict[str, object]) -> None:
    if not bool(summary.get("all_nine_tunings_accepted")):
        raise ValueError("round-2 tuning has not accepted all nine groups")
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise ValueError("round-2 tuning summary has no groups")
    group = next(
        item
        for item in groups
        if float(item["target_c"]) == 0.5 and int(item["physical_Nt"]) == 3
    )
    selected = group["local_point_selection"]["selected_beta_values"]
    if 4.4 not in selected:
        raise ValueError("beta 4.40 is not selected and needs no statistics replacement")


def _reanalysis_manifest() -> str:
    source = ROOT / "data/mg_nt_fixed/pilot/manifest.tsv"
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        rows = [
            row
            for row in reader
            if row["target_c"] == "2.0"
            and row["physical_Nt"] == "4"
            and row["beta"] in REANALYSIS_BETAS
        ]
    if len(rows) != 12:
        raise ValueError("expected twelve c=2,Nt=4 pilot manifest rows")
    for index, row in enumerate(rows):
        row["array_index"] = str(index)
        row["purpose"] = "final_bootstrap_replacement"
    output = ["\t".join(fieldnames)]
    output.extend("\t".join(row[name] for name in fieldnames) for row in rows)
    return "\n".join(output) + "\n"


def generate(tuning_summary: Path, output: Path) -> list[Path]:
    """Write immutable production, validation, and reanalysis manifests."""
    summary_text = tuning_summary.read_text(encoding="utf-8")
    _validate_trigger(json.loads(summary_text))
    all_seeds = (*PRODUCTION_SEEDS, SMOKE_SEED, TIMING_SEED)
    manifests = {
        (output / "manifest.tsv").resolve(),
        (output / "validation_manifest.tsv").resolve(),
        (output / "reanalysis_manifest.tsv").resolve(),
    }
    collision = set(all_seeds) & _existing_seeds(manifests)
    if len(set(all_seeds)) != len(all_seeds) or collision:
        raise RuntimeError(f"final-check seed collision: {sorted(collision)}")

    uv_sha = pilot.sha256_file(ROOT / "uv.lock")
    pending: list[tuple[Path, str]] = []
    rows: list[dict[str, object]] = []
    for ensemble_index, (ensemble, sweeps, purpose) in enumerate(ENSEMBLES):
        name = pilot.ensemble_name(ensemble)
        for chain_index in range(4):
            array_index = 4 * ensemble_index + chain_index
            seed = PRODUCTION_SEEDS[array_index]
            relative = Path(name) / f"seed_{seed}" / "input.yaml"
            text = pilot.input_text(ensemble, seed, sweeps)
            pending.append((output / relative, text))
            rows.append(
                _row(
                    array_index,
                    chain_index,
                    seed,
                    ensemble,
                    sweeps,
                    purpose,
                    relative,
                    text,
                    uv_sha,
                )
            )

    validation_rows = []
    validation_ensemble = ENSEMBLES[-1][0]
    for index, (label, seed, sweeps) in enumerate(
        (("smoke", SMOKE_SEED, 25), ("timing", TIMING_SEED, 100))
    ):
        relative = Path(label) / "input.yaml"
        text = pilot.input_text(validation_ensemble, seed, sweeps)
        pending.append((output / relative, text))
        validation_rows.append(
            _row(
                index,
                0,
                seed,
                validation_ensemble,
                sweeps,
                f"final_checks_{label}",
                relative,
                text,
                uv_sha,
            )
        )

    seed_text = "purpose\tchain_index\tseed\tmaster_seed_draw_index\n"
    for index, seed in enumerate(PRODUCTION_SEEDS):
        seed_text += f"production\t{index % 4}\t{seed}\t{119 + index}\n"
    seed_text += f"smoke\t0\t{SMOKE_SEED}\t131\n"
    seed_text += f"timing\t0\t{TIMING_SEED}\t132\n"
    pending.extend(
        (
            (output / "manifest.tsv", pilot.manifest_text(rows)),
            (output / "validation_manifest.tsv", pilot.manifest_text(validation_rows)),
            (output / "reanalysis_manifest.tsv", _reanalysis_manifest()),
            (output / "seed_provenance.tsv", seed_text),
            (
                output / "trigger_tuning_summary_sha256.txt",
                f"{pilot.sha256_text(summary_text)}\n",
            ),
            (output / "master_seed.txt", f"{pilot.MASTER_SEED}\n"),
            (output / "klft_commit.txt", f"{pilot.KLFT_COMMIT}\n"),
            (output / "heatbath_sha256.txt", f"{pilot.HEATBATH_SHA256}\n"),
            (output / "uv_lock_sha256.txt", f"{uv_sha}\n"),
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
    print(f"wrote {len(paths)} immutable final-check files")


if __name__ == "__main__":
    main()
