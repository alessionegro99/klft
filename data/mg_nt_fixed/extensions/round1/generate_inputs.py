#!/usr/bin/env python3
"""Generate the first fixed-``m_G N_t`` bracket extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from data.mg_nt_fixed.pilot import generate_inputs as pilot  # noqa: E402


# Draws 105--112 (zero based) from the campaign's documented RNG stream.
EXTENSION_SEEDS = (
    180883465,
    1644106386,
    389869302,
    636217469,
    1633219016,
    718539186,
    516540454,
    273039637,
)
SEED_DRAW_INDICES = tuple(range(105, 113))
EXTENSIONS = (
    pilot.Ensemble("0.5", 3, "4.27", 80),
    pilot.Ensemble("0.5", 4, "5.50", 96),
)


def _failed_groups(summary: dict[str, object]) -> dict[tuple[float, int], list[str]]:
    """Return failed tuning groups and their recorded reasons."""
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


def _existing_seeds(excluded_manifest: Path) -> set[int]:
    """Return every explicit seed already recorded in a repository manifest."""
    seeds: set[int] = set()
    for manifest in (ROOT / "data").rglob("manifest.tsv"):
        if manifest.resolve() == excluded_manifest.resolve():
            continue
        with manifest.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None or "seed" not in reader.fieldnames:
                continue
            seeds.update(int(row["seed"]) for row in reader)
    return seeds


def generate(tuning_summary: Path, output: Path) -> list[Path]:
    """Write eight extension inputs, refusing all existing targets."""
    summary_text = tuning_summary.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    failures = _failed_groups(summary)
    expected = {(0.5, 3), (0.5, 4)}
    if set(failures) != expected:
        raise ValueError(f"unexpected failed tuning groups: {sorted(failures)}")
    if "log-linear interpolation has p < 0.05" not in failures[(0.5, 3)]:
        raise ValueError("the Nt=3 extension is not justified by the tuning gate")
    if "target is not bracketed" not in failures[(0.5, 4)]:
        raise ValueError("the Nt=4 extension is not justified by the tuning gate")
    if len(set(EXTENSION_SEEDS)) != len(EXTENSION_SEEDS):
        raise RuntimeError("extension seeds are not unique")
    collision = set(EXTENSION_SEEDS) & _existing_seeds(output / "manifest.tsv")
    if collision:
        raise RuntimeError(f"extension seed is already used: {min(collision)}")

    uv_lock_sha256 = pilot.sha256_file(ROOT / "uv.lock")
    pending: list[tuple[Path, str]] = []
    rows: list[dict[str, object]] = []
    for ensemble_index, ensemble in enumerate(EXTENSIONS):
        name = pilot.ensemble_name(ensemble)
        for chain_index in range(4):
            array_index = 4 * ensemble_index + chain_index
            seed = EXTENSION_SEEDS[array_index]
            relative_input = Path(name) / f"seed_{seed}" / "input.yaml"
            text = pilot.input_text(ensemble, seed)
            pending.append((output / relative_input, text))
            rows.append(
                {
                    "array_index": array_index,
                    "campaign_type": pilot.CAMPAIGN_TYPE,
                    "target_c": ensemble.target_c,
                    "run": (
                        f"mg_nt_fixed/extensions/round1/{name}/seed_{seed}"
                    ),
                    "input": relative_input.as_posix(),
                    "purpose": "failed_tuning_gate_bracket_extension",
                    "physical_Nt": ensemble.nt,
                    "stored_L2": ensemble.nt + 1,
                    "Ns": ensemble.ns,
                    "beta": ensemble.beta,
                    "chain_index": chain_index,
                    "seed": seed,
                    "sweeps": pilot.SWEEPS,
                    "measurement_interval": pilot.MEASUREMENT_INTERVAL,
                    "dirichlet_suite_measurement_interval": (
                        pilot.SUITE_MEASUREMENT_INTERVAL
                    ),
                    "input_sha256": pilot.sha256_text(text),
                    "heatbath_sha256": pilot.HEATBATH_SHA256,
                    "klft_commit": pilot.KLFT_COMMIT,
                    "uv_lock_sha256": uv_lock_sha256,
                }
            )

    seed_rows = "purpose\tchain_index\tseed\tmaster_seed_draw_index\n"
    for index, (seed, draw_index) in enumerate(
        zip(EXTENSION_SEEDS, SEED_DRAW_INDICES, strict=True)
    ):
        seed_rows += (
            f"extension_{index // 4}\t{index % 4}\t{seed}\t{draw_index}\n"
        )
    pending.extend(
        (
            (output / "manifest.tsv", pilot.manifest_text(rows)),
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
    print("wrote eight chains at two failed-gate extension points")
    print(f"files created: {len(paths)}")


if __name__ == "__main__":
    main()
