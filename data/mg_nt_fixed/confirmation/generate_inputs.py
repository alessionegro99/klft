#!/usr/bin/env python3
"""Generate the tuned ``Nt=4,c=1`` confirmation ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.mg_nt_fixed.pilot import generate_inputs as pilot  # noqa: E402


def selected_beta(summary: dict[str, object]) -> str:
    """Return the accepted tuned coupling at ``Nt=4,c=1``."""
    if not bool(summary["all_nine_tunings_accepted"]):
        raise ValueError("all nine pilot tunings must pass before confirmation")
    value = summary.get("confirmation_Nt4_c1_beta")
    if value is None:
        raise ValueError("the tuning summary has no Nt=4,c=1 coupling")
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def generate(tuning_summary: Path, output: Path) -> list[Path]:
    """Write four confirmation inputs and refuse all existing targets."""
    summary_text = tuning_summary.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    beta = selected_beta(summary)
    ensemble = pilot.Ensemble("1.0", 4, beta, 48)
    uv_lock_sha256 = pilot.sha256_file(ROOT / "uv.lock")
    pending: list[tuple[Path, str]] = []
    rows: list[dict[str, object]] = []
    name = pilot.ensemble_name(ensemble)
    for chain_index, seed in enumerate(pilot.CONFIRMATION_SEEDS):
        relative_input = Path("confirmation") / name / f"seed_{seed}" / "input.yaml"
        text = pilot.input_text(ensemble, seed)
        pending.append((output / relative_input, text))
        rows.append(
            {
                "array_index": chain_index,
                "campaign_type": pilot.CAMPAIGN_TYPE,
                "target_c": ensemble.target_c,
                "run": f"mg_nt_fixed/confirmation/{name}/seed_{seed}",
                "input": relative_input.as_posix(),
                "purpose": "Nt4_c1_tuning_confirmation",
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
    pending.extend(
        (
            (output / "confirmation_manifest.tsv", pilot.manifest_text(rows)),
            (
                output / "confirmation_tuning_summary_sha256.txt",
                f"{pilot.sha256_text(summary_text)}\n",
            ),
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
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    paths = generate(args.tuning_summary, args.output)
    print("wrote four tuned confirmation inputs")
    print(f"files created: {len(paths)}")


if __name__ == "__main__":
    main()
