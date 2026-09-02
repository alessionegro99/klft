#!/usr/bin/env python3
"""Extract static-potential plateaus from blocked orbifold Wilson loops.

The adjacent-time estimator follows the Wilson-loop spectral form used in
Donnellan et al., Nucl. Phys. B 849 (2011) 45, arXiv:1012.3037:

    a_t V_eff(R,T+1/2) = log[W(R,T) / W(R,T+1)].

Input must be the vector hierarchical-bootstrap archive written by
``orbifold_hmc_stats.py`` for either HMC or heatbath chains. Plateau constants
use its complete covariance.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def parse_plateau(text: str) -> tuple[int, int, int]:
    """Parse ``R:Tmin-Tmax`` for adjacent-time effective potentials."""
    match = re.fullmatch(r"([1-9]\d*):([1-9]\d*)-([1-9]\d*)", text)
    if match is None:
        raise argparse.ArgumentTypeError("plateau must have form R:Tmin-Tmax")
    r, t_min, t_max = map(int, match.groups())
    if t_max <= t_min:
        raise argparse.ArgumentTypeError("a plateau needs at least two time ratios")
    return r, t_min, t_max


def fit_plateau(
    mean: FloatArray,
    bootstrap: FloatArray,
    index: dict[tuple[int, int], int],
    plateau: tuple[int, int, int],
) -> tuple[float, FloatArray, float, int, FloatArray]:
    """Fit one correlated constant and return value, samples, Q, dof, weights."""
    r, t_min, t_max = plateau
    try:
        columns = [
            (index[(r, t)], index[(r, t + 1)])
            for t in range(t_min, t_max + 1)
        ]
    except KeyError as error:
        raise ValueError(
            f"missing Wilson loop {error.args[0]} for plateau "
            f"{r}:{t_min}-{t_max}"
        ) from error
    if any(mean[i] <= 0.0 or mean[j] <= 0.0 for i, j in columns):
        raise ValueError(
            f"non-positive central Wilson loop in plateau {r}:{t_min}-{t_max}"
        )
    if any(
        np.any((bootstrap[:, i] <= 0.0) | (bootstrap[:, j] <= 0.0))
        for i, j in columns
    ):
        raise ValueError(
            f"non-positive bootstrap Wilson loop in plateau {r}:{t_min}-{t_max}"
        )

    values = np.array([np.log(mean[i] / mean[j]) for i, j in columns])
    samples = np.column_stack(
        [np.log(bootstrap[:, i] / bootstrap[:, j]) for i, j in columns]
    )
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
    ones = np.ones(len(columns), dtype=np.float64)
    solved = np.linalg.solve(covariance, ones)
    weights = solved / np.sum(solved)
    fitted = float(weights @ values)
    fitted_samples = samples @ weights
    residual = values - fitted
    q = float(residual @ np.linalg.solve(covariance, residual))
    return fitted, fitted_samples, q, len(columns) - 1, weights


def analyze(
    source: Path,
    output: Path,
    a_s: float,
    a_t: float,
    plateaus: list[tuple[int, int, int]],
) -> None:
    """Write effective ratios and selected complete-covariance plateaus."""
    with np.load(source) as data:
        pairs = np.asarray(data["pairs"], dtype=np.int64)
        mean = np.asarray(data["mean"], dtype=np.float64)
        bootstrap = np.asarray(data["bootstrap_means"], dtype=np.float64)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or mean.shape != (len(pairs),)
        or bootstrap.ndim != 2
        or bootstrap.shape[0] < 2
        or bootstrap.shape[1] != len(pairs)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(bootstrap))
    ):
        raise ValueError("invalid orbifold statistics archive")
    index = {tuple(pair): i for i, pair in enumerate(pairs)}
    if len(index) != len(pairs):
        raise ValueError("duplicate Wilson-loop pair")
    if len({r for r, _, _ in plateaus}) != len(plateaus):
        raise ValueError("specify at most one plateau per spatial separation")

    effective_rows = []
    for r, t in sorted(index):
        if (r, t + 1) not in index:
            continue
        i, j = index[(r, t)], index[(r, t + 1)]
        valid = (bootstrap[:, i] > 0.0) & (bootstrap[:, j] > 0.0)
        fraction = float(np.mean(valid))
        value = (
            float(np.log(mean[i] / mean[j]))
            if mean[i] > 0.0 and mean[j] > 0.0
            else np.nan
        )
        error = (
            float(np.std(np.log(bootstrap[:, i] / bootstrap[:, j]), ddof=1))
            if np.all(valid)
            else np.nan
        )
        effective_rows.append((r, t, t + 0.5, r * a_s, value, error, fraction))

    potential_rows = []
    potential_samples = []
    summaries = []
    for plateau in plateaus:
        value, samples, q, dof, weights = fit_plateau(mean, bootstrap, index, plateau)
        r, t_min, t_max = plateau
        error = float(np.std(samples, ddof=1))
        potential_rows.append(
            (
                r,
                r * a_s,
                t_min,
                t_max,
                value,
                error,
                value * a_s / a_t,
                error * a_s / a_t,
                value / a_t,
                error / a_t,
                q,
                dof,
            )
        )
        potential_samples.append(samples)
        summaries.append(
            f"plateau {r}:{t_min}-{t_max} weights "
            + " ".join(f"{weight:.17g}" for weight in weights)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    effective_output = output.with_suffix(".effective.tsv")
    np.savetxt(
        effective_output,
        np.asarray(effective_rows),
        header="R T T_midpoint r a_t_Veff hierarchical_bootstrap_se bootstrap_positive_fraction",
        fmt=["%d", "%d"] + ["%.17g"] * 5,
    )
    np.savetxt(
        output,
        np.asarray(potential_rows),
        header="R r T_min T_max a_t_V a_t_V_se a_s_V a_s_V_se V V_se Q dof",
        fmt=["%d", "%.17g", "%d", "%d"] + ["%.17g"] * 7 + ["%d"],
    )
    fitted_bootstrap = np.column_stack(potential_samples)
    covariance = np.atleast_2d(np.cov(fitted_bootstrap, rowvar=False, ddof=1))
    np.savez_compressed(
        output.with_suffix(".npz"),
        R=np.asarray([row[0] for row in potential_rows], dtype=np.int64),
        r=np.asarray([row[1] for row in potential_rows]),
        T_min=np.asarray([row[2] for row in potential_rows], dtype=np.int64),
        T_max=np.asarray([row[3] for row in potential_rows], dtype=np.int64),
        a_t_V=np.asarray([row[4] for row in potential_rows]),
        a_t_V_bootstrap=fitted_bootstrap,
        a_t_V_covariance=covariance,
        a_s=np.float64(a_s),
        a_t=np.float64(a_t),
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output.with_suffix(".summary.txt").write_text(
        "\n".join(
            [
                f"source {source}",
                f"source_sha256 {source_hash}",
                f"a_s {a_s:.17g}",
                f"a_t {a_t:.17g}",
                f"bootstrap_samples {len(bootstrap)}",
                "estimator a_t*V_eff(R,T+1/2)=log(W(R,T)/W(R,T+1))",
                "fit complete-covariance generalized least-squares constant",
                *summaries,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def self_test() -> None:
    """Recover a known exponential and reject an undefined log ratio."""
    rng = np.random.default_rng(10123017)
    pairs = np.array([(1, t) for t in range(1, 6)], dtype=np.int64)
    times = pairs[:, 1]
    potentials = 0.7 + rng.normal(0.0, 0.01, size=(4000, 1))
    amplitudes = np.exp(rng.normal(0.0, 0.02, size=(4000, 1)))
    noise = rng.normal(0.0, 0.002, size=(4000, len(times)))
    bootstrap = amplitudes * np.exp(-potentials * times + noise)
    mean = np.mean(bootstrap, axis=0)
    index = {tuple(pair): i for i, pair in enumerate(pairs)}
    value, samples, q, dof, _ = fit_plateau(mean, bootstrap, index, (1, 1, 3))
    assert abs(value - 0.7) < 5.0 * np.std(samples, ddof=1)
    assert np.isfinite(q) and dof == 2
    bootstrap[0, 2] = -1.0
    try:
        fit_plateau(mean, bootstrap, index, (1, 1, 3))
    except ValueError:
        return
    raise AssertionError("non-positive bootstrap loop was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a-s", type=float)
    parser.add_argument("--a-t", type=float)
    parser.add_argument("--plateau", type=parse_plateau, action="append")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if (
        args.source is None
        or args.output is None
        or args.a_s is None
        or args.a_t is None
        or not args.plateau
    ):
        parser.error(
            "source, --output, --a-s, --a-t, and at least one --plateau "
            "are required"
        )
    if args.a_s <= 0.0 or args.a_t <= 0.0:
        parser.error("lattice spacings must be positive")
    analyze(args.source, args.output, args.a_s, args.a_t, args.plateau)


if __name__ == "__main__":
    main()
