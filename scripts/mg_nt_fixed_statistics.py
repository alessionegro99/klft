"""Fixed-chain, used-measurement-weighted analysis for the fixed-mG Nt campaign.

References: Efron & Tibshirani, An Introduction to the Bootstrap (1993),
chapters 6 and 8 (resampling and standard errors); Sokal, Monte Carlo Methods
in Statistical Mechanics (1997), section 3 (autocorrelation window).
The periodic K0 model is Nebbia's thesis, Eq. (4.14), in docs/.

Every operation makes exactly 500 draws. Independent chain identities stay
fixed; one draw resamples complete block vectors within each chain. Pooled
weights count the measurements used in complete blocks. Shared fixed-beta
analysis defaults are deliberately unchanged.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from beta_nt_fixed.analysis import (
    _uncorrelated_k0_fit_with_errors, finite_volume_k0, k0_r_max_values,
    stable_three_length_groups,
)
from dirichlet_analysis import block_values

Array = NDArray[np.float64]
NBOOT = 500
BLOCK_LENGTHS = (5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 240, 280, 320)


def operation_seed(key: str, operation: str) -> int:
    """Derive a deterministic, independently named stream from a master seed."""
    return int.from_bytes(hashlib.sha256(f'20260724:{key}:{operation}'.encode()).digest()[:8], 'little')


def pooled_bootstrap(blocks: Sequence[Array], seed: int, nboot: int = NBOOT,
                     measurements_per_block: Sequence[int] | None = None) -> tuple[Array, Array, Array]:
    """Return pooled mean, 500 joint replicas and fixed chain weights.

    Multiplicity matrices are an exact implementation of independent uniform
    block-index draws; they avoid materializing a replica x block x R array.
    No chain is resampled as an outer statistical unit.
    """
    if nboot != 500:
        raise ValueError('every bootstrap operation requires exactly 500 samples')
    if not blocks or any(len(b) < 2 or b.ndim != 2 for b in blocks):
        raise ValueError('each chain needs at least two complete vector blocks')
    if len({b.shape[1] for b in blocks}) != 1:
        raise ValueError('different vector dimensions')
    sizes = np.ones(len(blocks)) if measurements_per_block is None else np.asarray(measurements_per_block)
    if sizes.shape != (len(blocks),) or np.any(sizes <= 0):
        raise ValueError('invalid measurement counts per block')
    counts = np.asarray([len(b) for b in blocks])
    weights = counts * sizes / np.dot(counts, sizes)
    central = np.sum([w * b.mean(axis=0) for w, b in zip(weights, blocks)], axis=0)
    samples = np.zeros((NBOOT, blocks[0].shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for weight, chain in zip(weights, blocks):
        n = len(chain)
        for start in range(0, NBOOT, 25):
            indices = rng.integers(0, n, size=(25, n))
            multiplicities = np.stack([np.bincount(row, minlength=n) for row in indices]).astype(np.float64)
            samples[start:start+25] += (weight / n) * (multiplicities @ chain)
    return central, samples, weights


def xi_second(vectors: Array, ns: int) -> Array:
    """Ensemble structure-factor length; unresolved replicas stay NaN."""
    ratio = vectors[..., 2] / vectors[..., 3] - 1.0
    with np.errstate(invalid='ignore'):
        return np.where(vectors[..., 3] > 0, np.sqrt(np.maximum(ratio, 0)), np.nan) / (2*np.sin(np.pi/ns)) * np.where(ratio > 0, 1.0, np.nan)


def bootstrap_error(samples: Array) -> float | None:
    """Return a finite conditional standard error, retaining failed slots."""
    finite = samples[np.isfinite(samples)]
    return float(np.std(finite, ddof=1)) if len(finite) >= 2 else None


def error_uncertainty(blocks: Sequence[Array], weights: Array, error: Array) -> Array:
    """Approximate uncertainty of a bootstrap SE for plateau diagnostics.

    Welch-Satterthwaite effective degrees of freedom combine independent
    chain variance estimates. Add the normal-reference Monte Carlo variance
    of an SE estimated from 500 replicas. This approximation only chooses
    diagnostic plateau ranges; quoted observable errors remain bootstrap SDs.
    """
    contributions = np.asarray([w*w*np.var(b, axis=0, ddof=1)/len(b) for w, b in zip(weights, blocks)])
    variance = contributions.sum(axis=0)
    denominator = np.sum([v*v/(len(b)-1) for v, b in zip(contributions, blocks)], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_df = np.divide(denominator, variance*variance, out=np.zeros_like(variance), where=variance > 0)
    return error * np.sqrt(0.5*inv_df + 1/(2*(NBOOT-1)))


def blocking_scan(series: Sequence[Array], labels: list[str], ns: int,
                  minimum_length: int, key: str) -> dict:
    """Scan all post-cut histories; each vector represents five sweeps."""
    lengths = [b for b in BLOCK_LENGTHS if min(len(s) for s in series)//(b//5) >= 2]
    if minimum_length not in lengths and min(len(s) for s in series)//(minimum_length//5) >= 2:
        lengths = sorted(lengths + [minimum_length])
    errors, uncertainties, estimates, counts, draws, blocks_all, seeds = [], [], [], [], [], [], []
    for length in lengths:
        blocks = [block_values(s, length//5) for s in series]
        seed = operation_seed(key, f'blocking:{length}')
        mean, samples, weights = pooled_bootstrap(blocks, seed)
        xi = xi_second(samples, ns)
        extra = np.column_stack((xi, ns/xi))
        err = np.std(samples, axis=0, ddof=1)
        extra_err = np.asarray([bootstrap_error(v) or np.nan for v in extra.T])
        # Conservative minimum-chain df for nonlinear diagnostics.
        extra_unc = extra_err*np.sqrt(1/(2*(min(map(len, blocks))-1)) + 1/(2*(NBOOT-1)))
        errors.append(np.r_[err, extra_err])
        uncertainties.append(np.r_[error_uncertainty(blocks, weights, err), extra_unc])
        x0 = float(xi_second(mean, ns))
        estimates.append(np.r_[mean, x0, ns/x0])
        counts.append([len(b) for b in blocks])
        draws.append(samples)
        blocks_all.append(blocks)
        seeds.append(seed)
    if not lengths:
        raise ValueError('fewer than two blocks in a chain')
    per_observable, common = stable_three_length_groups(lengths, np.asarray(errors), np.asarray(uncertainties), minimum_length)
    reliable = [g for g in common if min(counts[lengths.index(g[-1])]) >= 30]
    if reliable:
        selected = reliable[0][-1]
    else:
        eligible = [b for b, n in zip(lengths, counts) if b >= minimum_length and min(n) >= 30]
        if not eligible:
            eligible = [b for b in lengths if b >= minimum_length]
        if not eligible:
            raise ValueError('insufficient history for even two autocorrelation-length blocks')
        selected = eligible[-1]
    return dict(lengths=lengths, labels=labels+['xi_2nd', 'Ns_over_xi_2nd'],
                errors=np.asarray(errors), uncertainties=np.asarray(uncertainties),
                estimates=np.asarray(estimates), counts=counts, samples=draws, blocks=blocks_all,
                seeds=seeds, selected=selected, common_groups=reliable,
                observable_groups=per_observable, minimum_length=minimum_length)


def fit_window(r: Array, mean: Array, samples: Array, covariance: Array,
               ns: int, low: int, high: int,
               initial_mass: float | None = None) -> tuple[dict, Array]:
    """Diagonal central/refit objective, full-covariance descriptive residual.

    The shared fit_utils.boot_fit cannot retain individual failed refits,
    constrain positive K0 parameters or report full-covariance diagnostics;
    the existing project-specific K0 optimizer supplies this treatment.
    No analytic p-value is calculated: this objective does not minimize the
    reported full-covariance statistic. Covariance is never regularized.
    """
    mask = (r >= low) & (r <= high)
    x, y, cov = r[mask], mean[mask], covariance[np.ix_(mask, mask)]
    error = np.sqrt(np.maximum(np.diag(cov), 0))
    row = dict(r_min=low, r_max=high, n_points=int(mask.sum()), nominal_dof=int(mask.sum())-2,
               amplitude=None, mass=None, amplitude_error=None, mass_error=None,
               chi2_full=None, chi2_full_per_dof=None, covariance_condition=None,
               bootstrap_successes=0, bootstrap_samples=NBOOT, flags=[], failure=None)
    parameters = np.full((NBOOT, 2), np.nan)
    try:
        initial = None
        if initial_mass is not None:
            normalization = finite_volume_k0(x[:1], 1., initial_mass, ns)[0]
            initial = np.asarray([max(abs(y[0])/normalization, 1e-8), initial_mass])
        central = _uncorrelated_k0_fit_with_errors(x, y, error, ns, initial)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        row['failure'] = str(exc)
        row['flags'] = ['undefined_central_fit']
        return row, parameters
    row.update(amplitude=central['amplitude'], mass=central['mass'])
    try:
        # Centering 500 replica vectors bounds covariance rank by 499.
        # Roundoff can otherwise let Cholesky spuriously succeed here.
        if len(x) >= NBOOT:
            raise linalg.LinAlgError('fitted points exceed bootstrap covariance rank')
        condition = float(np.linalg.cond(cov))
        row['covariance_condition'] = condition if np.isfinite(condition) else None
        factor = linalg.cholesky(cov, lower=True)
        residual = y - finite_volume_k0(x, row['amplitude'], row['mass'], ns)
        whitened = linalg.solve_triangular(factor, residual, lower=True)
        chi2 = float(whitened @ whitened)
        row.update(chi2_full=chi2, chi2_full_per_dof=chi2/row['nominal_dof'])
        if condition > 1e12:
            row['flags'].append('ill_conditioned_covariance')
    except (ValueError, linalg.LinAlgError):
        row['flags'].append('singular_full_covariance')
    initial = np.asarray([row['amplitude'], row['mass']])
    for i, replica in enumerate(samples[:, mask]):
        try:
            fit = _uncorrelated_k0_fit_with_errors(x, replica, error, ns, initial)
            parameters[i] = fit['amplitude'], fit['mass']
        except (ValueError, RuntimeError, FloatingPointError):
            pass
    row['bootstrap_successes'] = int(np.all(np.isfinite(parameters), axis=1).sum())
    row['amplitude_error'] = bootstrap_error(parameters[:, 0])
    row['mass_error'] = bootstrap_error(parameters[:, 1])
    if row['bootstrap_successes'] < 450:
        row['flags'].append('bootstrap_success_below_90_percent')
    if row['mass_error'] is None:
        row['flags'].append('undefined_mass_error')
    elif row['mass_error']/row['mass'] > .25:
        row['flags'].append('relative_mass_error_above_25_percent')
    return row, parameters


def usable(row: dict) -> bool:
    """Only an undefined numerical estimate is unavailable for review."""
    return (row.get('mass') is not None and row.get('mass_error') is not None
            and row['mass'] > 0 and np.isfinite(row['mass_error']))


def choose_window(rows: list[dict]) -> tuple[int | None, list[str]]:
    """Earliest stable 3x3 mass patch; otherwise smallest pairwise drift.

    Adjacent windows are correlated. The two-combined-SE rule is a visual
    stability heuristic, not an independent-test confidence level.
    """
    lookup = {(r['r_min'], r['r_max']): i for i, r in enumerate(rows)}
    patches = []
    for (low, high), index in lookup.items():
        ids = [lookup.get((lo, hi)) for lo in range(low, low+3) for hi in (high, high-2, high-4)]
        if any(i is None or not usable(rows[i]) for i in ids):
            continue
        scores = [abs(rows[i]['mass']-rows[j]['mass']) / max(math.hypot(rows[i]['mass_error'], rows[j]['mass_error']), 1e-300)
                  for i, j in combinations(ids, 2)]
        patches.append((max(scores), low, -high, index, ids))
    for row in rows:
        row.update(stable_three_by_three_patch=False, selected=False)
    stable = [p for p in patches if p[0] <= 2]
    for _, _, _, _, ids in stable:
        for i in ids:
            rows[i]['stable_three_by_three_patch'] = True
    if stable:
        chosen = min(stable, key=lambda p: (p[1], p[2]))
        rows[chosen[3]]['patch_max_pairwise_z'] = chosen[0]
        return chosen[3], []
    if patches:
        chosen = min(patches, key=lambda p: (p[0], p[1], p[2]))
        rows[chosen[3]]['patch_max_pairwise_z'] = chosen[0]
        return chosen[3], ['no_stable_three_by_three_patch']
    candidates = [i for i, row in enumerate(rows) if usable(row)]
    if candidates:
        return min(candidates, key=lambda i: (rows[i]['r_min'], -rows[i]['r_max'])), ['no_complete_three_by_three_patch']
    return None, ['mass_unavailable']


def scan_mass_windows(r: Array, mean: Array, samples: Array, ns: int) -> tuple[list[dict], Array, int | None, list[str]]:
    """Fit every contiguous permitted window using the same 500 joint draws."""
    covariance = np.cov(samples, rowvar=False, ddof=1)
    rows, fits = [], []
    for high in k0_r_max_values(ns, True):
        for low in range(1, high-3):
            row, parameters = fit_window(r, mean, samples, covariance, ns, low, high)
            rows.append(row)
            fits.append(parameters)
        print(f'K0 Rmax={high}: {len(rows)} windows completed', flush=True)
    selected, flags = choose_window(rows)
    if selected is not None:
        rows[selected]['selected'] = True
    return rows, np.asarray(fits), selected, flags


def review_window(rows: list[dict], suggested: int | None, review: dict | None) -> int | None:
    """Validate a manual override without dropping its numerical flags."""
    if not review or not review.get('r_min'):
        return suggested
    candidates = [i for i, row in enumerate(rows) if row['r_min'] == int(review['r_min'])
                  and row['r_max'] == int(review['r_max'])]
    if len(candidates) != 1 or not usable(rows[candidates[0]]):
        raise ValueError('manual window must identify one numerically usable saved fit')
    return candidates[0]
