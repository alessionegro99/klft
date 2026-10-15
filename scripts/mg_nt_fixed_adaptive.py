#!/usr/bin/env python3
"""Approved coarse endpoint scan with dense local plateau checks for large Nt=1 boxes.

Every window keeps ALL integer separations and the same 500 vector replicas.
Only the endpoint grid is sampled. Recommendations are earliest among tested
patches; they do not establish an exhaustive earliest asymptotic window.
The diagonal objective and full-covariance diagnostic are unchanged.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import mg_nt_fixed_ensemble as ensemble
from beta_nt_fixed.analysis import k0_r_max_values
from mg_nt_fixed_inventory import sha256, write_json
from mg_nt_fixed_parallel import parallel_scan
from mg_nt_fixed_statistics import choose_window, fit_window, usable

_CONTEXT = None
_INITIAL_MASS = None


def coarse_bounds(ns: int) -> list[tuple[int, int]]:
    """Seven allowed upper endpoints and logarithmically spaced lower endpoints."""
    highs = k0_r_max_values(ns, True)
    chosen = [highs[i] for i in np.unique(np.linspace(0, len(highs)-1, 7).astype(int))]
    lows = sorted(set([1, 2, 3] + np.rint(np.geomspace(1, ns//2-4, 24)).astype(int).tolist()))
    return [(low, high) for high in chosen for low in lows if high-low >= 4]


def dense_bounds(rows: list[dict], ns: int) -> list[tuple[int, int]]:
    """Refine up to three coarse patches, prioritizing earliest stable candidates.

    The same two-combined-SE heuristic ranks coarse neighboring patches.
    Actual acceptance/recommendation still uses adjacent integer Rmin and
    step-two Rmax values via choose_window after the dense checks.
    """
    lookup = {(r['r_min'], r['r_max']):r for r in rows}
    highs = sorted({r['r_max'] for r in rows}, reverse=True)
    lows = sorted({r['r_min'] for r in rows})
    candidates = []
    for ih, high in enumerate(highs[:-2]):
        for il, low in enumerate(lows[:-2]):
            patch = [lookup.get((lo, hi)) for lo in lows[il:il+3] for hi in highs[ih:ih+3]]
            if any(r is None or not usable(r) for r in patch):
                continue
            score = max(abs(a['mass']-b['mass'])/max(math.hypot(a['mass_error'], b['mass_error']), 1e-300)
                        for a,b in combinations(patch,2))
            candidates.append((score, low, high))
    stable = sorted((p for p in candidates if p[0] <= 2), key=lambda p:(p[1],-p[2]))
    ranked = stable + sorted(candidates, key=lambda p:(p[0],p[1],-p[2]))
    if not ranked:
        ranked = [(math.inf,r['r_min'],r['r_max']) for r in rows if usable(r)]
    centers = []
    for _, low, high in ranked:
        # Different lower endpoints probe the onset and an alternative plateau.
        if low not in [lo for lo,hi in centers]:
            centers.append((low,high))
        if len(centers) == 3:
            break
    allowed = set(k0_r_max_values(ns, True))
    bounds = set()
    for low, high in centers:
        for lo in range(max(1,low-2), low+3):
            for hi in (high, high-2, high-4):
                if hi in allowed and hi-lo >= 4:
                    bounds.add((lo,hi))
    return sorted(bounds-set(lookup), key=lambda p:(-p[1],p[0]))


def initialize(context: tuple) -> None:
    """Keep input arrays read-only and shared by forked CPU workers."""
    global _CONTEXT
    _CONTEXT = context


def fit_one(bounds: tuple[int, int]) -> tuple[dict, np.ndarray]:
    """Refit every saved replica; no resampling is performed by the scheduler."""
    r, mean, samples, covariance, ns, initial_mass = _CONTEXT
    started = time.monotonic()
    row, parameters = fit_window(r, mean, samples, covariance, ns, *bounds, initial_mass=initial_mass)
    row['fit_seconds'] = time.monotonic()-started
    return row, parameters


def adaptive_scan(r: np.ndarray, mean: np.ndarray, samples: np.ndarray, ns: int,
                  workers: int | None = None) -> tuple:
    """Coarse-to-dense grid for Ns>=512; exhaustive established scan below it."""
    if ns < 512:
        return parallel_scan(r, mean, samples, ns, workers)
    workers = workers or int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    covariance = np.cov(samples, rowvar=False, ddof=1)
    rows, parameters = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
            initargs=((r,mean,samples,covariance,ns,_INITIAL_MASS),)) as pool:
        for stage in ('coarse', 'dense'):
            bounds = coarse_bounds(ns) if stage == 'coarse' else dense_bounds(rows,ns)
            for i,(row,draws) in enumerate(pool.map(fit_one,bounds,chunksize=1)):
                row['scan_stage'] = stage
                rows.append(row); parameters.append(draws)
                if (i+1)%10==0 or i+1==len(bounds):
                    print(f'Adaptive {stage}: {i+1}/{len(bounds)} windows; {workers} CPUs',flush=True)
    order = sorted(range(len(rows)), key=lambda i:(-rows[i]['r_max'],rows[i]['r_min']))
    rows = [rows[i] for i in order]
    parameters = np.asarray(parameters)[order]
    selected, flags = choose_window(rows)
    if selected is not None:
        rows[selected]['selected'] = True
    return rows, parameters, selected, flags+['adaptive_endpoint_scan_review']


def main() -> None:
    """Use the unchanged first-stage resampler and mass-blocking checks."""
    original = ensemble.analyse
    def recorded(group: dict, output_root: Path, manifest_path: Path,
                 first_stage_only: bool = False) -> None:
        global _INITIAL_MASS
        first_stage = json.loads((output_root/group['ensemble']/'first_stage.json').read_text())
        xi = first_stage['xi_2nd']
        _INITIAL_MASS = 1/xi if xi and xi > 0 else 1/group['Ns']
        # A measured scale is an optimizer starting value, never a mass prior.
        ensemble.fit_window = partial(fit_window, initial_mass=_INITIAL_MASS)
        original(group, output_root, manifest_path, first_stage_only)
        path = output_root/group['ensemble']/'analysis.json'
        result = json.loads(path.read_text())
        result['provenance'].update(window_driver_sha256=sha256(Path(__file__)),
            allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']), initial_mass=_INITIAL_MASS)
        result['window_scan_policy'] = ('coarse endpoints then dense adjacent plateau checks; all separations retained; earliest among tested patches'
            if group['Ns'] >= 512 else 'exhaustive contiguous windows')
        write_json(path,result)
    ensemble.scan_mass_windows = adaptive_scan
    ensemble.analyse = recorded
    ensemble.main()


if __name__ == '__main__':
    main()
