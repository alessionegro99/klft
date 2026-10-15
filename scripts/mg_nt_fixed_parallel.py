#!/usr/bin/env python3
"""Parallelize independent mass windows without changing bootstrap draws.

The serial Ns128 scan processed 60 of 550 windows in roughly 13 minutes on
qbig. Each window only reads the same 500 ensemble vectors. Process-level
parallelism is bounded by the allocated Slurm CPU count; BLAS stays serial.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

import mg_nt_fixed_ensemble as ensemble
from beta_nt_fixed.analysis import k0_r_max_values
from mg_nt_fixed_inventory import sha256, write_json
from mg_nt_fixed_statistics import choose_window, fit_window

_CONTEXT = None


def initialize(context: tuple) -> None:
    """Install read-only fit inputs in each allocated worker."""
    global _CONTEXT
    _CONTEXT=context


def fit_one(bounds: tuple[int,int]) -> tuple[dict,np.ndarray]:
    """Return one deterministic window fit; no RNG operation occurs here."""
    low,high=bounds
    r,mean,samples,covariance,ns=_CONTEXT
    return fit_window(r,mean,samples,covariance,ns,low,high)


def parallel_scan(r: np.ndarray,mean: np.ndarray,samples: np.ndarray,ns: int,
                  workers: int | None = None) -> tuple:
    """Match serial window order, parameter draws, flags and selection."""
    if workers is None:
        workers=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
    covariance=np.cov(samples,rowvar=False,ddof=1)
    bounds=[(low,high) for high in k0_r_max_values(ns,True) for low in range(1,high-3)]
    rows,parameters=[],[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,
                             initargs=((r,mean,samples,covariance,ns),)) as pool:
        for i,(row,draws) in enumerate(pool.map(fit_one,bounds,chunksize=1)):
            rows.append(row);parameters.append(draws)
            if (i+1)%25==0 or i+1==len(bounds):
                print(f'Parallel K0: {i+1}/{len(bounds)} windows complete ({workers} allocated CPUs)',flush=True)
    selected,flags=choose_window(rows)
    if selected is not None:
        rows[selected]['selected']=True
    return rows,np.asarray(parameters),selected,flags


def main() -> None:
    """Use the unchanged ensemble workflow with a parallel window scheduler."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('parallel analysis requires a qbig compute allocation')
    original=ensemble.analyse
    def recorded(group: dict, output_root: Path, manifest_path: Path,
                 first_stage_only: bool = False) -> None:
        original(group,output_root,manifest_path,first_stage_only)
        path=output_root/group['ensemble']/('first_stage.json' if first_stage_only else 'analysis.json')
        result=json.loads(path.read_text())
        result['provenance']['parallel_window_driver_sha256']=sha256(Path(__file__))
        result['provenance']['allocated_cpus']=int(os.environ['SLURM_CPUS_PER_TASK'])
        write_json(path,result)
    ensemble.scan_mass_windows=parallel_scan
    ensemble.analyse=recorded
    ensemble.main()


if __name__=='__main__':
    main()
