#!/usr/bin/env python3
"""Refresh ensemble figures from saved estimates; no new fits or bootstrap draws."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from beta_nt_fixed.analysis import load_chain
from mg_nt_fixed_canonical_plots import plot_ensemble, plot_first_stage
from mg_nt_fixed_inventory import ROOT, load_campaign_manifest, sha256, write_json


def saved_scan(result: dict, rows: list[dict]) -> dict:
    """Recover the recorded error curves and block counts without resampling."""
    labels = result['labels']+['xi_2nd', 'Ns_over_xi_2nd']
    lengths = sorted({int(row['block_length_sweeps']) for row in rows})
    lookup = {(int(row['block_length_sweeps']), row['observable']):row for row in rows}
    def values(field: str) -> np.ndarray:
        return np.asarray([[float(lookup[length,label][field] or 'nan')
                            for label in labels] for length in lengths], dtype=np.float64)
    counts = [[chain['retained_sweeps']//length for chain in result['chains']]
              for length in lengths]
    for length, count in zip(lengths, counts):
        row = lookup[length, labels[0]]
        if (min(count), max(count)) != (int(row['min_blocks']), int(row['max_blocks'])):
            raise ValueError('saved block counts disagree with the chain inventory')
    minimum = 5*math.ceil((math.floor(2*max(c['tau_int_max_sweeps'] for c in result['chains']))+1)/5)
    return dict(labels=labels, lengths=lengths, errors=values('bootstrap_error'),
                uncertainties=values('se_uncertainty_approx'), counts=counts,
                selected=result['block_length_sweeps'], minimum_length=minimum)


def replot_ensemble(key: str) -> str:
    """Read histories only for display traces; retain every recorded cut and fit."""
    leaf = ROOT/'analysis/mg_nt_fixed'/key
    result = json.loads((leaf/'analysis.json').read_text())
    with (leaf/'blocking_scan.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    scan = saved_scan(result, rows)
    # Load one chain at a time; no history is pooled or rethermalized here.
    raw = (load_chain(Path(chain['raw_path']), maximum_sweep=chain['sweeps'])
           for chain in result['chains'])
    plot_first_stage(leaf, result, raw, scan)
    with np.load(leaf/'bootstrap.npz') as boot:
        mass_rows = [r for r in rows if r['observable']=='mass']
        plot_ensemble(leaf, result, boot, mass_rows, scan)
    write_json(leaf/'plots/plot_provenance.json', dict(
        analysis_sha256=sha256(leaf/'analysis.json'),
        bootstrap_sha256=sha256(leaf/'bootstrap.npz'),
        source_sha256={name:sha256(ROOT/name) for name in
                      ('scripts/mg_nt_fixed_replot.py','scripts/mg_nt_fixed_canonical_plots.py')},
        job_id=os.environ.get('SLURM_JOB_ID'), new_bootstrap_operations=0,
        displayed_titles=False, displayed_footnotes=False))
    return key


def main() -> None:
    """Render the existing campaign on its allocated qbig CPUs."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('history plotting requires a qbig compute allocation')
    keys = [g['ensemble'] for g in load_campaign_manifest()['ensembles']]
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, key in enumerate(pool.map(replot_ensemble, keys), 1):
            print(f'Replotted {i}/{len(keys)}: {key}; no new fits or bootstrap draws', flush=True)


if __name__=='__main__':
    main()
