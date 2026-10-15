"""Analyse one canonical mg_nt ensemble on a qbig compute node."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

from beta_nt_fixed.analysis import (candidate_thermalization_cut, load_chain,
                                    profile_label, stable_three_length_groups)
from dirichlet_analysis import block_values, integrated_autocorrelation_time, require_matching_grids
from mg_nt_fixed_inventory import ROOT, sha256, write_csv, write_json
from mg_nt_fixed_statistics import (NBOOT, blocking_scan, bootstrap_error, fit_window,
                                    operation_seed, pooled_bootstrap, scan_mass_windows, xi_second)


def clean(value: object) -> object:
    """Encode undefined floating-point diagnostics as JSON null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyse(group: dict, output_root: Path, manifest_path: Path,
            first_stage_only: bool = False) -> None:
    """Verify raw provenance, reanalyse every chain and save compact products."""
    key = group['ensemble']
    output = output_root / key
    if (output / 'analysis.json').exists():
        raise FileExistsError(f'completed result already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    raw = [load_chain(Path(row['raw_path'])) for row in group['chains']]
    require_matching_grids([c.correlator for c in raw])
    require_matching_grids([c.profile for c in raw])
    r = raw[0].correlator.keys[:, 0]
    nr = len(r)
    labels = ['Tr_G_real', 'Tr_G_imag', 'Ctilde_0', 'Ctilde_pmin'] + [f'C_G_R{int(x)}' for x in r]
    labels += [profile_label(k) for k in raw[0].profile.keys]
    chains, taus, series, provenance = [], [], [], {}
    for chain, record in zip(raw, group['chains'], strict=True):
        if sha256(chain.directory / 'input.yaml') != record['input_sha256']:
            raise ValueError(f'input hash mismatch: {chain.directory}')
        if len(chain.summary.steps) != record['sweeps'] or int(chain.summary.steps[-1]) != record['sweeps']:
            raise ValueError(f'incomplete or unexpected history: {chain.directory}')
        for name in ('input.yaml', 'holonomy_summary.out', 'holonomy_correlator.out', 'plaquette_profile.out'):
            provenance[str(chain.directory / name)] = sha256(chain.directory / name)
            expected = record.get('output_sha256', {}).get(name)
            if expected and provenance[str(chain.directory / name)] != expected:
                raise ValueError(f'completed-output hash mismatch: {chain.directory / name}')
        cut, diagnostic = candidate_thermalization_cut(chain, 500)
        primary = np.column_stack((chain.summary.values[chain.summary.steps > cut, 0, :],
                                   chain.correlator.values[chain.correlator.steps > cut, :, 0]))
        profile = chain.profile.values[chain.profile.steps > cut, :, 0]
        elementary = np.column_stack((block_values(primary, 5), profile))
        series.append(elementary)
        chain_taus = []
        for i, label in enumerate(labels):
            values, spacing = (primary[:, i], 1) if i < 4+nr else (profile[:, i-4-nr], 5)
            tau = spacing * integrated_autocorrelation_time(values)
            chain_taus.append(tau)
            taus.append(dict(ensemble=key, seed=record['seed'], observable=label,
                             measurement_interval_sweeps=spacing, tau_int_sweeps=tau,
                             thermalization_cut_sweeps=cut, measurements=len(values)))
        chains.append(dict(ensemble=key, seed=record['seed'], raw_path=record['raw_path'],
                           source_campaign=record['source_campaign'], sweeps=record['sweeps'],
                           thermalization_cut_sweeps=cut, retained_sweeps=len(primary),
                           tau_int_R1_sweeps=chain_taus[5], tau_int_max_sweeps=max(chain_taus),
                           flags=record['flags'].copy(), mser5=diagnostic))
    minimum = 5*math.ceil((math.floor(2*max(c['tau_int_max_sweeps'] for c in chains))+1)/5)
    scan = blocking_scan(series, labels, group['Ns'], minimum, key)
    bindex = scan['lengths'].index(scan['selected'])
    blocks = scan['blocks'][bindex]
    samples = scan['samples'][bindex]
    mean = scan['estimates'][bindex, :len(labels)]
    weights = np.asarray([len(b) for b in blocks], dtype=np.float64)
    weights /= weights.sum()
    flags = [] if scan['common_groups'] else ['observable_blocking_plateau_unresolved']
    chain_means, chain_errors, chain_boots = [], [], []
    for chain, chain_blocks, weight in zip(chains, blocks, weights):
        chain.update(block_length_sweeps=scan['selected'], n_complete_blocks=len(chain_blocks),
                     used_sweeps=len(chain_blocks)*scan['selected'], pooled_weight=float(weight),
                     discarded_partial_sweeps=chain['retained_sweeps']-len(chain_blocks)*scan['selected'])
        if len(chain_blocks) < 50:
            chain['flags'].append('below_50_blocks')
        if len(chain_blocks) < 30:
            chain['flags'].append('below_30_blocks')
        cm, cb, _ = pooled_bootstrap([chain_blocks], operation_seed(key, f'chain:{chain["seed"]}'))
        chain_means.append(cm)
        chain_errors.append(np.std(cb, axis=0, ddof=1))
        chain_boots.append(cb)
        middle = len(chain_blocks)//2
        split = []
        for half, vector in enumerate((chain_blocks[:middle], chain_blocks[middle:])):
            if len(vector) >= 2:
                av, bs, _ = pooled_bootstrap([vector], operation_seed(key, f'split:{chain["seed"]}:{half}'))
                split.append((av, np.std(bs, axis=0, ddof=1)))
        if len(split) == 2:
            den = np.hypot(split[0][1], split[1][1])
            z = np.divide(abs(split[0][0]-split[1][0]), den, out=np.zeros_like(den), where=den > 0)
            j = int(np.argmax(z))
            chain.update(split_max_z=float(z[j]), split_max_observable=labels[j])
            if z[j] > 3:
                chain['flags'].append('split_history_review')
    pairwise = []
    for i in range(len(chains)):
        for j in range(i+1, len(chains)):
            den = np.hypot(chain_errors[i], chain_errors[j])
            z = np.divide(abs(chain_means[i]-chain_means[j]), den, out=np.zeros_like(den), where=den > 0)
            k = int(np.argmax(z))
            pairwise.append(dict(seed_a=chains[i]['seed'], seed_b=chains[j]['seed'], observable=labels[k], max_z=float(z[k])))
    if any(p['max_z'] > 3 for p in pairwise):
        flags.append('between_chain_review')
    flags += sorted({flag for chain in chains for flag in chain['flags']})
    covariance = np.cov(samples[:, 4:4+nr], rowvar=False, ddof=1)
    preliminary = dict(**{k: group[k] for k in ('ensemble', 'target_c', 'physical_Nt', 'stored_L2', 'Ns', 'beta')},
                       chains=chains, labels=labels, R=r.tolist(), bootstrap_samples=500,
                       block_length_sweeps=scan['selected'])
    preliminary.update(reference_c=group.get('reference_c', group['target_c']),
                       fixed_c_continuum_input=group.get('fixed_c_continuum_input', True))
    write_json(output / 'first_stage.json', clean(preliminary))
    from mg_nt_fixed_canonical_plots import plot_first_stage
    plot_first_stage(output, preliminary, raw, scan)
    print(f'{key}: verified {len(chains)} chains; B={scan["selected"]}; starting all-window fits', flush=True)
    if first_stage_only:
        rows, fitted, selected, window_flags = [], np.empty((0, NBOOT, 2)), None, ['mass_scan_pending']
    else:
        rows, fitted, selected, window_flags = scan_mass_windows(r, mean[4:4+nr], samples[:, 4:4+nr], group['Ns'])
    flags += window_flags
    selected_fit = rows[selected] if selected is not None else None
    mass_scan, mass_draws = [], []
    if selected is not None:
        flags += selected_fit['flags']
        for i, length in enumerate(scan['lengths']):
            if i == bindex:
                row, parameters = selected_fit, fitted[selected]
            else:
                draws = scan['samples'][i][:, 4:4+nr]
                row, parameters = fit_window(r, scan['estimates'][i, 4:4+nr], draws,
                                              np.cov(draws, rowvar=False, ddof=1), group['Ns'],
                                              selected_fit['r_min'], selected_fit['r_max'])
            mass_scan.append(row)
            mass_draws.append(parameters)
        errors = np.asarray([[bootstrap_error(p[:, 1]), bootstrap_error(1/p[:, 1])] for p in mass_draws], dtype=np.float64)
        uncertainties = errors*np.asarray([math.sqrt(1/(2*(min(n)-1)) + 1/(2*(NBOOT-1))) for n in scan['counts']])[:, None]
        _, mass_plateaus = stable_three_length_groups(scan['lengths'], errors, uncertainties, minimum)
        mass_plateaus = [g for g in mass_plateaus if min(scan['counts'][scan['lengths'].index(g[-1])]) >= 30]
        if not any(scan['selected'] in g for g in mass_plateaus):
            flags.append('mass_blocking_plateau_review')
    else:
        mass_plateaus = []
    result = dict(preliminary, campaign_type=group.get('campaign_type', 'nonperturbative_mg_nt'),
                  status='first_stage_complete' if first_stage_only else 'preliminary_visual_review',
                  flags=sorted(set(flags)), mass=selected_fit['mass'] if selected_fit else None,
                  mass_error=selected_fit['mass_error'] if selected_fit else None,
                  selected_window_index=selected, suggested_window_index=selected,
                  C_G=mean[4:4+nr].tolist(), C_G_error=np.sqrt(np.diag(covariance)).tolist(),
                  xi_2nd=float(xi_second(mean, group['Ns'])),
                  xi_2nd_error=bootstrap_error(xi_second(samples, group['Ns'])),
                  fit_windows=rows, pairwise_chain_diagnostics=pairwise,
                  observable_plateau_groups=scan['common_groups'], mass_plateau_groups=mass_plateaus,
                  bootstrap_policy='500 fixed-chain within-chain vector draws; weights proportional to used complete-block measurements',
                  fit_policy='diagonal objective; bootstrap parameter SD; full covariance chi2 descriptive; no p-value',
                  boundary_conditions='spatial periodic; temporal Dirichlet; stored L2=Nt+1',
                  coupling_convention='3D SU(2) Wilson beta=4/(a*g3_squared)',
                  sweep_definition='one heatbath plus five overrelaxation sweeps',
                  measurement_interval_sweeps=1, profile_interval_sweeps=5,
                  input_records=group['chains'],
                  provenance=dict(manifest_sha256=sha256(manifest_path), raw_sha256=provenance,
                                  host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
                                  array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
                                  completed_utc=datetime.now(timezone.utc).isoformat(),
                                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                      [Path(__file__), ROOT/'scripts/mg_nt_fixed_statistics.py',
                                       ROOT/'scripts/mg_nt_fixed_canonical_plots.py',
                                       ROOT/'scripts/beta_nt_fixed/analysis.py', ROOT/'scripts/dirichlet_analysis.py',
                                       ROOT/'scripts/thermalization_diagnostics.py', ROOT/'uv.lock']},
                                  bootstrap_master_seed=20260724,
                                  operation_seed_rule='first 8 SHA256 bytes, little endian, of 20260724:ensemble:operation',
                                  selected_bootstrap_seed=scan['seeds'][bindex],
                                  blocking_seeds=[str(s) for s in scan['seeds']] ))
    np.savez_compressed(output/'bootstrap.npz', R=r, mean=mean, samples=samples, covariance=covariance,
                        window_parameters=fitted, window_bounds=np.asarray([[row['r_min'], row['r_max']] for row in rows]),
                        block_lengths=scan['lengths'], blocking_samples=np.asarray(scan['samples']),
                        blocking_means=scan['estimates'], mass_blocking_parameters=np.asarray(mass_draws),
                        chain_means=chain_means, chain_errors=chain_errors,
                        chain_samples=np.asarray(chain_boots), chain_weights=weights)
    write_json(output/('first_stage.json' if first_stage_only else 'analysis.json'), clean(result))
    write_csv(output/'chains.csv', [{k:v for k,v in c.items() if k != 'mser5'} for c in chains])
    write_csv(output/'autocorrelations.csv', taus)
    write_csv(output/'correlators.csv', [dict(ensemble=key, R=int(x), C_G=mean[4+i], bootstrap_error=float(np.sqrt(covariance[i,i]))) for i,x in enumerate(r)])
    write_csv(output/'fit_windows.csv', [dict(ensemble=key, window_index=i, **row) for i,row in enumerate(rows)])
    blocking_rows = [dict(ensemble=key, observable=label, block_length_sweeps=b,
                         estimate=clean(scan['estimates'][i,j]), bootstrap_error=clean(scan['errors'][i,j]),
                         se_uncertainty_approx=clean(scan['uncertainties'][i,j]),
                         min_blocks=min(scan['counts'][i]), max_blocks=max(scan['counts'][i]),
                         selected=b == scan['selected'], bootstrap_seed=str(scan['seeds'][i]), bootstrap_samples=500)
                     for i,b in enumerate(scan['lengths']) for j,label in enumerate(scan['labels'])]
    for i,row in enumerate(mass_scan):
        for name, values in [('mass', mass_draws[i][:,1]), ('xi_exp', 1/mass_draws[i][:,1])]:
            blocking_rows.append(dict(ensemble=key, observable=name, block_length_sweeps=scan['lengths'][i],
                                      estimate=row['mass'] if name == 'mass' else 1/row['mass'] if row['mass'] else None,
                                      bootstrap_error=bootstrap_error(values), min_blocks=min(scan['counts'][i]),
                                      max_blocks=max(scan['counts'][i]), selected=i == bindex, bootstrap_samples=500,
                                      bootstrap_seed=str(scan['seeds'][i])))
    write_csv(output/'blocking_scan.csv', blocking_rows)
    from mg_nt_fixed_canonical_plots import plot_ensemble
    plot_ensemble(output, clean(result), np.load(output/'bootstrap.npz'), mass_scan, scan)
    print(f'{key}: mass={result["mass"]} +/- {result["mass_error"]}; flags={result["flags"]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/mg_nt_fixed/manifest.json')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'analysis/mg_nt_fixed')
    parser.add_argument('--first-stage-only', action='store_true',
                        help='save all cuts, blocking, covariance and 500 replicas before mass scanning')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('raw-history analysis must run in a qbig compute allocation')
    manifest = json.loads(args.manifest.read_text())
    analyse(manifest['ensembles'][args.index], args.output_root, args.manifest, args.first_stage_only)


if __name__ == '__main__':
    main()
