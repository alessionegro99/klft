#!/usr/bin/env python3
"""Bind the completed Nt=1 A100 runs to a separate immutable analysis inventory.

The reference c identifies the source Nt=2 tuning. No Nt=1 target c was tuned.
Only production records listed in the checked execution plan are eligible;
smoke and timing histories never enter this manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mg_nt_fixed_inventory import ROOT, sha256, write_json
from mg_nt_fixed_nt1_run import validate_scaling_manifest


def build_inventory(data: Path) -> dict:
    """Verify completion and provenance without reading raw Monte Carlo data."""
    plan_path = data/'nt1_a100_production.json'
    plan = json.loads(plan_path.read_text())
    groups = {}
    for run in plan['runs']:
        input_path = data/run['input']
        result_path = input_path.parent/'execution_result.json'
        record = json.loads(result_path.read_text())
        if (record['status'] != 'complete' or record['purpose'] != 'production'
                or record['manifest_sha256'] != sha256(plan_path)
                or record['heatbath_sha256'] != plan['heatbath_sha256']
                or record['input_sha256'] != sha256(input_path)
                or record['input_sha256'] != run['input_sha256']
                or record['seed'] != run['seed']
                or record['checks']['summary_measurements'] != 8000
                or record['checks']['profile_measurements'] != 1600
                or record['checks']['correlator_rows'] != 8000*(record['Ns']//2+1)
                or record['target_c'] is not None or record['fixed_c_continuum_input']):
            raise ValueError(f'invalid completed production record: {result_path}')
        key = record['ensemble']
        group = groups.setdefault(key, dict(ensemble=key, campaign_type='nt1_matched_beta',
            reference_c=record['reference_c'], target_c=None, fixed_c_continuum_input=False,
            physical_Nt=1, stored_L2=2, Ns=record['Ns'], beta=record['beta'], chains=[]))
        group['chains'].append(dict(record, source_campaign='mg_nt_fixed', flags=[],
            execution_result_sha256=sha256(result_path)))
    ensembles = sorted(groups.values(), key=lambda p:p['reference_c'])
    for i, group in enumerate(ensembles):
        group['index'] = i
        group['chains'].sort(key=lambda c:c['seed'])
    chains = [c for g in ensembles for c in g['chains']]
    if len(ensembles) != 3 or len(chains) != 12 or len({c['seed'] for c in chains}) != 12:
        raise ValueError('expected three ensembles and twelve unique production chains')
    return dict(campaign_type='nt1_matched_beta', bootstrap_samples=500,
        reference_c_definition='source Nt=2 target; no Nt=1 target was imposed',
        fixed_c_continuum_input=False, ensemble_count=3, chain_count=12, total_sweeps=96000,
        execution_plan_sha256=sha256(plan_path), ensembles=ensembles)


def build_scaling_inventory(data: Path) -> dict:
    """Bind the four completed Nt=1 matched-y sets to their P100 manifest."""
    manifest_path = data/'nt1_scaling_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    validate_scaling_manifest(manifest)
    manifest_hash = sha256(manifest_path)
    groups = {}
    for run in manifest['runs']:
        if run['purpose'] != 'production':
            continue
        input_path = data/run['input']
        result_path = input_path.parent/'execution_result.json'
        record = json.loads(result_path.read_text())
        checks = record.get('checks', {})
        checkpoint = checks.get('checkpoint', {})
        ns, sweeps = run['Ns'], run['sweeps']
        expected_counts = dict(summary_measurements=sweeps,
                               correlator_rows=sweeps*(ns//2+1),
                               profile_measurements=sweeps//5)
        outputs = {'holonomy_correlator.out', 'holonomy_summary.out',
                   'plaquette_profile.out', 'configuration_final.klft'}
        if (record.get('status') != 'complete'
                or any(record.get(k) != value for k, value in run.items())
                or record.get('manifest_sha256') != manifest_hash
                or sha256(input_path) != run['input_sha256']
                or any(checks.get(k) != value for k, value in expected_counts.items())
                or checkpoint.get('boundary_flag') != 3
                or checkpoint.get('physical_links') != 5*ns*ns
                or not 0 <= checkpoint.get('max_squared_norm_error', float('inf')) <= 1.e-10
                or set(record.get('output_sha256', {})) != outputs):
            raise ValueError(f'invalid completed Nt=1 matched-y record: {result_path}')
        key = run['ensemble']
        group = groups.setdefault(key, dict(ensemble=key, campaign_type='matched_y_scaling',
            reference_c=run['reference_c'], target_c=None, fixed_c_continuum_input=False,
            physical_Nt=1, stored_L2=2, Ns=ns, beta=run['beta'],
            beta_over_nt=run['beta_over_nt'], chains=[]))
        group['chains'].append(dict(record, source_campaign='mg_nt_fixed', flags=[],
                                   execution_result_sha256=sha256(result_path)))
    ensembles = sorted(groups.values(), key=lambda group: group['beta'])
    for i, group in enumerate(ensembles):
        group['index'] = i
        group['chains'].sort(key=lambda chain: chain['seed'])
    chains = [chain for group in ensembles for chain in group['chains']]
    if len(ensembles) != 4 or any(len(group['chains']) != 4 for group in ensembles):
        raise ValueError('expected four Nt=1 ensembles with four independent chains each')
    return dict(campaign_type='matched_y_scaling', bootstrap_samples=500,
        reference_c_definition=manifest['reference_c_definition'],
        fixed_c_continuum_input=False, ensemble_count=4, chain_count=16,
        total_sweeps=sum(chain['sweeps'] for chain in chains),
        manifest_sha256=manifest_hash, ensembles=ensembles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-y', action='store_true',
                        help='inventory the four completed Nt=1, Ns=128 matched-y sets')
    args = parser.parse_args()
    data = ROOT/'data/mg_nt_fixed'
    destination = data/('nt1_scaling_analysis_manifest.json' if args.matched_y
                       else 'nt1_analysis_manifest.json')
    if destination.exists():
        raise FileExistsError('immutable Nt=1 analysis inventory already exists')
    result = build_scaling_inventory(data) if args.matched_y else build_inventory(data)
    write_json(destination, result)
    for group in result['ensembles']:
        write_json(data/group['ensemble']/'provenance.json', group)
    print(f'Verified {result["chain_count"]} complete Nt=1 chains; target_c=null; 500 analysis replicas.')


if __name__ == '__main__':
    main()
