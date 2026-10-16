#!/usr/bin/env python3
"""Bind completed matched-y runs to a separate immutable analysis inventory.

This metadata-only step excludes smoke/timing runs and never reads histories.
The c folder is a reference family; these ensembles have no imposed target c.
"""
from __future__ import annotations

import json
from pathlib import Path

from mg_nt_fixed_inventory import ROOT, sha256, write_json
from mg_nt_fixed_scaling_run import OUTPUTS, validate_manifest


def build_inventory(data: Path) -> dict:
    """Validate every completed chain against its immutable execution inputs."""
    manifest_path = data/'scaling_manifest.json'
    plan_path = data/'scaling_execution.json'
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    validate_manifest(manifest)
    manifest_hash, plan_hash = sha256(manifest_path), sha256(plan_path)
    if plan['manifest_sha256'] != manifest_hash:
        raise ValueError('execution plan does not match the immutable manifest')
    groups = {}
    for run in manifest['runs']:
        if run['purpose'] != 'production':
            continue
        input_path = data/run['input']
        result_path = input_path.parent/'execution_result.json'
        record = json.loads(result_path.read_text())
        nt, ns, sweeps = run['physical_Nt'], run['Ns'], run['sweeps']
        checks = record.get('checks', {})
        checkpoint = checks.get('checkpoint', {})
        expected_counts = dict(summary_measurements=sweeps,
            correlator_rows=sweeps*(ns//2+1), profile_measurements=sweeps//5,
            profile_rows=(sweeps//5)*(2*nt+1))
        expected_geometry = dict(boundary_flag=3, physical_Nt=nt, stored_L2=nt+1,
                                 physical_links=ns*ns*(3*nt+2))
        if (record.get('status') != 'complete'
                or any(record.get(k) != v for k, v in run.items())
                or record.get('manifest_sha256') != manifest_hash
                or record.get('execution_plan_sha256') != plan_hash
                or sha256(input_path) != run['input_sha256']
                or any(checks.get(k) != v for k, v in expected_counts.items())
                or any(checkpoint.get(k) != v for k, v in expected_geometry.items())
                or not 0 <= checkpoint.get('max_squared_norm_error', float('inf')) <= 1.e-10
                or set(record.get('output_sha256', {})) != set(OUTPUTS)):
            raise ValueError(f'invalid completed production record: {result_path}')
        key = run['ensemble']
        group = groups.setdefault(key, dict(ensemble=key, campaign_type='matched_y_scaling',
            reference_c=run['reference_c'], target_c=None, fixed_c_continuum_input=False,
            physical_Nt=nt, stored_L2=nt+1, Ns=ns, beta=run['beta'],
            beta_over_nt=run['beta_over_nt'], chains=[]))
        group['chains'].append(dict(record, source_campaign='mg_nt_fixed', flags=[],
                                   execution_result_sha256=sha256(result_path)))
    ensembles = sorted(groups.values(), key=lambda g:(g['physical_Nt'], g['Ns'], g['beta']))
    for i, group in enumerate(ensembles):
        group['index'] = i
        group['chains'].sort(key=lambda c:c['seed'])
    chains = [c for g in ensembles for c in g['chains']]
    if (len(ensembles) != 4 or any(len(g['chains']) != 4 for g in ensembles)
            or len({c['seed'] for c in chains}) != 16):
        raise ValueError('expected four ensembles and sixteen unique production chains')
    return dict(campaign_type='matched_y_scaling', bootstrap_samples=500,
        reference_c_definition=manifest['reference_c_definition'],
        fixed_c_continuum_input=False, ensemble_count=4, chain_count=16,
        total_sweeps=sum(c['sweeps'] for c in chains),
        manifest_sha256=manifest_hash, execution_plan_sha256=plan_hash, ensembles=ensembles)


def main() -> None:
    data = ROOT/'data/mg_nt_fixed'
    destination = data/'scaling_analysis_manifest.json'
    if destination.exists():
        raise FileExistsError('immutable matched-y analysis inventory already exists')
    result = build_inventory(data)
    write_json(destination, result)
    for group in result['ensembles']:
        write_json(data/group['ensemble']/'provenance.json', group)
    print('Verified sixteen complete matched-y chains; target_c=null; 500 analysis replicas.')


if __name__ == '__main__':
    main()
