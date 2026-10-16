#!/usr/bin/env python3
"""Build the physics-keyed manifest from immutable inputs and saved provenance.

This command only inventories metadata; it never reads Monte Carlo histories.
Historical stage names survive only in provenance and unchanged qbig raw paths.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/qbigwork2/negro/dirichlet')
ARCHIVE_HASH = '84b0b7ca8e3c6b07f6ea8dc0dd07976e6cb0a57e90dc2499ee459abc4169aef6'


def token(value: str | float | int) -> str:
    """Canonical exact decimal token, with no redundant trailing zeroes."""
    return format(Decimal(str(value)).normalize(), 'f').replace('.', 'p')


def ensemble_key(c: float, nt: int, ns: int, beta: float) -> str:
    """Return the sole active path for a physical ensemble."""
    return f'c_{token(c)}/Nt_{nt}/Ns_{ns}/beta_{token(beta)}'


def sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write strict JSON; invalid numerical results must be explicit nulls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write rectangular, UTF-8 colleague-facing data with explicit columns."""
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ';'.join(map(str, v)) if isinstance(v, list) else v
                             for k, v in row.items()})


def load_campaign_manifest(data: Path | None = None) -> dict:
    """Combine immutable physics-role inventories without modifying originals.

    Untuned matched-beta and matched-y ensembles belong in the canonical tree
    and statistics register, but cannot enter fixed-c tuning or extrapolation.
    """
    data = ROOT/'data/mg_nt_fixed' if data is None else data
    paths = [data/'manifest.json']
    for name in ('nt1_analysis_manifest.json', 'scaling_analysis_manifest.json'):
        if (data/name).exists():
            paths.append(data/name)
    inventories = [json.loads(p.read_text()) for p in paths]
    groups = [dict(g, campaign_type=g.get('campaign_type', m['campaign_type']),
                   reference_c=g.get('reference_c', g['target_c']),
                   fixed_c_continuum_input=g.get('fixed_c_continuum_input', True))
              for m in inventories for g in m['ensembles']]
    chains = [c for g in groups for c in g['chains']]
    if len({g['ensemble'] for g in groups}) != len(groups) or len({c['raw_path'] for c in chains}) != len(chains):
        raise ValueError('overlapping ensemble or raw-chain inventories')
    if len({c['seed'] for c in chains}) != len(chains):
        raise ValueError('reused seed across campaign inventories')
    if any(g['fixed_c_continuum_input'] and g['target_c'] is None for g in groups):
        raise ValueError('fixed-c input must declare its target')
    groups.sort(key=lambda g:(g['reference_c'], g['physical_Nt'], g['Ns'], g['beta']))
    return dict(ensemble_count=len(groups), chain_count=len(chains),
                total_sweeps=sum(c['sweeps'] for c in chains), ensembles=groups,
                source_manifest_sha256={p.name:sha256(p) for p in paths})


def inventory() -> dict:
    """Deduplicate by raw chain path, checking manifest/input agreement."""
    data = ROOT / 'data/mg_nt_fixed'
    native = {}
    for manifest in data.rglob('*manifest.tsv'):
        with manifest.open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if not row.get('run') or not row.get('input_sha256'):
                    continue
                path = str(REMOTE / 'data' / row['run'])
                source = manifest.parent / row['input']
                if not source.is_file():
                    continue
                if sha256(source) != row['input_sha256']:
                    raise ValueError(f'input changed: {source}')
                record = dict(row, source_input=str(source.relative_to(ROOT)),
                              source_manifest=str(manifest.relative_to(ROOT)))
                if path in native and native[path]['input_sha256'] != record['input_sha256']:
                    raise ValueError(f'conflicting input: {path}')
                native[path] = record
    groups = {}
    for path in sorted((ROOT / 'analysis/mg_nt_fixed').rglob('analysis.json')):
        result = json.loads(path.read_text())
        if result.get('target_c') is None or 'thermalization_and_blocking' not in result:
            continue
        c, nt, ns, beta = (float(result['target_c']), int(result['physical_Nt']),
                           int(result['Ns']), float(result['beta']))
        key = ensemble_key(c, nt, ns, beta)
        group = groups.setdefault(key, dict(ensemble=key, target_c=c, physical_Nt=nt,
                                           stored_L2=nt+1, Ns=ns, beta=beta,
                                           chains={}, historical_analyses=[]))
        group['historical_analyses'].append(str(path.relative_to(ROOT)))
        for chain in result['thermalization_and_blocking']['chains']:
            raw = chain['chain']
            if raw in group['chains']:
                continue
            seed = int(Path(raw).name.removeprefix('seed_'))
            if raw in native:
                record = native[raw].copy()
                record['source_campaign'] = 'mg_nt_fixed'
            else:
                matches = list((ROOT / 'data/beta_nt_fixed').rglob(f'seed_{seed}/input.yaml'))
                if len(matches) != 1:
                    raise ValueError(f'ambiguous prior input for {raw}: {matches}')
                record = dict(source_input=str(matches[0].relative_to(ROOT)),
                              source_manifest='data/mg_nt_fixed/pilot/prior_points.tsv',
                              source_campaign='beta_nt_fixed', input_sha256=sha256(matches[0]),
                              sweeps=10000, measurement_interval=1,
                              dirichlet_suite_measurement_interval=5,
                              klft_commit='unknown_prior_revision', heatbath_sha256='unknown_prior_hash')
            for field in ('sweeps', 'measurement_interval', 'dirichlet_suite_measurement_interval'):
                record[field] = int(record[field])
            record.update(seed=seed, raw_path=raw,
                          input=f'{key}/seed_{seed}/input.yaml',
                          flags=['length_outlier'] if record['sweeps'] == 80000 else
                                ['prior_10000_sweeps'] if record['sweeps'] == 10000 else [])
            group['chains'][raw] = record
    ensembles = sorted(groups.values(), key=lambda d: (d['target_c'], d['physical_Nt'], d['Ns'], d['beta']))
    for i, group in enumerate(ensembles):
        group['index'] = i
        group['chains'] = sorted(group['chains'].values(), key=lambda d: d['seed'])
    chains = [r for g in ensembles for r in g['chains']]
    if len(ensembles) != 38 or len(chains) != 156 or len({r['raw_path'] for r in chains}) != 156:
        raise ValueError(f'inventory differs from audited 38/156: {len(ensembles)}/{len(chains)}')
    if sum(r['sweeps'] for r in chains) != 1552000 or len({r['seed'] for r in chains}) != 156:
        raise ValueError('sweep total or seed uniqueness differs from audit')
    return dict(campaign_type='nonperturbative_mg_nt', bootstrap_samples=500,
                archive_sha256=ARCHIVE_HASH,
                archive_path=str(REMOTE / 'archive/mg_nt_fixed' / ARCHIVE_HASH / 'workspace.tar.gz'),
                boundary_conditions='spatial periodic; temporal Dirichlet; L2=Nt+1',
                coupling_convention='3D SU(2) Wilson beta=4/(a*g3_squared)',
                ensemble_count=38, chain_count=156, total_sweeps=1552000,
                excluded_validation_histories=9, ensembles=ensembles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    destination = ROOT / 'data/mg_nt_fixed/manifest.json'
    if destination.exists():
        raise FileExistsError('immutable inventory already exists')
    result = inventory()
    migration = []
    for group in result['ensembles']:
        leaf = ROOT / 'data/mg_nt_fixed' / group['ensemble']
        for chain in group['chains']:
            target = ROOT / 'data/mg_nt_fixed' / chain['input']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(ROOT / chain['source_input'], target)
            if sha256(target) != chain['input_sha256']:
                raise ValueError(f'copy hash mismatch: {target}')
            migration.append(dict(ensemble=group['ensemble'], seed=chain['seed'],
                                  old_input=chain['source_input'], new_input=str(target.relative_to(ROOT)),
                                  raw_path=chain['raw_path'], input_sha256=chain['input_sha256']))
        write_json(leaf / 'provenance.json', group)
        (leaf / 'README.md').write_text(f"# {group['ensemble']}\n\nImmutable input copies; raw histories remain at the paths in provenance.json.\n")
    write_json(destination, result)
    write_csv(ROOT / 'data/mg_nt_fixed/migration.csv', migration)
    print(f"Inventoried {len(result['ensembles'])} ensembles, 156 unique chains, 1,552,000 sweeps.")


if __name__ == '__main__':
    main()
