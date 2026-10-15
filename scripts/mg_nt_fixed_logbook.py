#!/usr/bin/env python3
"""Build the human-readable canonical campaign logbook; no resampling."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from mg_nt_fixed_inventory import ROOT, sha256, write_json

RESULTS = ROOT/'analysis/mg_nt_fixed'
REPORT = ROOT/'docs/mg_nt_fixed'
PRODUCTS = REPORT/'generated'
ACTIVE_SOURCE_PATHS = (
    'scripts/mg_nt_fixed_analysis.py', 'scripts/mg_nt_fixed_ensemble.py',
    'scripts/mg_nt_fixed_statistics.py', 'scripts/mg_nt_fixed_inventory.py',
    'scripts/mg_nt_fixed_parallel.py', 'scripts/mg_nt_fixed_adaptive.py',
    'scripts/mg_nt_fixed_campaign.py', 'scripts/mg_nt_fixed_canonical_plots.py',
    'scripts/mg_nt_fixed_verify.py', 'scripts/mg_nt_fixed_logbook.py',
    'scripts/mg_nt_fixed_nt1_inventory.py', 'scripts/mg_nt_fixed_nt1_inputs.py',
    'scripts/mg_nt_fixed_nt1_run.py', 'scripts/mg_nt_fixed_nt1_a100.py',
    'scripts/beta_nt_fixed/analysis.py', 'scripts/dirichlet_analysis.py',
    'scripts/pcm_beta_pilot_analysis.py', 'scripts/thermalization_diagnostics.py',
    'scripts/plot_thesis_results.py', 'uv.lock',
)


def read_json(path: Path) -> dict:
    """Load one strict compact result."""
    return json.loads(path.read_text())


def selected_fit(result: dict) -> dict:
    """Read a selected historical fit for archive-review compatibility."""
    return next((fit for fit in result.get('K0_fit_scan',[]) if fit.get('selected')), {})


def maximum_tau(chain: dict) -> float:
    """Read nested historical tau diagnostics without changing their policy."""
    def values(record: dict) -> list[float]:
        return [x for value in record.values() for x in
                (values(value) if isinstance(value,dict) else
                 [float(value)] if isinstance(value,(int,float)) else [])]
    return max(values(chain['tau_int_sweeps']))


def source_hashes() -> dict[str,str]:
    """Track all canonical results, review choices, plots and report sources."""
    paths = [REPORT/'logbook.tex',REPORT/'review.json',ROOT/'data/mg_nt_fixed/manifest.json']
    paths += list((ROOT/'data/mg_nt_fixed').glob('nt1_*manifest.json'))
    paths += list((ROOT/'data/mg_nt_fixed').glob('nt1_a100*json'))
    # Retired stage helpers are archived separately. Requiring every matching
    # filename made a published report depend on unrelated unversioned files.
    paths += [ROOT/path for path in ACTIVE_SOURCE_PATHS]
    paths += list(RESULTS.glob('*.csv'))+list(RESULTS.glob('*.json'))
    for target in sorted(RESULTS.glob('c_*')):
        paths += [p for p in target.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz','.pdf')]
    paths += list((RESULTS/'comparison').glob('*.pdf'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}


def check_freshness() -> None:
    """Fail if any result, human review, figure or source has changed."""
    recorded=read_json(PRODUCTS/'build_manifest.json')
    current=source_hashes()
    if recorded['source_sha256']!=current:
        changed=sorted(k for k in current.keys()|recorded['source_sha256'].keys() if current.get(k)!=recorded['source_sha256'].get(k))
        raise ValueError('logbook needs updating: '+', '.join(changed))
    for path,expected in recorded['output_sha256'].items():
        if sha256(ROOT/path)!=expected:
            raise ValueError(f'logbook product changed: {path}')
    print('Logbook matches every canonical result, review choice and figure.')


def table(path: Path) -> list[dict]:
    """Read a colleague-facing CSV as the report's numerical source."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(value: str | float | None, digits: int = 5) -> str:
    """Compact decimal formatting without inventing missing estimates."""
    return f'{float(value):.{digits}g}' if value not in ('',None) else '--'


def build() -> None:
    """Generate compact tables and compile the reviewed LaTeX narrative."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('build the campaign logbook on a qbig compute node')
    PRODUCTS.mkdir(parents=True,exist_ok=True)
    status=read_json(RESULTS/'campaign.json')
    review=read_json(REPORT/'review.json')
    if review.get('policy')!='canonical_pooled_500_flagged_p1':
        raise ValueError('review.json must explicitly acknowledge the current policy')
    macros={'Ensembles':status['ensembles'],'Chains':status['chains'],'TotalSweeps':status['total_sweeps'],
            'RetainedSweeps':status['retained_sweeps'],'UsedSweeps':status['used_sweeps'],
            'NumericalMasses':status['numerical_masses'],'TuningPoints':status['tuning_points'],
            'ContinuumTargets':status['continuum_targets']}
    (PRODUCTS/'numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{key}}}{{{value:,}}}' for key,value in macros.items())+'\n')
    rows=table(RESULTS/'tuning.csv')
    lines=[r'\begin{tabular}{rrrrrl}',r'\toprule',r'$c$ & $N_t$ & $\beta_\star$ & 16--84\% interval & interp. shift & status\\\midrule']
    for row in rows:
        interval='['+number(row.get('stat_low'))+', '+number(row.get('stat_high'))+']'
        lines.append(' & '.join([number(row['target_c']),row['physical_Nt'],number(row['beta_star']),interval,
                                 number(row.get('interpolation_systematic'),3),'flagged' if row.get('flags') else 'preliminary'])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PRODUCTS/'tuning_table.tex').write_text('\n'.join(lines)+'\n')
    rows=table(RESULTS/'continuum.csv')
    names={.5:'Half',1.:'One',2.:'Two'}
    commands=[]
    for row in rows:
        name='Q'+names[float(row['target_c'])]+('All' if row['fit']=='all_Nt' else 'Drop')
        commands.append(f'\\newcommand{{\\{name}}}{{{number(row["Q0"])}}}')
    numbers_path=PRODUCTS/'numbers.tex'
    numbers_path.write_text(numbers_path.read_text()+'\n'.join(commands)+'\n')
    lines=[r'\begin{tabular}{rlrrrr}',r'\toprule',r'$c$ & resolutions & $Q_0$ & boot. SE & interp. bound & dof\\\midrule']
    for row in rows:
        lines.append(' & '.join([number(row['target_c']),row['Nt_values'].replace(';',', '),number(row['Q0']),number(row['Q0_error'],3),number(row['Q0_interpolation_shift_bound'],3),row['nominal_dof']])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PRODUCTS/'continuum_table.tex').write_text('\n'.join(lines)+'\n')
    rows=table(RESULTS/'ensembles.csv')
    lines=[r'\begin{longtable}{rrrrrrrl}',r'\toprule',r'$c$ & $N_t$ & $N_s$ & $\beta$ & $am_G$ & SE & window & used sweeps\\\midrule\endhead']
    for row in rows:
        if row.get('fixed_c_continuum_input')=='False':
            continue
        lines.append(' & '.join([number(row['target_c']),row['physical_Nt'],row['Ns'],number(row['beta'],9),number(row['mass']),number(row['mass_error'],3),
                                 row['r_min']+'--'+row['r_max'],format(int(row['used_sweeps']),',')])+r'\\')
    lines += [r'\bottomrule',r'\end{longtable}']
    (PRODUCTS/'ensemble_table.tex').write_text('\n'.join(lines)+'\n')
    nt1 = [r for r in rows if r.get('campaign_type')=='nt1_matched_beta']
    lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',
           r'$c_{\rm ref}$ & $\beta$ & $N_s$ & $am_G$ & boot. SE & $\xi_{\rm 2nd}/a$ & min. blocks\\\midrule']
    for row in nt1:
        lines.append(' & '.join([number(row['reference_c']),number(row['beta'],7),row['Ns'],
            number(row['mass']),number(row['mass_error'],3),number(row['xi_2nd'],4),row['min_blocks']])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PRODUCTS/'nt1_table.tex').write_text('\n'.join(lines)+'\n')
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={PRODUCTS}',str(REPORT/'logbook.tex')]
    for _ in range(2):
        completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        (PRODUCTS/'compile.log').write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-6000:])
    shutil.copy2(PRODUCTS/'logbook.pdf',REPORT/'logbook.pdf')
    outputs=[REPORT/'logbook.pdf',PRODUCTS/'numbers.tex',PRODUCTS/'tuning_table.tex',PRODUCTS/'continuum_table.tex',PRODUCTS/'ensemble_table.tex',PRODUCTS/'nt1_table.tex']
    write_json(PRODUCTS/'build_manifest.json',dict(source_sha256=source_hashes(),
               output_sha256={str(p.relative_to(ROOT)):sha256(p) for p in outputs},
               build_job=os.environ.get('SLURM_JOB_ID'),bootstrap_operations=0))
    check_freshness()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    check_freshness() if args.check else build()


if __name__=='__main__':
    main()
