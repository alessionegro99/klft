#!/usr/bin/env python3
"""Check canonical products, chain accounting, 500-slot arrays and PDF geometry."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from mg_nt_fixed_inventory import ROOT, load_campaign_manifest, sha256, write_json


def read_rows(path: Path) -> list[dict]:
    """Load a CSV without inferred spreadsheet types."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check_pdf_geometry(pdfs: list[Path]) -> None:
    """Check PDF metadata with Poppler, locally when qbig lacks pdfinfo."""
    if shutil.which('pdfinfo') is None:
        raise RuntimeError('Poppler pdfinfo is required for the PDF-only QA stage')
    for pdf in pdfs:
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        assert re.search(r'^Pages:\s+1$',info,re.MULTILINE),pdf
        dimensions=re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts',info)
        assert dimensions and tuple(map(float,dimensions.groups()))==(1152.,648.),pdf


def verify() -> dict:
    """Require complete unique coverage and honest finite-fit accounting."""
    data=ROOT/'data/mg_nt_fixed';analysis=ROOT/'analysis/mg_nt_fixed'
    manifest=load_campaign_manifest(data)
    roots=read_rows(analysis/'ensembles.csv');chains=read_rows(analysis/'chains.csv')
    assert len(roots)==manifest['ensemble_count'] and len(chains)==manifest['chain_count']
    assert len({r['raw_path'] for r in chains})==manifest['chain_count']
    assert sum(int(r['sweeps']) for r in chains)==manifest['total_sweeps']
    for row in roots:
        if row['campaign_type'] in ('nt1_matched_beta', 'matched_y_scaling'):
            assert row['target_c']=='' and row['fixed_c_continuum_input']=='False'
    assert sum('length_outlier' in r['flags'] for r in chains)==4
    assert all((int(r['sweeps'])==80000)==('length_outlier' in r['flags']) for r in chains)
    window_count=0;pdfs=[];source_hashes={}
    for group in manifest['ensembles']:
        leaf=analysis/group['ensemble'];result=json.loads((leaf/'analysis.json').read_text())
        assert result['fixed_c_continuum_input']==group['fixed_c_continuum_input']
        with np.load(leaf/'bootstrap.npz') as boot:
            # NpzFile decompresses on every lookup: cache the all-window
            # array once rather than rereading it for each of hundreds of fits.
            window_parameters=boot['window_parameters']
            np.testing.assert_array_equal(boot['R'],np.arange(group['Ns']//2+1))
            assert boot['samples'].shape[0]==500
            assert window_parameters.shape[1:]==(500,2)
            assert boot['blocking_samples'].shape[1]==500
            assert boot['chain_samples'].shape[1]==500
            np.testing.assert_allclose(boot['chain_weights'].sum(),1,atol=1e-14)
            used=np.asarray([c['used_sweeps'] for c in result['chains']])
            np.testing.assert_allclose(boot['chain_weights'],used/used.sum())
            np.testing.assert_allclose(boot['samples'][:,4],1,atol=1e-13)
            for i,row in enumerate(result['fit_windows']):
                valid=np.all(np.isfinite(window_parameters[i]),axis=1)
                assert row['bootstrap_successes']==int(valid.sum())
                assert row['n_points']>=5
                assert not any('p_value' in field for field in row)
            window_count+=len(result['fit_windows'])
        for c in result['chains']:
            assert c['used_sweeps']==c['n_complete_blocks']*c['block_length_sweeps']
            assert c['retained_sweeps']==c['sweeps']-c['thermalization_cut_sweeps']
            assert c['used_sweeps']+c['discarded_partial_sweeps']==c['retained_sweeps']
        for record in group['chains']:
            input_path=data/record['input']
            assert sha256(input_path)==record['input_sha256']
            parameters=dict(re.findall(r'^[ \t]+([A-Za-z_0-9]+):[ \t]*([^\n]+)',input_path.read_text(),re.MULTILINE))
            for name,expected in [('L0',group['Ns']),('L1',group['Ns']),('L2',group['physical_Nt']+1),
                                  ('seed',record['seed']),('nSweep',record['sweeps']),
                                  ('measurement_interval',1),('dirichlet_suite_measurement_interval',5),('nOverrelax',5)]:
                assert int(parameters[name])==expected,(input_path,name)
            assert float(parameters['beta'])==group['beta']
            assert parameters['temporal_dirichlet']=='true'
            assert parameters['start'].strip('"')=='hot'
        pdfs+=list(leaf.rglob('*.pdf'))
        source_hashes[str(leaf.relative_to(ROOT)/'analysis.json')]=sha256(leaf/'analysis.json')
    assert len(read_rows(analysis/'fit_windows.csv'))==window_count
    continuum=read_rows(analysis/'continuum.csv')
    assert all(row['p']=='1' for row in continuum)
    assert all('1' not in row['Nt_values'].split(';') for row in continuum)
    assert len(read_rows(analysis/'tuning.csv'))==10
    for path in analysis.glob('c_*/Nt_*/comparison/tuning_bootstrap.npz'):
        with np.load(path) as boot:
            assert boot['beta_star'].shape[1]==500
            assert boot['parameters'].shape[1:]==(500,2)
    for path in analysis.glob('c_*/comparison/continuum_bootstrap.npz'):
        with np.load(path) as boot:
            assert all(boot[key].shape==(500,2) for key in boot.files)
    pdfs+=list(analysis.glob('c_*/**/comparison/*.pdf'))+list((analysis/'comparison').glob('*.pdf'))
    pdfs=sorted(set(pdfs))
    geometry_status='pending_local_poppler'
    if shutil.which('pdfinfo'):
        check_pdf_geometry(pdfs)
        geometry_status='passed'
    result=dict(ensembles=manifest['ensemble_count'],chains=manifest['chain_count'],
                measured_sweeps=manifest['total_sweeps'],mass_windows=window_count,
                pdfs=len(pdfs),bootstrap_samples=500,pdf_canvas_points=[1152,648],
                pdf_geometry_status=geometry_status,
                source_sha256=source_hashes,job_id=os.environ.get('SLURM_JOB_ID'),status='passed')
    write_json(ROOT/'docs/mg_nt_fixed/generated/verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-only',action='store_true',help='local Poppler QA; no statistical calculations')
    args=parser.parse_args()
    if args.pdf_only:
        record_path=ROOT/'docs/mg_nt_fixed/generated/verification.json'
        record=json.loads(record_path.read_text())
        root=ROOT/'analysis/mg_nt_fixed'
        pdfs=sorted({p for target in root.glob('c_*') for p in target.rglob('*.pdf')} | set((root/'comparison').glob('*.pdf')))
        assert len(pdfs)==record['pdfs'],(len(pdfs),record['pdfs'])
        check_pdf_geometry(pdfs)
        record['pdf_geometry_status']='passed'
        write_json(record_path,record)
        print(f'Poppler verified {len(pdfs)} single-page 16:9 PDFs; no statistical calculation performed.')
    else:
        verify()


if __name__=='__main__':
    main()
