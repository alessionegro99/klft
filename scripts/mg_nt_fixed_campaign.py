#!/usr/bin/env python3
"""Collect canonical masses, coupling windows and p=1 continuum comparisons.

Numerical work runs on qbig. Existing 500-replica arrays are propagated by
replica index, preserving covariance between alternate windows using the same
ensemble. Different ensembles use independent named RNG streams. No extra
bootstrap or resampling of a bootstrap distribution is necessary.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations, groupby
import json
import os
from pathlib import Path

import numpy as np
from scipy import linalg

from mg_nt_fixed_canonical_plots import finish, plot_ensemble, title
from mg_nt_fixed_ensemble import clean
from mg_nt_fixed_inventory import ROOT, load_campaign_manifest, sha256, token, write_csv, write_json
from mg_nt_fixed_statistics import bootstrap_error, fit_window, review_window
from mg_nt_fixed_scaling_plots import SCALING_PDFS, plot_scaling, squared_length_error
from plot_thesis_results import FIGURE_SIZE, plt, pu

RESULTS = ROOT/'analysis/mg_nt_fixed'


def read_csv(path: Path) -> list[dict]:
    """Read a rectangular exported table."""
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def line_fit(x: np.ndarray, y: np.ndarray, error: np.ndarray, samples: np.ndarray) -> dict:
    """Weighted line and joint 500-replica propagation, without p-values.

    Unlike fit_utils.boot_fit, this needs missing-replica slot preservation
    and full joint covariance reporting for downstream window comparisons.
    QR least squares avoids explicitly inverting the normal equations.
    """
    if samples.shape != (500,len(x)) or len(x) < 2 or np.any(error <= 0):
        raise ValueError('line fit requires 500 aligned replicas and positive errors')
    design = np.column_stack((np.ones_like(x),x))
    scaled = design/error[:,None]
    projection = linalg.lstsq(scaled, np.diag(1/error))[0]
    central = projection @ y
    parameters = samples @ projection.T
    residual = y - design @ central
    valid = np.all(np.isfinite(samples), axis=1)
    covariance = np.cov(samples[valid], rowvar=False, ddof=1)
    chi2, condition = None, None
    flags = []
    try:
        condition = float(np.linalg.cond(covariance))
        factor = linalg.cholesky(covariance, lower=True)
        whitened = linalg.solve_triangular(factor, residual, lower=True)
        chi2 = float(whitened @ whitened)
    except (ValueError,linalg.LinAlgError):
        flags.append('singular_full_covariance')
    return dict(parameters=central, bootstrap_parameters=parameters, residual=residual,
                chi2_full=chi2, nominal_dof=len(x)-2, covariance_condition=condition,
                flags=flags, covariance=covariance)


def read_review(manifest: dict) -> dict[tuple[str,str],dict]:
    """Create a minimal editable review register once; preserve all edits."""
    path = RESULTS/'review.csv'
    rows = read_csv(path) if path.exists() else []
    keys = {(r['scope'], r['key']) for r in rows}
    required = [('ensemble', g['ensemble']) for g in manifest['ensembles']]
    required += [('tuning', f'c_{token(c)}/Nt_{nt}') for c, nt in sorted({
        (g['target_c'], g['physical_Nt']) for g in manifest['ensembles'] if g['fixed_c_continuum_input']})]
    for scope, key in required:
        if (scope, key) not in keys:
            rows.append(dict(scope=scope,key=key,decision='auto',r_min='',r_max='',betas='',note=''))
    write_csv(path,rows)
    if any(r['decision'] not in ('auto','include','exclude') for r in rows):
        raise ValueError('review decisions must be auto, include or exclude')
    keys = [(r['scope'],r['key']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate review keys')
    return dict(zip(keys,rows))


def enforce_covariance_rank_bound(result: dict) -> bool:
    """Leave chi-square undefined when the block inventory proves singularity.

    Within a chain with n blocks, resampling multiplicity deviations sum to
    zero, so its centered replicas span at most n-1 directions. Fixed-weight
    pooling therefore has rank at most sum_k(n_k-1), also bounded by 499 for
    500 centered replicas. Roundoff can otherwise make Cholesky succeed for
    a mathematically singular covariance. Fit parameters/refits are unchanged.
    """
    limit = min(499, sum(c['n_complete_blocks']-1 for c in result['chains']))
    selected_changed = False
    for i, row in enumerate(result['fit_windows']):
        row['covariance_rank_limit'] = limit
        if row['n_points'] > limit:
            selected_changed |= i == result.get('selected_window_index') and row['chi2_full'] is not None
            row.update(chi2_full=None, chi2_full_per_dof=None)
            row['flags'] = sorted(set(row['flags']+['singular_full_covariance']))
    return selected_changed


def load_ensembles(manifest: dict, reviews: dict) -> list[dict]:
    """Load each physical set once and apply explicit manual choices."""
    results = []
    for group in manifest['ensembles']:
        leaf = RESULTS/group['ensemble']
        result = json.loads((leaf/'analysis.json').read_text())
        result.update(reference_c=group['reference_c'],
                      fixed_c_continuum_input=group['fixed_c_continuum_input'])
        rank_diagnostic_changed = enforce_covariance_rank_bound(result)
        # Older prior-input manifests predate the campaign's hash fields.
        # Recover those fields from the original run metadata on qbig, and
        # retain a checksum of that metadata alongside the raw-file checksums.
        for record in result['input_records']:
            metadata_path=Path(record['raw_path'])/'run_metadata.txt'
            if metadata_path.exists():
                metadata=dict(line.split(None,1) for line in metadata_path.read_text().splitlines() if len(line.split(None,1))==2)
                for field in ('klft_commit','heatbath_sha256'):
                    if str(record.get(field,'')).startswith('unknown') and field in metadata:
                        record[field]=metadata[field]
                result['provenance']['raw_sha256'][str(metadata_path)]=sha256(metadata_path)
                write_json(ROOT/'data/mg_nt_fixed'/group['ensemble']/f'seed_{record["seed"]}'/'run_metadata.json',metadata)
        write_json(ROOT/'data/mg_nt_fixed'/group['ensemble']/'provenance.json',
                   dict(group,chains=result['input_records'],metadata_verified_on_qbig=True))
        boot = np.load(leaf/'bootstrap.npz')
        review = reviews.get(('ensemble',group['ensemble']),{})
        suggested = result['suggested_window_index']
        result.setdefault('automatic_flags',result['flags'].copy())
        per_window_flags={flag for row in result['fit_windows'] for flag in row['flags']}
        derived_flags={'late_window_drift','finite_volume_safety_review','manual_window_override','manual_window_blocking_review'}
        result['flags']=sorted(set(result['automatic_flags'])-per_window_flags-derived_flags)
        selected = review_window(result['fit_windows'],suggested,review)
        result['selected_window_index'] = selected
        for i,row in enumerate(result['fit_windows']):
            row['selected'] = i == selected
        result['review'] = review
        if selected is not None:
            row = result['fit_windows'][selected]
            result['flags']=sorted(set(result['flags']+row['flags']))
            result['mass'], result['mass_error'] = row['mass'],row['mass_error']
            parameters=boot['window_parameters'][selected]
            result['_mass_samples'] = parameters[:,1]
            result['_parameters'] = parameters
            later=[window for window in result['fit_windows'] if window['r_min']>row['r_min']
                   and window['r_max']==row['r_max'] and window['mass_error'] is not None]
            if any(abs(window['mass']-row['mass']) > 2*np.hypot(window['mass_error'],row['mass_error']) for window in later):
                result['flags']=sorted(set(result['flags']+['late_window_drift']))
            if result['mass']*result['Ns'] < 12:
                result['flags']=sorted(set(result['flags']+['finite_volume_safety_review']))
        else:
            result['_mass_samples'] = np.full(500,np.nan)
        if selected != suggested:
            result['flags'] = sorted(set(result['flags']+['manual_window_override','manual_window_blocking_review']))
        result['include'] = review.get('decision') != 'exclude' and selected is not None
        result['review_status'] = 'user_included' if review.get('decision') == 'include' else 'user_excluded' if review.get('decision') == 'exclude' else 'pending_visual_review'
        result['_boot'] = boot
        results.append(result)
        # The authoritative JSON records the current manual choice as well as
        # the immutable suggested index and all original window fits.
        write_json(leaf/'analysis.json',clean({k:v for k,v in result.items() if not k.startswith('_')}))
        write_csv(leaf/'fit_windows.csv',[dict(ensemble=group['ensemble'],window_index=i,**row) for i,row in enumerate(result['fit_windows'])])
        write_csv(leaf/'chain_consistency.csv',[dict(ensemble=group['ensemble'],**row) for row in result['pairwise_chain_diagnostics']])
        if selected != suggested or rank_diagnostic_changed:
            plot_ensemble(leaf,result,boot)
    return results


def primary_volumes(points: list[dict]) -> list[dict]:
    """One explicit volume per beta; retain other volumes for comparisons."""
    selected = []
    for beta, iterator in groupby(sorted(points,key=lambda p:p['beta']), key=lambda p:p['beta']):
        same = list(iterator)
        c,nt = same[0]['target_c'],same[0]['physical_Nt']
        preferred = 96 if (c,nt)==(.5,4) and abs(beta-5.58394149)<1e-9 and any(p['Ns']==96 for p in same) else max(p['Ns'] for p in same)
        chosen = next(p for p in same if p['Ns']==preferred)
        selected.append(chosen)
    return selected


def tuning_windows(points: list[dict], c: float, nt: int) -> tuple[list[dict],list[dict],int | None]:
    """Fit every contiguous triple; only target-bracketing triples can tune."""
    records, details = [], []
    for start in range(len(points)-2):
        triple = points[start:start+3]
        beta = np.asarray([p['beta'] for p in triple])
        mnt = nt*np.asarray([p['mass'] for p in triple])
        sigma = np.asarray([p['mass_error']/p['mass'] for p in triple])
        mass_samples = nt*np.column_stack([p['_mass_samples'] for p in triple])
        logs = np.where(mass_samples > 0,np.log(np.maximum(mass_samples,1e-300)),np.nan)
        fit = line_fit(beta,np.log(mnt),sigma,logs)
        a,b = fit['parameters']
        params = fit['bootstrap_parameters']
        with np.errstate(divide='ignore',invalid='ignore'):
            roots = (np.log(c)-params[:,0])/params[:,1]
        bracketed = float(min(mnt)) <= c <= float(max(mnt))
        valid = np.isfinite(roots) & (params[:,1] != 0)
        # Replica extrapolation is diagnostic and retained, conditional on a
        # central-data bracket; failed slopes retain their original NaN slots.
        roots[~valid] = np.nan
        interval = np.percentile(roots[valid],[16,50,84]) if valid.sum()>=2 else [np.nan]*3
        central_root = (np.log(c)-a)/b if b != 0 else np.nan
        eligible = bracketed and b != 0 and valid.sum()>=2 and beta[0] <= central_root <= beta[-1]
        flags = sorted(set(flag for p in triple for flag in p['flags']))+fit['flags']
        if not bracketed:
            flags.append('target_not_bracketed')
        if not eligible:
            flags.append('tuning_unavailable')
        if b > 0:
            flags.append('positive_slope_review')
        if valid.sum() < 450:
            flags.append('bootstrap_success_below_90_percent')
        if any(mnt[i+1] > mnt[i]+2*np.hypot(triple[i]['mass_error']*nt,triple[i+1]['mass_error']*nt) for i in (0,1)):
            flags.append('mass_nonmonotonic_review')
        shifts=[]
        two_point=[]
        for i,j in combinations(range(3),2):
            if min(mnt[i],mnt[j]) <= c <= max(mnt[i],mnt[j]) and mnt[i] != mnt[j]:
                estimate = beta[i]+(np.log(c)-np.log(mnt[i]))*(beta[j]-beta[i])/(np.log(mnt[j])-np.log(mnt[i]))
                shifts.append(abs(float(estimate)-central_root))
                two_point.append(dict(beta_lo=beta[i],beta_hi=beta[j],beta_star=float(estimate)))
        if shifts and np.max(shifts) > .5*(interval[2]-interval[0]):
            flags.append('interpolation_shift_above_statistical_error')
        row = dict(target_c=c,physical_Nt=nt,window_index=start,betas=beta.tolist(),
                   ensembles=[p['ensemble'] for p in triple], bracketed=bracketed,available=bool(eligible),
                   beta_star=float(interval[1]) if eligible else None,
                   beta_star_central=float(central_root) if np.isfinite(central_root) else None,
                   stat_low=float(interval[0]) if eligible else None,
                   stat_high=float(interval[2]) if eligible else None,
                   beta_star_error=bootstrap_error(roots) if eligible else None,
                   interpolation_systematic=float(max(shifts)) if shifts and eligible else None,
                   intercept=float(a),slope=float(b),chi2_full=fit['chi2_full'],nominal_dof=1,
                   covariance_condition=fit['covariance_condition'],bootstrap_successes=int(valid.sum()),
                   bootstrap_extrapolated_fraction=float(np.sum(valid & ((roots<beta[0])|(roots>beta[-1])))/max(valid.sum(),1)),
                   bootstrap_slope_sign_change_fraction=float(np.mean(params[:,1]*b < 0)),
                   bootstrap_samples=500,flags=sorted(set(flags)),selected=False)
        records.append(row)
        details.append(dict(fit=fit,roots=roots,points=triple,two_point=two_point))
    candidates = [i for i,row in enumerate(records) if row['available']]
    selected = min(candidates,key=lambda i:(max(abs(np.log(p['mass']*nt/c)) for p in details[i]['points']),
                                               max(records[i]['betas'])-min(records[i]['betas']))) if candidates else None
    return records,details,selected


def plot_tuning(points: list[dict], rows: list[dict], details: list[dict], selected: int | None,
                output: Path, c: float, nt: int) -> None:
    """Show all data, all coupling windows and each window's residuals."""
    fig,ax=plt.subplots(figsize=FIGURE_SIZE)
    for i,p in enumerate(points):
        ax.errorbar([p['beta']],[p['mass']*nt],[p['mass_error']*nt], label=rf'$N_s={p["Ns"]},\ \beta={p["beta"]:g}$', **pu.data(1+i%9,markersize=9))
    ax.axhline(c,label=rf'target $c={c:g}$',**pu.fit(0))
    if selected is not None:
        d=details[selected]; beta=rows[selected]['betas']; x=np.linspace(min(beta),max(beta),200)
        a,b=d['fit']['parameters']; ax.plot(x,np.exp(a+b*x),label='suggested local interpolation',**pu.fit(2))
        params=d['fit']['bootstrap_parameters']; models=np.exp(params[:,0,None]+params[:,1,None]*x)
        lo,hi=np.nanpercentile(models,[16,84],axis=0); ax.fill_between(x,lo,hi,color=pu.color_dict[2],alpha=.15)
    finish(fig,ax,output/'tuning.pdf',r'$\beta$',r'$m_G N_t$',rf'$c={c:g},\ N_t={nt}$; preliminary')
    fig,ax=plt.subplots(figsize=FIGURE_SIZE)
    for i,row in enumerate(rows):
        if row['beta_star'] is not None:
            ax.errorbar([i+1],[row['beta_star']],[[row['beta_star']-row['stat_low']],[row['stat_high']-row['beta_star']]],
                        label='suggested' if i==selected else None,**pu.data(2 if i==selected else 1))
    ax.set_xticks(np.arange(len(rows))+1,[' / '.join(f'{b:g}' for b in row['betas']) for row in rows],rotation=25,fontsize=15)
    finish(fig,ax,output/'coupling_windows.pdf',r'coupling window ($\beta$ values)',r'$\beta_\star$',rf'$c={c:g},\ N_t={nt}$; bracketing windows',legend=selected is not None)
    for i,(row,detail) in enumerate(zip(rows,details)):
        fig,ax=plt.subplots(figsize=FIGURE_SIZE)
        error=np.asarray([p['mass_error']/p['mass'] for p in detail['points']])
        ax.errorbar(row['betas'],detail['fit']['residual']/error,np.ones(3),**pu.data(1))
        ax.axhline(0,**pu.fit(0))
        finish(fig,ax,output/f'coupling_window_{i}_residuals.pdf',r'$\beta$',r'$(\log(m_GN_t)-a-b\beta)/\sigma$',
               rf'$c={c:g},\ N_t={nt}$; window {i}; '+('bracketed' if row['bracketed'] else 'unbracketed'),legend=False)


def continuum_fit(rows: list[dict], c: float, drop_coarsest: bool = False) -> tuple[dict,dict]:
    """Fit Q=c beta*/(4 Nt) with p=1; only statistical errors in the band."""
    selected=sorted(rows,key=lambda r:r['physical_Nt'])[int(drop_coarsest):]
    nt=np.asarray([r['physical_Nt'] for r in selected],dtype=np.float64)
    y=np.asarray([c*r['beta_star']/(4*r['physical_Nt']) for r in selected])
    replicas=np.column_stack([c*r['_roots']/(4*r['physical_Nt']) for r in selected])
    error=np.asarray([bootstrap_error(col) for col in replicas.T])
    fit=line_fit(1/nt,y,error,replicas)
    q0=fit['bootstrap_parameters'][:,0]; valid=np.isfinite(q0); interval=np.percentile(q0[valid],[16,50,84])
    flags=sorted(set(f for r in selected for f in r['flags']))
    if len(nt)==2:
        flags.append('zero_residual_dof')
    flags.append('preliminary_conditional_on_windows_and_volumes')
    row=dict(target_c=c,p=1,fit='drop_coarsest' if drop_coarsest else 'all_Nt',Nt_values=nt.astype(int).tolist(),
              Q0=float(fit['parameters'][0]),Q0_error=bootstrap_error(q0),Q0_median=float(interval[1]),
              stat_low=float(interval[0]),stat_high=float(interval[2]),slope=float(fit['parameters'][1]),
              beta_over_Nt_continuum=float(4*fit['parameters'][0]/c),
              beta_over_Nt_continuum_error=float(4*bootstrap_error(q0)/c),
              chi2_full=fit['chi2_full'],nominal_dof=len(nt)-2,bootstrap_samples=500,
              bootstrap_successes=int(valid.sum()),flags=flags)
    # A conservative linear-response bound, allowing independently signed
    # two-point interpolation shifts at every resolution. Keep this separate
    # from the bootstrap statistical error, with no quadrature combination.
    design=np.column_stack((np.ones_like(nt),1/nt))
    response=linalg.lstsq(design/error[:,None],np.diag(1/error))[0][0]
    row['Q0_interpolation_shift_bound']=float(np.sum(np.abs(response*c/(4*nt))*
        np.asarray([r.get('interpolation_systematic') or 0 for r in selected])))
    return row,dict(fit=fit,nt=nt,y=y,error=error,replicas=replicas)


def aggregate() -> None:
    """Produce the single active campaign tables and physics comparisons."""
    manifest=load_campaign_manifest()
    reviews=read_review(manifest)
    results=load_ensembles(manifest,reviews)
    summaries=[]
    for p in results:
        fit=p['fit_windows'][p['selected_window_index']] if p['selected_window_index'] is not None else {}
        summaries.append(dict(ensemble=p['ensemble'],target_c=p['target_c'],reference_c=p['reference_c'],
                              campaign_type=p['campaign_type'],fixed_c_continuum_input=p['fixed_c_continuum_input'],
                              physical_Nt=p['physical_Nt'],Ns=p['Ns'],beta=p['beta'],
                              chains=len(p['chains']),total_sweeps=sum(c['sweeps'] for c in p['chains']),
                              retained_sweeps=sum(c['retained_sweeps'] for c in p['chains']),used_sweeps=sum(c['used_sweeps'] for c in p['chains']),
                              block_length_sweeps=p['block_length_sweeps'],min_blocks=min(c['n_complete_blocks'] for c in p['chains']),
                              mass=p['mass'],mass_error=p['mass_error'],mass_Nt=p['mass']*p['physical_Nt'] if p['mass'] else None,
                              mass_Nt_error=p['mass_error']*p['physical_Nt'] if p['mass_error'] is not None else None,
                              mass_Ns=p['mass']*p['Ns'] if p['mass'] else None,
                              xi_2nd=p['xi_2nd'],xi_2nd_error=p['xi_2nd_error'],
                              xi_2nd_squared=p['xi_2nd']**2 if p['xi_2nd'] is not None else None,
                              xi_2nd_squared_error=squared_length_error(p),
                              Ns_over_xi_2nd=p['Ns']/p['xi_2nd'] if p['xi_2nd'] else None,
                              r_min=fit.get('r_min'),r_max=fit.get('r_max'),chi2_full=fit.get('chi2_full'),nominal_dof=fit.get('nominal_dof'),
                              bootstrap_samples=500,include=p['include'],review_status=p['review_status'],flags=p['flags']))
        write_csv(RESULTS/p['ensemble']/'ensembles.csv',[summaries[-1]])
    write_csv(RESULTS/'ensembles.csv',summaries)
    for name in ('chains','autocorrelations','correlators','fit_windows'):
        write_csv(RESULTS/f'{name}.csv',[dict(r, campaign_type=p['campaign_type'],
            target_c=p['target_c'], reference_c=p['reference_c'], physical_Nt=p['physical_Nt'],
            Ns=p['Ns'], beta=p['beta'], fixed_c_continuum_input=p['fixed_c_continuum_input'])
            for p in results for r in read_csv(RESULTS/p['ensemble']/f'{name}.csv')])
    write_csv(RESULTS/'comparison/nt1_ensembles.csv',
              [r for r in summaries if r['campaign_type']=='nt1_matched_beta'])
    tunings=[]
    fixed_results=[p for p in results if p['fixed_c_continuum_input']]
    for (c,nt),iterator in groupby(sorted(fixed_results,key=lambda p:(p['target_c'],p['physical_Nt'])),key=lambda p:(p['target_c'],p['physical_Nt'])):
        all_points=list(iterator)
        points=primary_volumes([p for p in all_points if p['include']])
        key=f'c_{token(c)}/Nt_{nt}'; output=RESULTS/key/'comparison'; output.mkdir(parents=True,exist_ok=True)
        records,details,selected=tuning_windows(points,c,nt)
        review=reviews.get(('tuning',key),{})
        if review.get('betas'):
            betas=[float(x) for x in review['betas'].split(';')]
            matches=[i for i,r in enumerate(records) if r['betas']==betas and r['available']]
            if len(matches)!=1:
                raise ValueError(f'{key}: manual beta window must be an available bracketing triple')
            selected=matches[0]
        if selected is not None:
            records[selected]['selected']=True
            chosen=dict(records[selected],include=review.get('decision')!='exclude',review_status=review.get('decision','auto'),
                        _roots=details[selected]['roots'])
            tunings.append(chosen)
        else:
            tunings.append(dict(target_c=c,physical_Nt=nt,include=False,beta_star=None,flags=['tuning_unavailable']))
        write_csv(output/'interpolation_windows.csv',clean(records))
        write_json(output/'tuning.json',clean({k:v for k,v in tunings[-1].items() if not k.startswith('_')}))
        np.savez_compressed(output/'tuning_bootstrap.npz',beta_star=np.asarray([d['roots'] for d in details]),
                            parameters=np.asarray([d['fit']['bootstrap_parameters'] for d in details]))
        plot_tuning(points,records,details,selected,output,c,nt)
        plot_comparisons(all_points,output)
    write_csv(RESULTS/'tuning.csv',clean([{k:v for k,v in r.items() if not k.startswith('_')} for r in tunings]))
    continuum=[]; continuum_details=[]
    for c in sorted({r['target_c'] for r in tunings}):
        group=[r for r in tunings if r['target_c']==c and r.get('include')]
        if len(group)<2:
            continue
        output=RESULTS/f'c_{token(c)}'/'comparison'; output.mkdir(parents=True,exist_ok=True)
        pairs=[continuum_fit(group,c)]
        if len(group)>=3:
            pairs.append(continuum_fit(group,c,True))
            shift=pairs[1][0]['Q0']-pairs[0][0]['Q0']
            shifted_replicas=(pairs[1][1]['fit']['bootstrap_parameters'][:,0]
                              -pairs[0][1]['fit']['bootstrap_parameters'][:,0])
            shift_error=bootstrap_error(shifted_replicas)
            for row,_ in pairs:
                row['drop_coarsest_shift']=float(shift)
                row['drop_coarsest_shift_error']=shift_error
                if shift_error is not None and abs(shift)>2*shift_error:
                    row['flags'].append('continuum_resolution_range_sensitivity')
        for row,detail in pairs:
            continuum.append(row); continuum_details.append(detail)
        write_csv(output/'continuum.csv',[r for r,d in pairs])
        np.savez_compressed(output/'continuum_bootstrap.npz',**{r['fit']:d['fit']['bootstrap_parameters'] for r,d in pairs})
        plot_continuum(pairs,output,c)
    write_csv(RESULTS/'continuum.csv',continuum)
    plot_all_targets(continuum,continuum_details,summaries)
    plot_matched_nt1([p for p in results if p['campaign_type']=='nt1_matched_beta'])
    diagnostics = [p for p in results if p['campaign_type']=='matched_y_scaling']
    for (c, nt), group in groupby(sorted(diagnostics, key=lambda p:(p['reference_c'],p['physical_Nt'])),
                                key=lambda p:(p['reference_c'],p['physical_Nt'])):
        plot_comparisons(list(group), RESULTS/f'c_{token(c)}/Nt_{nt}/comparison')
    plot_scaling(results, RESULTS)
    write_readmes(manifest,results,tunings,continuum)
    write_json(RESULTS/'campaign.json',dict(campaign_type='nonperturbative_mg_nt',bootstrap_samples=500,
               ensembles=len(results),chains=sum(len(p['chains']) for p in results),
               fixed_c_ensembles=len(fixed_results),
               matched_beta_ensembles=sum(p['campaign_type']=='nt1_matched_beta' for p in results),
               matched_y_ensembles=sum(p['campaign_type']=='matched_y_scaling' for p in results),
               total_sweeps=sum(r['total_sweeps'] for r in summaries),
               retained_sweeps=sum(r['retained_sweeps'] for r in summaries),used_sweeps=sum(r['used_sweeps'] for r in summaries),
               length_outlier_chains=sum('length_outlier' in c['flags'] for r in results for c in r['chains']),
               numerical_masses=sum(r['mass'] is not None for r in results),tuning_points=sum(r.get('include',False) for r in tunings),
               continuum_targets=len({r['target_c'] for r in continuum}),
               manifest_sha256=sha256(ROOT/'data/mg_nt_fixed/manifest.json'),review_sha256=sha256(RESULTS/'review.csv'),
               source_manifest_sha256=manifest['source_manifest_sha256'],
               analysis_jobs=sorted({r['provenance']['job_id'] for r in results}),report_job=os.environ.get('SLURM_JOB_ID')))


def plot_matched_nt1(points: list[dict]) -> None:
    """Show the untuned Nt=1 measurements separately from fixed-c fits."""
    if not points:
        return
    points = sorted(points, key=lambda p:p['beta'])
    output = RESULTS/'comparison'
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for i, p in enumerate(points):
        r = np.asarray(p['R']); keep = r > 0
        ax.errorbar(r[keep], np.asarray(p['C_G'])[keep], np.asarray(p['C_G_error'])[keep],
            label=rf'$\beta={p["beta"]:.6f},\ N_s={p["Ns"]}$', **pu.data(1+i, markersize=4))
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=max(max(p['C_G_error']) for p in points))
    # Omit crowded +/-1e-4 tick labels around the narrow linear region.
    ax.set_yticks([-1e-2, -1e-3, 0., 1e-3, 1e-2, 1e-1, 1.])
    finish(fig, ax, output/'nt1_correlators.pdf', r'$R/a$', r'$C_G(R)$',
           r'$N_t=1$: couplings matched to $N_t=2$; untuned screening masses')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    defined = [p for p in points if p['mass'] is not None]
    for i, p in enumerate(defined):
        ax.errorbar([p['beta']], [p['mass']], [p['mass_error']],
                    label=rf'$N_s={p["Ns"]},\ c_{{\rm ref}}={p["reference_c"]:g}$', **pu.data(1+i))
    finish(fig, ax, output/'nt1_masses.pdf', r'$\beta$', r'$am_G$ (bootstrap SE)',
           r'$N_t=1$: preliminary masses; excluded from fixed-$c$ continuum fits', legend=bool(defined))


def plot_comparisons(points: list[dict], output: Path) -> None:
    """Keep equal-volume correlators and matched-volume masses distinguishable."""
    for ns in sorted({p['Ns'] for p in points}):
        same=[p for p in points if p['Ns']==ns]
        folder=output.parent/f'Ns_{ns}'/'comparison'
        fig,ax=plt.subplots(figsize=FIGURE_SIZE)
        for i,p in enumerate(same):
            r=np.asarray(p['R']); keep=r>0
            ax.errorbar(r[keep],np.asarray(p['C_G'])[keep],np.asarray(p['C_G_error'])[keep],
                        label=rf'$\beta={p["beta"]:g}$',**pu.data(1+i%9,markersize=5))
        ax.set_yscale('symlog',linthresh=max(max(p['C_G_error']) for p in same))
        c_label = (rf'c={same[0]["target_c"]:g}' if same[0]['target_c'] is not None
                   else rf'c_{{\rm ref}}={same[0]["reference_c"]:g}')
        finish(fig,ax,folder/'correlators.pdf',r'$R/a$',r'$C_G(R)$',rf'${c_label},\ N_t={same[0]["physical_Nt"]},\ N_s={ns}$')
    for beta in sorted({p['beta'] for p in points}):
        same=[p for p in points if p['beta']==beta and p['mass'] is not None]
        if len(same)<2:
            continue
        fig,ax=plt.subplots(figsize=FIGURE_SIZE)
        ax.errorbar([p['Ns'] for p in same],[p['mass']*p['physical_Nt'] for p in same],
                    [p['mass_error']*p['physical_Nt'] for p in same],**pu.data(1))
        finish(fig,ax,output/f'volume_beta_{token(beta)}.pdf',r'$N_s$',r'$m_G N_t$',rf'$\beta={beta:g}$; matched-volume comparison',legend=False)
        reference=min(same,key=lambda p:p['Ns'])
        write_csv(output/f'volume_beta_{token(beta)}.csv',
                  [dict(ensemble=p['ensemble'],Ns=p['Ns'],mass=p['mass'],mass_error=p['mass_error'],
                        mass_Ns=p['mass']*p['Ns'],reference_Ns=reference['Ns'],
                        mass_difference=p['mass']-reference['mass'],
                        difference_combined_sigma=abs(p['mass']-reference['mass'])/np.hypot(p['mass_error'],reference['mass_error']),
                        flags=p['flags']) for p in same])
    for p in points:
        if len(p['chains'])!=8:
            continue
        nr=len(p['R']); boot=p['_boot']; fit=p['fit_windows'][p['selected_window_index']]
        rows=[]; draws=[]
        for label,indices in [('short_only',[i for i,c in enumerate(p['chains']) if c['sweeps']==8000]),
                              ('long_only',[i for i,c in enumerate(p['chains']) if c['sweeps']==80000]),
                              ('pooled',list(range(8)))]:
            weights=boot['chain_weights'][indices]; weights=weights/weights.sum()
            mean=np.einsum('i,ij->j',weights,boot['chain_means'][indices])
            samples=np.einsum('i,ibj->bj',weights,boot['chain_samples'][indices])
            # The three comparison estimators share these same per-chain
            # draws, including the pooled estimator. This preserves their
            # joint covariance; the main mass uses its recorded primary RNG
            # stream, so its finite-500 SE can differ slightly.
            row,params=fit_window(np.asarray(p['R']),mean[4:4+nr],samples[:,4:4+nr],
                                  np.cov(samples[:,4:4+nr],rowvar=False,ddof=1),p['Ns'],fit['r_min'],fit['r_max'])
            rows.append(dict(sample=label,mass=row['mass'],mass_error=row['mass_error'],r_min=fit['r_min'],r_max=fit['r_max']))
            draws.append(params)
        folder=RESULTS/p['ensemble']/'plots'
        fig,ax=plt.subplots(figsize=FIGURE_SIZE)
        ax.errorbar(range(3),[r['mass'] for r in rows],[r['mass_error'] for r in rows],**pu.data(1))
        ax.set_xticks(range(3),['4 short chains','4 long chains','pooled (weighted)'],fontsize=20)
        finish(fig,ax,folder/'short_long_pooled.pdf','independent-chain groups',r'$m_G a$',title(p),legend=False)
        write_csv(RESULTS/p['ensemble']/'short_long_pooled.csv',rows)
        np.savez_compressed(RESULTS/p['ensemble']/'short_long_bootstrap.npz',parameters=np.asarray(draws))


def plot_continuum(pairs: list[tuple], output: Path, c: float) -> None:
    """Single-target p=1 central curves and statistical bands."""
    for quantity,scale in [('mass_in_coupling_units',1.),('beta_over_Nt',4/c)]:
        fig,ax=plt.subplots(figsize=FIGURE_SIZE)
        detail=pairs[0][1]
        ax.errorbar(1/detail['nt'],scale*detail['y'],scale*detail['error'],label='all numerical tuning points (flags in CSV)',**pu.data(1))
        for i,(row,d) in enumerate(pairs):
            x=np.linspace(0,max(1/d['nt']),200); params=d['fit']['parameters']; bp=d['fit']['bootstrap_parameters']
            ax.plot(x,scale*(params[0]+params[1]*x),label='all resolutions' if i==0 else 'drop coarsest',**pu.fit(2+i))
            curves=scale*(bp[:,0,None]+bp[:,1,None]*x); lo,hi=np.nanpercentile(curves,[16,84],axis=0)
            ax.fill_between(x,lo,hi,color=pu.color_dict[2+i],alpha=.12)
        finish(fig,ax,output/f'continuum_{quantity}_p1.pdf',r'$1/N_t$',r'$m_G/g_{3,0}^2=c\beta_\star/(4N_t)$' if scale==1 else r'$\beta_\star/N_t$',rf'$c={c:g}$; preliminary $p=1$ continuum fit')


def plot_all_targets(rows: list[dict],details: list[dict],summaries: list[dict]) -> None:
    """Root comparisons combine targets; preserve target-specific scales."""
    folder=RESULTS/'comparison'
    fig,ax=plt.subplots(figsize=FIGURE_SIZE)
    for i,(row,d) in enumerate((r,d) for r,d in zip(rows,details) if r['fit']=='all_Nt'):
        c=row['target_c'];x=np.linspace(0,max(1/d['nt']),200);p=d['fit']['parameters']
        ax.errorbar(1/d['nt'],d['y'],d['error'],label=rf'$c={c:g}$',**pu.data(1+i))
        ax.plot(x,p[0]+p[1]*x,**pu.fit(1+i))
    finish(fig,ax,folder/'continuum_all_targets_p1.pdf',r'$1/N_t$',r'$m_G/g_{3,0}^2$',r'Fixed $m_GN_t$: preliminary $p=1$ extrapolations')
    fig,ax=plt.subplots(figsize=FIGURE_SIZE)
    for i,c in enumerate(sorted({r['reference_c'] for r in summaries})):
        selected=[(j,r) for j,r in enumerate(summaries) if r['reference_c']==c]
        ax.plot([j+1 for j,r in selected],[r['retained_sweeps'] for j,r in selected],label=rf'$c\ {{\rm or}}\ c_{{\rm ref}}={c:g}$',**pu.data_noerr(1+i))
    ax.set_yscale('log')
    finish(fig,ax,folder/'statistics_inventory.pdf','ensemble row in ensembles.csv','post-cut measured sweeps','Statistics inventory; untuned points use reference targets; long chains flagged')


def write_readmes(manifest: dict,results: list[dict],tunings: list[dict],continuum: list[dict]) -> None:
    """Give every physics level a small index and relative plot links."""
    # Filtered summaries at each physics level let colleagues inspect a
    # target/resolution without manually filtering the all-campaign CSV.
    for name in ('ensembles','chains','autocorrelations','correlators','fit_windows'):
        rows=read_csv(RESULTS/f'{name}.csv')
        for target in sorted(RESULTS.glob('c_*')):
            levels=[target]+list(target.glob('Nt_*'))+list(target.glob('Nt_*/Ns_*'))
            for directory in levels:
                prefix=str(directory.relative_to(RESULTS))+'/'
                write_csv(directory/f'{name}.csv',[row for row in rows if row['ensemble'].startswith(prefix)])
    for root in (RESULTS,ROOT/'data/mg_nt_fixed'):
        directories={root}
        for group in manifest['ensembles']:
            leaf=root/group['ensemble']
            directories.update([leaf,leaf.parent,leaf.parent.parent,leaf.parent.parent.parent])
        if root==RESULTS:
            directories.update(p for p in root.glob('c_*/**/comparison') if p.is_dir())
            directories.add(root/'comparison')
        for directory in sorted(directories):
            directory.mkdir(parents=True,exist_ok=True)
            relative=str(directory.relative_to(root))
            lines=[f'# {"Fixed mG Nt" if relative=="." else relative}','']
            if root==RESULTS:
                same=next((p for p in results if p['ensemble']==relative),None)
                if same:
                    lines += [f"mG a = {same['mass']} ± {same['mass_error']} (500-replica bootstrap).",
                              f"Review: {same['review_status']}. Flags: {', '.join(same['flags']) or 'none' }.",
                              '','[Correlator and window plots](plots/) · [Cuts and blocking](plots/first_stage/).','']
                    if not same['fixed_c_continuum_input']:
                        role = 'Nt=1 matched-beta' if same['campaign_type']=='nt1_matched_beta' else 'Matched-beta/Nt'
                        lines += [f"{role} diagnostic. Folder c={same['reference_c']} is a reference family; target_c=null. Excluded from fixed-c continuum fits.",'']
                if relative=='.':
                    lines += [f"{manifest['ensemble_count']} physical ensembles; {manifest['chain_count']} independent chains; exactly 500 replicas per operation.",
                              'Nt=1 matched-beta and matched-beta/Nt diagnostics have target_c=null; their folder c is a reference family. They do not enter fixed-c tuning or continuum fits.',
                              'All numerical results are preliminary and available for visual review. Full-covariance chi-square is descriptive.',
                              'Edit review.csv to choose mass windows or exclude points, then rerun the campaign collector and logbook on qbig.',
                              'The four 80,000-sweep chains carry length_outlier; eight prior-input chains have 10,000 sweeps.',
                              '','[Logbook](../../docs/mg_nt_fixed/logbook.pdf) · [CSV column guide](../../docs/mg_nt_fixed/csv_columns.md).','']
                if relative=='comparison':
                    lines += ['All simulations are included in the existing top-level tables:',
                              '[Ensembles](../ensembles.csv) · [Correlators](../correlators.csv) · '
                              '[Chains](../chains.csv) · [Autocorrelations](../autocorrelations.csv) · '
                              '[All mass windows](../fit_windows.csv).',
                              'Untuned diagnostics retain their explicit campaign role; fixed-c tuning uses only its tuned input set.',
                              'All nine scaling plots use beta/Nt on the horizontal axis, the same Nt palette, open markers and Nt-only legends.',
                              'For p=0,1,3/2 the ordinates are amG*Nt^p, (xi2nd/a)/Nt^p and (xi2nd/a)^2/Nt^(2p).',
                              'Plot titles, footer notes and grids are omitted; flags and campaign roles remain in the tables and logbook.', '',
                              '| Observable | Unscaled | p=1 | p=3/2 |',
                              '|---|---|---|---|']
                    for i, label in enumerate(('amG', 'xi2nd/a', '(xi2nd/a)^2')):
                        links = [f'[PDF]({name})' for name in SCALING_PDFS[3*i:3*i+3]]
                        lines.append('| '+label+' | '+' | '.join(links)+' |')
                    lines += ['', 'Other comparisons and data:', '']
            else:
                lines += ['Immutable input copies and provenance. Raw histories remain on qbig at the original manifest paths.','']
                if relative=='.':
                    lines += ['[Immutable manifest](manifest.json) · [Simulation build](build_provenance.json) · [Migration map](migration.csv).',
                              '[Nt=1 analysis inventory](nt1_analysis_manifest.json): matched-beta diagnostics; folder c is the source Nt=2 reference target, and target_c is null.',
                              '[Matched-y inputs](scaling_manifest.json) · [Completed matched-y analysis inventory](scaling_analysis_manifest.json) · [Execution plan](scaling_execution.json) · [Scheduler completion](scaling_schedule_status.json).',
                              '[Nt=1 matched-y inputs](nt1_scaling_manifest.json) · [Nt=1 matched-y analysis inventory](nt1_scaling_analysis_manifest.json).',
                              'The manifest records the checksum-verified qbig recovery archive. Historical stage directories are retired after product verification.','']
                if '/Nt_1' in relative:
                    lines += ['Nt=1 matched-beta or matched-beta/Nt diagnostic. The c folder identifies its source reference family; target_c=null and these inputs are excluded from fixed-c continuum fits.','']
                same = next((p for p in results if p['ensemble']==relative), None)
                if same and same['campaign_type']=='matched_y_scaling':
                    lines += ['Matched-beta/Nt diagnostic; target_c=null. The c folder is a reference family; excluded from fixed-c continuum fits.','']
            children=[p for p in sorted(directory.iterdir()) if p.is_dir() and (p.name.startswith(('c_','Nt_','Ns_','beta_','seed_')) or p.name=='comparison')]
            lines += [f'- [{p.name}]({p.name}/)' for p in children]
            lines += ['']+[f'- [{p.name}]({p.name})' for p in sorted(directory.glob('*.csv'))]
            lines += [f'- [{p.name}]({p.name})' for p in sorted(directory.glob('*.pdf'))
                      if not (root==RESULTS and relative=='comparison' and p.name in SCALING_PDFS)]
            (directory/'README.md').write_text('\n'.join(lines)+'\n')
    for p in results:
        for folder in (RESULTS/p['ensemble']/'plots',RESULTS/p['ensemble']/'plots/first_stage'):
            (folder/'README.md').write_text('# Plots\n\n'+'\n'.join(f'- [{f.name}]({f.name})' for f in sorted(folder.glob('*.pdf')))+'\n')


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('campaign numerical analysis must run on a qbig compute node')
    aggregate()


if __name__ == '__main__':
    main()
