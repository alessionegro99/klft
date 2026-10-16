"""Grid-free, one-plot-per-page figures for canonical mg_nt results."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from beta_nt_fixed.analysis import finite_volume_k0
from mg_nt_fixed_statistics import bootstrap_error
from plot_thesis_results import FIGURE_SIZE, plt, pu, save_figure


def title(result: dict) -> str:
    """Physics parameters, without historical stage or date labels."""
    label = (rf'c={result["target_c"]:g}' if result['target_c'] is not None else
             rf'c_{{\rm ref}}={result["reference_c"]:g}')
    return rf'${label},\ N_t={result["physical_Nt"]},\ N_s={result["Ns"]},\ \beta={result["beta"]:g}$'


def finish(fig: plt.Figure, ax: plt.Axes, path: Path, xlabel: str,
           ylabel: str, heading: str, legend: bool = True) -> None:
    """Keep axes and legends, with no displayed title or footer (user policy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.set_title('')
    for annotation in list(fig.texts):
        annotation.remove()
    pu.style_axes(ax, xlabel=xlabel, ylabel=ylabel, legend=legend,
                  grid=False, legend_kwargs={'fontsize': 18, 'loc': 'best'})
    ax.grid(False, which='both')
    save_figure(fig, path)


def plot_first_stage(output: Path, result: dict, raw: list, scan: dict) -> None:
    """Show every independent cut, bootstrap-error saturation and block count."""
    folder = output/'plots/first_stage'
    for row, chain in zip(result['chains'], raw):
        for name, values in [('Ctilde_0', chain.summary.values[:,0,2]),
                             ('C_G_R1', chain.correlator.values[:,1,0])]:
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            width = max(1, len(values)//800)
            n = len(values)//width
            x = chain.summary.steps[:n*width].reshape(n, width).mean(axis=1)
            y = values[:n*width].reshape(n, width).mean(axis=1)
            ax.plot(x, y, label=f'{width}-sweep display averages', **pu.fit(1, linestyle='-', linewidth=.8))
            ax.axvspan(0, row['thermalization_cut_sweeps'], color=pu.color_dict[2], alpha=.13)
            ax.axvline(row['thermalization_cut_sweeps'], label=f'cut: {row["thermalization_cut_sweeps"]} sweeps', **pu.fit(2))
            suffix = '; 80,000 sweeps: length outlier' if row['sweeps'] == 80000 else ''
            finish(fig, ax, folder/f'thermalization_seed_{row["seed"]}_{name}.pdf', 'composite sweep',
                   r'$\widetilde C(0)$' if name == 'Ctilde_0' else r'$C_G(1)$',
                   title(result)+f'\nseed {row["seed"]}{suffix}')
    labels = ['Tr_G_real', 'Ctilde_0', 'Ctilde_pmin', 'C_G_R1', 'xi_2nd']
    for name in labels:
        j = scan['labels'].index(name)
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.errorbar(scan['lengths'], scan['errors'][:,j], scan['uncertainties'][:,j],
                    label='500-replica bootstrap SE; approximate SE uncertainty', **pu.data(1, linestyle='-', markersize=7))
        ax.axvline(scan['selected'], label=f'chosen B={scan["selected"]}', **pu.fit(2))
        ax.axvline(scan['minimum_length'], label=r'$2\tau_{\rm int}$ lower bound (rounded)', **pu.fit(3))
        ax.set_ylim(bottom=0)
        finish(fig, ax, folder/f'blocking_{name}.pdf', 'block size (sweeps)',
               'bootstrap standard error', title(result)+'\n'+name.replace('_', r'\_'))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for i, row in enumerate(result['chains']):
        ax.plot(scan['lengths'], [n[i] for n in scan['counts']], label=f'seed {row["seed"]}', **pu.fit(1+i%9))
    ax.axhline(30, label='30-block warning threshold', **pu.fit(0))
    ax.axhline(50, label='50-block target', **pu.fit(3))
    ax.axvline(scan['selected'], **pu.fit(2))
    ax.set_yscale('log')
    finish(fig, ax, folder/'blocking_counts.pdf', 'block size (sweeps)', 'complete blocks per chain', title(result))


def plot_ensemble(output: Path, result: dict, boot: object,
                  mass_scan: list[dict] | None = None, scan: dict | None = None) -> None:
    """Display the chosen fit and every window's mass, including flagged fits."""
    folder = output/'plots'
    r, y, err = (np.asarray(result[k], dtype=np.float64) for k in ('R', 'C_G', 'C_G_error'))
    selected = result['selected_window_index']
    fit = result['fit_windows'][selected] if selected is not None else None
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    positive = r > 0
    ax.errorbar(r[positive], y[positive], err[positive], label='pooled correlator; bootstrap errors', **pu.data(1, markersize=7))
    if fit:
        x = np.linspace(fit['r_min'], fit['r_max'], 200)
        model = finite_volume_k0(x, fit['amplitude'], fit['mass'], result['Ns'])
        ax.plot(x, model, label=f'window {fit["r_min"]}--{fit["r_max"]}; preliminary', **pu.fit(2))
        params = boot['window_parameters'][selected]
        valid = np.all(np.isfinite(params), axis=1)
        models = np.asarray([finite_volume_k0(x, a, m, result['Ns']) for a, m in params[valid]])
        if len(models) >= 2:
            lo, hi = np.percentile(models, [16, 84], axis=0)
            ax.fill_between(x, lo, hi, color=pu.color_dict[2], alpha=.2)
        ax.axvspan(fit['r_min'], fit['r_max'], color=pu.color_dict[3], alpha=.06)
    ax.set_yscale('symlog', linthresh=max(float(np.max(err[positive])), 1e-12))
    finish(fig, ax, folder/'correlator_fit.pdf', r'$R/a$', r'$C_G(R)$', title(result))
    if fit:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        model = finite_volume_k0(r[positive], fit['amplitude'], fit['mass'], result['Ns'])
        ax.errorbar(r[positive], (y[positive]-model)/err[positive], np.ones(sum(positive)),
                    label='marginal-error standardized residuals (correlated)', **pu.data(1, markersize=7))
        ax.axhline(0, **pu.fit(0))
        ax.axvspan(fit['r_min'], fit['r_max'], color=pu.color_dict[3], alpha=.12)
        chi = f'{fit["chi2_full_per_dof"]:.3g}' if fit['chi2_full_per_dof'] is not None else 'undefined'
        finish(fig, ax, folder/'fit_residuals.pdf', r'$R/a$', r'$(C_G-C_{\rm fit})/\sigma_R$',
               title(result)+'\n'+rf'full-covariance $\chi^2/\nu={chi}$ (descriptive)')
    for high in sorted({row['r_max'] for row in result['fit_windows']}, reverse=True):
        rows = [row for row in result['fit_windows'] if row['r_max'] == high]
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        good = [row for row in rows if row['mass_error'] is not None]
        ax.errorbar([row['r_min'] for row in good], [row['mass']*result['physical_Nt'] for row in good],
                    [row['mass_error']*result['physical_Nt'] for row in good],
                    label='all numerical fits; 500-replica errors', **pu.data(1, markersize=7))
        if fit:
            ax.axhspan((fit['mass']-fit['mass_error'])*result['physical_Nt'],
                       (fit['mass']+fit['mass_error'])*result['physical_Nt'], color=pu.color_dict[2], alpha=.15,
                       label='suggested window mass')
            ax.axvline(fit['r_min'], **pu.fit(2))
        failed = [row['r_min'] for row in rows if row['mass_error'] is None]
        if failed:
            ax.plot(failed, np.zeros(len(failed)), label='undefined fits; see CSV failure text', **pu.data_noerr(3))
        finish(fig, ax, folder/f'mass_windows_Rmax_{high}.pdf', r'$R_{\min}/a$', r'$m_G N_t$',
               title(result)+'\n'+rf'$R_{{\max}}={high}$; no goodness-of-fit veto')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    length = (result['physical_Nt']/result['target_c'] if result['target_c'] is not None
              else result.get('xi_2nd') or 1)
    radius = max(1, min(len(r)-1, int(round(length))))
    idx = 4+radius
    ax.errorbar(np.arange(len(result['chains']))+1, boot['chain_means'][:,idx], boot['chain_errors'][:,idx],
                label=rf'$C_G({int(r[radius])})$; independent-chain bootstrap', **pu.data(1))
    ax.axhspan(y[radius]-err[radius], y[radius]+err[radius], color=pu.color_dict[2], alpha=.15, label='pooled estimate')
    ax.set_xticks(np.arange(len(result['chains']))+1,
                  [str(c['seed']) for c in result['chains']], rotation=30, fontsize=17)
    finish(fig, ax, folder/'chain_consistency.pdf', 'chain seed', rf'$C_G({int(r[radius])})$', title(result))
    if mass_scan is not None and scan is not None and mass_scan:
        for observable, column in [('mass', 1), ('xi_exp', 1)]:
            params = boot['mass_blocking_parameters'][:,:,column]
            if observable == 'xi_exp':
                params = 1/params
            errors = [bootstrap_error(v) for v in params]
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.errorbar(scan['lengths'], errors, label='500-replica SE; fixed suggested mass window', **pu.data(1, linestyle='-', markersize=7))
            ax.axvline(scan['selected'], label=f'chosen B={scan["selected"]}', **pu.fit(2))
            ax.set_ylim(bottom=0)
            finish(fig, ax, folder/f'first_stage/blocking_{observable}.pdf', 'block size (sweeps)',
                   'bootstrap standard error', title(result))
