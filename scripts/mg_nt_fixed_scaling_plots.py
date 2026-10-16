"""Matched-y comparisons from saved 500-slot replicas; no new resampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mg_nt_fixed_canonical_plots import finish
from mg_nt_fixed_inventory import write_csv
from mg_nt_fixed_statistics import bootstrap_error
from plot_thesis_results import FIGURE_SIZE, plt, pu


def effective_powers(points: list[dict]) -> list[dict]:
    """Adjacent-resolution log ratios at fixed y, with joint replica errors.

    If m(Nt)=A*Nt**(-alpha), log(m1/m2)/log(Nt2/Nt1)=alpha.
    Here this identity defines a finite-spacing diagnostic, without fitting
    a scaling law. Independent ensembles retain their aligned 500 draw slots.
    """
    points = sorted(points, key=lambda p:p['physical_Nt'])
    rows = []
    for low, high in zip(points[:-1], points[1:]):
        nt1, nt2 = low['physical_Nt'], high['physical_Nt']
        if nt1 == nt2 or not np.isclose(low['beta']/nt1, high['beta']/nt2, atol=1.e-12, rtol=0):
            raise ValueError('effective powers require distinct Nt at the same beta/Nt')
        b1, b2 = (np.asarray(p['_mass_samples']) for p in (low, high))
        if b1.shape != (500,) or b2.shape != (500,):
            raise ValueError('effective powers require 500 aligned slots')
        valid = np.isfinite(b1) & np.isfinite(b2) & (b1 > 0) & (b2 > 0)
        samples = np.full(500, np.nan)
        denominator = np.log(nt2/nt1)
        samples[valid] = np.log(b1[valid]/b2[valid])/denominator
        rows.append(dict(ensemble_low=low['ensemble'], ensemble_high=high['ensemble'],
            beta_over_Nt=low['beta']/nt1, Nt_low=nt1, Nt_high=nt2,
            effective_power=float(np.log(low['mass']/high['mass'])/denominator),
            bootstrap_error=bootstrap_error(samples), bootstrap_samples=500,
            bootstrap_successes=int(valid.sum()),
            flags=sorted(set(low['flags']+high['flags']+['finite_spacing_diagnostic']))))
    return rows


def plot_scaling(results: list[dict], root: Path) -> None:
    """Update the visual collapse and explicit matched-y/volume diagnostics."""
    if not any(p['campaign_type']=='matched_y_scaling' for p in results):
        return
    points = [p for p in results if p['physical_Nt'] >= 2 and p['include']
              and p['mass'] is not None and p['mass_error'] is not None]
    for exponent, name in ((0., 'mass_vs_beta_over_Nt'), (1.5, 'mass_scaling_Nt_power_1p5')):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for i, nt in enumerate(sorted({p['physical_Nt'] for p in points}), start=1):
            same = sorted((p for p in points if p['physical_Nt']==nt), key=lambda p:p['beta'])
            ax.errorbar([p['beta']/nt for p in same], [p['mass']*nt**exponent for p in same],
                [p['mass_error']*nt**exponent for p in same], label=rf'$N_t={nt}$', **pu.data(i))
            new = [p for p in same if p['campaign_type']=='matched_y_scaling']
            ax.errorbar([p['beta']/nt for p in new], [p['mass']*nt**exponent for p in new],
                [p['mass_error']*nt**exponent for p in new],
                **pu.data(i, markerfacecolor=pu.color_dict[i]))
        heading = ('Preliminary masses; filled markers: matched-y batch' if exponent==0 else
                   r'Trial $N_t^{1.5}$ collapse (exponent chosen, not fitted); filled: new batch')
        finish(fig, ax, root/'comparison'/f'{name}.pdf', r'$\beta/N_t$',
               r'$am_G$' if exponent==0 else r'$am_G N_t^{1.5}$', heading)

    # This comparison belongs under the shared reference family, c_ref=0.5.
    matched = sorted([p for p in points if p['reference_c']==.5
                      and np.isclose(p['beta']/p['physical_Nt'], 1.5, atol=1.e-12, rtol=0)],
                     key=lambda p:p['physical_Nt'])
    plot_matched_y(matched, root)
    volumes = sorted([p for p in points if p['reference_c']==.5 and p['physical_Nt']==4
                      and np.isclose(p['beta'], 6.2, atol=1.e-12, rtol=0)], key=lambda p:p['Ns'])
    plot_volume(volumes, root)


def plot_matched_y(matched: list[dict], root: Path) -> None:
    """Compare defined masses and correlators at y=1.5, without a cutoff fit."""
    if len(matched) < 2:
        return
    folder = root/'c_0p5/comparison'
    write_csv(folder/'matched_y_ensembles.csv', [dict(ensemble=p['ensemble'],
        campaign_type=p['campaign_type'], target_c=p['target_c'],
        fixed_c_continuum_input=p['fixed_c_continuum_input'],
        Nt=p['physical_Nt'], Ns=p['Ns'], beta=p['beta'], beta_over_Nt=1.5,
        mass=p['mass'], mass_error=p['mass_error'], mass_Nt=p['mass']*p['physical_Nt'],
        mass_Nt_error=p['mass_error']*p['physical_Nt'], mass_Ns=p['mass']*p['Ns'],
        bootstrap_samples=500, flags=p['flags']) for p in matched])
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for i, p in enumerate(matched, start=1):
        nt = p['physical_Nt']
        ax.errorbar([1/nt], [p['mass']*nt], [p['mass_error']*nt],
                    label=rf'$N_t={nt},\ N_s={p["Ns"]}$', **pu.data(i))
    finish(fig, ax, folder/'matched_y_mass_Nt.pdf', r'$1/N_t$', r'$m_G L=am_G N_t$',
           r'$\beta/N_t=1.5$: measured masses; no fixed-$c$ tuning')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for i, p in enumerate(matched, start=1):
        r = np.asarray(p['R']); keep = r > 0
        ax.errorbar(r[keep]/p['physical_Nt'], np.asarray(p['C_G'])[keep],
                    np.asarray(p['C_G_error'])[keep],
                    label=rf'$N_t={p["physical_Nt"]},\ N_s={p["Ns"]}$', **pu.data(i, markersize=4))
    ax.set_yscale('symlog', linthresh=max(max(p['C_G_error']) for p in matched))
    finish(fig, ax, folder/'matched_y_correlators.pdf', r'$R/L=r/N_t$', r'$C_G(r)$',
           r'$\beta/N_t=1.5$: correlators at matched bare slab coupling')
    powers = effective_powers(matched)
    write_csv(folder/'matched_y_effective_powers.csv', powers)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.errorbar(range(len(powers)), [p['effective_power'] for p in powers],
                [p['bootstrap_error'] for p in powers], label='adjacent-resolution log ratio', **pu.data(1))
    ax.axhline(1.5, label='trial exponent 1.5', **pu.fit(2))
    ax.set_xticks(range(len(powers)), [rf'${p["Nt_low"]}\to{p["Nt_high"]}$' for p in powers])
    finish(fig, ax, folder/'matched_y_effective_powers.pdf', r'$N_t$ pair', r'$\alpha_{\rm eff}$',
           r'$\beta/N_t=1.5$: finite-spacing exponent diagnostic')


def plot_volume(volumes: list[dict], root: Path) -> None:
    """Compare defined independent-volume estimates at Nt=4, beta=6.2."""
    if len(volumes) < 2:
        return
    folder = root/'c_0p5/Nt_4/comparison'
    reference = volumes[0]
    rows = []
    for p in volumes:
        differences = p['_mass_samples']-reference['_mass_samples']
        difference_error = bootstrap_error(differences)
        rows.append(dict(ensemble=p['ensemble'], reference_ensemble=reference['ensemble'],
            Ns=p['Ns'], beta=6.2, mass=p['mass'], mass_error=p['mass_error'], mass_Ns=p['mass']*p['Ns'],
            mass_difference=p['mass']-reference['mass'], difference_error=difference_error,
            difference_combined_sigma=(abs(p['mass']-reference['mass'])/difference_error
                                       if difference_error else None),
            bootstrap_samples=500, flags=p['flags']))
    write_csv(folder/'volume_beta_6p2.csv', rows)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.errorbar([p['Ns'] for p in volumes], [p['mass'] for p in volumes],
                [p['mass_error'] for p in volumes], **pu.data(1))
    finish(fig, ax, folder/'volume_beta_6p2.pdf', r'$N_s$', r'$am_G$',
           r'$N_t=4,\ \beta=6.2$: independent-volume comparison', legend=False)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for i, p in enumerate(volumes, start=1):
        r = np.asarray(p['R']); keep = r > 0
        ax.errorbar(r[keep], np.asarray(p['C_G'])[keep], np.asarray(p['C_G_error'])[keep],
                    label=rf'$N_s={p["Ns"]}$', **pu.data(i, markersize=5))
    ax.set_yscale('symlog', linthresh=max(max(p['C_G_error']) for p in volumes))
    finish(fig, ax, folder/'volume_beta_6p2_correlators.pdf', r'$R/a$', r'$C_G(R)$',
           r'$N_t=4,\ \beta=6.2$: volume comparison of the correlator')
