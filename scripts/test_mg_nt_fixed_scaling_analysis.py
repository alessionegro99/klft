"""Completed matched-y runs retain their untuned role in all inventories."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mg_nt_fixed_inventory import ROOT, load_campaign_manifest
from mg_nt_fixed_scaling_inventory import build_inventory
from mg_nt_fixed_scaling_plots import (NT_PALETTE, SCALING_PDFS, ZOOM_PDFS, ZOOM_RANGE, effective_powers,
                                       plot_lengths, plot_masses, squared_length_error)
from mg_nt_fixed_replot import saved_scan
from mg_nt_fixed_canonical_plots import finish
from plot_thesis_results import FIGURE_SIZE, plt, pu


class ScalingAnalysisTest(unittest.TestCase):
    def test_zoom_preserves_values_errors_styles_and_filters_before_autoscaling(self) -> None:
        samples = np.ones((500, 4), dtype=np.float64)
        samples[:, 2] = np.linspace(1.1, 1.2, 500)
        points = [dict(ensemble=f'{nt}_{y}', physical_Nt=nt, beta=y*nt,
                       mass=y, mass_error=.01, xi_2nd=y, xi_2nd_error=.1,
                       Ns=32, _boot=dict(samples=samples), include=True)
                  for nt in (1, 2) for y in (1., 1.2, 1.5, 1.6, 2.9)]
        plots = {}

        def capture(fig, ax, path, xlabel, ylabel, heading) -> None:
            if path.stem.endswith('_zoom'):
                np.testing.assert_allclose(ax.get_xlim(), ZOOM_RANGE)
                # Off-screen data must not hold the vertical scale open.
                self.assertLess(ax.dataLim.ymax, 11.)
            curves = {}
            for curve in ax.containers:
                line = curve.lines[0]
                self.assertEqual(line.get_markerfacecolor(), 'none')
                curves[curve.get_label()] = (
                    line.get_xdata(orig=False), line.get_ydata(orig=False),
                    np.asarray(curve.lines[2][0].get_segments()))
            plots[path.name] = curves
            plt.close(fig)

        with patch('mg_nt_fixed_scaling_plots.finish', side_effect=capture):
            for zoom in (False, True):
                plot_masses(points, Path('/tmp/mgnt-zoom-test'), zoom=zoom)
                plot_lengths(points, Path('/tmp/mgnt-zoom-test'), zoom=zoom)
        self.assertEqual(set(plots), set(SCALING_PDFS+ZOOM_PDFS))
        for full, zoom in zip(SCALING_PDFS, ZOOM_PDFS):
            for label, (x, y, bars) in plots[full].items():
                keep = (x >= ZOOM_RANGE[0]) & (x <= ZOOM_RANGE[1])
                for observed, expected in zip(plots[zoom][label], (x[keep], y[keep], bars[keep])):
                    np.testing.assert_allclose(observed, expected)
                self.assertEqual(len(plots[zoom][label][0]), 3)

    def test_mass_scaling_includes_nt1_and_scales_errors(self) -> None:
        points = [dict(physical_Nt=nt, beta=2.5, mass=.1, mass_error=.01,
                       include=True, campaign_type='diagnostic') for nt in (1,2)]
        points += [dict(points[0], include=False, beta=99.),
                   dict(points[0], mass=None, beta=98.)]
        plots = {}

        def capture(fig, ax, path, *args) -> None:
            plots[path.stem] = {
                line.get_label(): (line.lines[0].get_xdata(orig=False), line.lines[0].get_ydata(orig=False),
                                   line.lines[2][0].get_segments())
                for line in ax.containers if line.get_label()
                and not line.get_label().startswith('_')}
            plt.close(fig)

        with patch('mg_nt_fixed_scaling_plots.finish', side_effect=capture):
            plot_masses(points, Path('/tmp/mgnt-mass-scaling-test'))
        for name, exponent in (('mass_vs_beta_over_Nt',0.),
                               ('mass_Nt_vs_beta_over_Nt',1.),
                               ('lorenzo_scaling_collapse',1.5)):
            self.assertEqual(len(plots[name]), 2)
            for label, (x,y,bars) in plots[name].items():
                nt = int(label.removeprefix('$N_t=').removesuffix('$'))
                np.testing.assert_allclose(x, [2.5/nt])
                np.testing.assert_allclose(y, [.1*nt**exponent])
                np.testing.assert_allclose(bars, [[[2.5/nt, .09*nt**exponent],
                                                 [2.5/nt, .11*nt**exponent]]])

    def test_nine_plots_use_all_nt_with_open_markers_and_exact_scaling(self) -> None:
        ns = 32
        samples = np.ones((500,4), dtype=np.float64)
        samples[:,2] = 1.+4*np.sin(np.pi/ns)**2*np.linspace(3.5,4.5,500)
        points = [dict(ensemble=f'Nt_{nt}', physical_Nt=nt, beta=1.5*nt,
                       mass=.1, mass_error=.01, xi_2nd=2., xi_2nd_error=.3,
                       Ns=ns, _boot=dict(samples=samples), include=True,
                       campaign_type='matched_y_scaling') for nt in NT_PALETTE]
        plots = {}

        def capture(fig, ax, path, xlabel, ylabel, heading) -> None:
            self.assertEqual(xlabel, r'$\beta/N_t$')
            self.assertEqual(heading, '')
            self.assertEqual(fig.texts, [])
            np.testing.assert_array_equal(fig.get_size_inches(), [16,9])
            self.assertEqual(len(ax.containers), len(NT_PALETTE))
            data = {}
            for curve in ax.containers:
                label = curve.get_label()
                nt = int(label.removeprefix('$N_t=').removesuffix('$'))
                self.assertEqual(label, rf'$N_t={nt}$')
                line = curve.lines[0]
                self.assertEqual(line.get_markerfacecolor(), 'none')
                self.assertEqual(line.get_color(), pu.color_dict[NT_PALETTE[nt]])
                self.assertEqual(line.get_marker(), pu.marker_dict[NT_PALETTE[nt]])
                data[nt] = (line.get_xdata(orig=False), line.get_ydata(orig=False),
                            curve.lines[2][0].get_segments())
            plots[path.name] = data
            plt.close(fig)

        with patch('mg_nt_fixed_scaling_plots.finish', side_effect=capture):
            plot_masses(points, Path('/tmp/mgnt-nine-plots-test'))
            plot_lengths(points, Path('/tmp/mgnt-nine-plots-test'))
        self.assertEqual(set(plots), set(SCALING_PDFS))
        self.assertEqual(len(plots), 9)
        square_error = squared_length_error(points[0])
        for i, name in enumerate(SCALING_PDFS):
            power = (0.,1.,1.5)[i % 3]
            for nt, (x, y, bars) in plots[name].items():
                factor = nt**(power if i < 3 else -power if i < 6 else -2*power)
                central = (.1 if i < 3 else 2. if i < 6 else 4.)*factor
                error = (.01 if i < 3 else .3 if i < 6 else square_error)*factor
                np.testing.assert_allclose(x, [1.5])
                np.testing.assert_allclose(y, [central])
                np.testing.assert_allclose(bars, [[[1.5,central-error],[1.5,central+error]]])

    def test_complete_production_inventory(self) -> None:
        result = build_inventory(ROOT/'data/mg_nt_fixed')
        self.assertEqual(result['ensemble_count'], 4)
        self.assertEqual(result['chain_count'], 16)
        self.assertEqual(result['total_sweeps'], 128000)
        for group in result['ensembles']:
            self.assertIsNone(group['target_c'])
            self.assertFalse(group['fixed_c_continuum_input'])
            self.assertEqual(len(group['chains']), 4)
            self.assertTrue(all(c['purpose'] == 'production' for c in group['chains']))

    def test_changed_completion_or_geometry_is_refused(self) -> None:
        original = json.loads
        for change in ({'status':'running'}, {'stored_L2':99}, {'target_c':.5}):
            def altered(text: str) -> dict:
                result = original(text)
                if result.get('status') == 'complete':
                    result.update(change)
                return result
            with self.subTest(change=change), patch(
                    'mg_nt_fixed_scaling_inventory.json.loads', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'invalid completed'):
                    build_inventory(ROOT/'data/mg_nt_fixed')

    def test_fixed_c_inventory_is_unchanged(self) -> None:
        result = load_campaign_manifest()
        fixed = [g for g in result['ensembles'] if g['fixed_c_continuum_input']]
        matched = [g for g in result['ensembles'] if g['campaign_type']=='matched_y_scaling']
        self.assertEqual(len(fixed), 38)
        self.assertEqual(len(matched), 8)
        self.assertEqual(sum(len(g['chains']) for g in fixed), 156)

    def test_effective_power_known_limit_and_failed_slot(self) -> None:
        points = [dict(ensemble=f'Nt_{nt}', physical_Nt=nt, beta=1.5*nt,
                       mass=nt**(-1.5), _mass_samples=np.full(500, nt**(-1.5)),
                       flags=['preliminary']) for nt in (2, 3, 4, 6)]
        points[1]['_mass_samples'][7] = np.nan
        rows = effective_powers(points)
        np.testing.assert_allclose([p['effective_power'] for p in rows], 1.5)
        self.assertEqual([p['bootstrap_successes'] for p in rows], [499,499,500])
        self.assertTrue(all('preliminary' in p['flags'] for p in rows))

    def test_effective_power_requires_matched_y(self) -> None:
        points = [dict(physical_Nt=nt, beta=4.) for nt in (2, 3)]
        with self.assertRaisesRegex(ValueError, 'same beta/Nt'):
            effective_powers(points)

    def test_squared_length_error_transforms_all_replicas(self) -> None:
        ns = 32
        # Negative noisy estimates remain in the squared-moment distribution.
        squared = np.linspace(-1., 16., 500, dtype=np.float64)
        samples = np.zeros((500,4), dtype=np.float64)
        samples[:,3] = 1.
        samples[:,2] = 1.+4*np.sin(np.pi/ns)**2*squared
        result = dict(Ns=ns, _boot=dict(samples=samples))
        self.assertAlmostEqual(squared_length_error(result), np.std(squared, ddof=1), places=12)
        with self.assertRaisesRegex(ValueError, 'exactly 500'):
            squared_length_error(dict(Ns=ns, _boot=dict(samples=samples[:499])))

    def test_plot_has_axes_but_no_title_or_footer(self) -> None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.set_title('old title')
        fig.text(.5, .01, 'old footer')
        try:
            with patch('mg_nt_fixed_canonical_plots.save_figure'):
                finish(fig, ax, Path('/tmp/mgnt-style-test.pdf'), 'x', 'y', 'new title', legend=False)
            self.assertEqual(ax.get_title(), '')
            self.assertEqual(fig.texts, [])
            self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ('x', 'y'))
        finally:
            plt.close(fig)

    def test_replot_recovers_saved_error_curves_and_counts(self) -> None:
        result = dict(labels=['Ctilde_0'], block_length_sweeps=240,
                      chains=[dict(retained_sweeps=n, tau_int_max_sweeps=48.8) for n in (7500,7250)])
        rows = [dict(observable=label, block_length_sweeps=str(length), bootstrap_error='.2',
                     se_uncertainty_approx='.01', min_blocks=str(7250//length), max_blocks=str(7500//length))
                for length in (80,240) for label in ('Ctilde_0','xi_2nd','Ns_over_xi_2nd')]
        scan = saved_scan(result, rows)
        self.assertEqual(scan['counts'], [[93,90],[31,30]])
        self.assertEqual(scan['minimum_length'], 100)
        np.testing.assert_array_equal(scan['errors'], np.full((2,3), .2))
        rows[0]['min_blocks']='1'
        with self.assertRaisesRegex(ValueError, 'block counts'):
            saved_scan(result, rows)


if __name__ == '__main__':
    unittest.main()
