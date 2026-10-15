"""Nt=1 execution provenance and untuned role must survive analysis."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from mg_nt_fixed_canonical_plots import title
from mg_nt_fixed_inventory import ROOT, load_campaign_manifest
from mg_nt_fixed_nt1_inventory import build_inventory
from mg_nt_fixed_adaptive import coarse_bounds, dense_bounds
from mg_nt_fixed_statistics import choose_window, fit_window


class Nt1AnalysisTest(unittest.TestCase):
    def test_adaptive_bounds_and_dense_adjacent_patch(self) -> None:
        bounds = coarse_bounds(2048)
        self.assertLess(len(bounds), 200)
        self.assertTrue(all(hi-lo >= 4 and hi % 2 == 0 and hi >= 683 for lo,hi in bounds))
        rows = [dict(r_min=lo,r_max=hi,mass=.007,mass_error=.0001) for lo,hi in bounds]
        dense = dense_bounds(rows, 2048)
        self.assertTrue(dense)
        self.assertEqual(len(set(bounds+dense)), len(bounds+dense))
        rows += [dict(r_min=lo,r_max=hi,mass=.007,mass_error=.0001) for lo,hi in dense]
        selected, flags = choose_window(rows)
        self.assertEqual((rows[selected]['r_min'], rows[selected]['r_max']), (1,1024))
        self.assertEqual(flags, [])

    def test_covariance_rank_bound_keeps_defined_mass(self) -> None:
        r = np.arange(1.,501.)
        with patch('mg_nt_fixed_statistics._uncorrelated_k0_fit_with_errors',
                   return_value={'amplitude':1., 'mass':.007}), patch(
                   'mg_nt_fixed_statistics.np.linalg.cond', side_effect=AssertionError('rank is known')):
            row, parameters = fit_window(r, np.ones(500), np.ones((500,500)), np.eye(500), 2048, 1, 500)
        self.assertIsNone(row['chi2_full'])
        self.assertIn('singular_full_covariance', row['flags'])
        self.assertEqual(row['bootstrap_successes'], 500)
        self.assertEqual(parameters.shape, (500,2))

    def test_combined_inventory_keeps_physics_roles(self) -> None:
        result = load_campaign_manifest()
        self.assertEqual(result['ensemble_count'], 41)
        self.assertEqual(result['chain_count'], 168)
        self.assertEqual(result['total_sweeps'], 1648000)
        fixed = [g for g in result['ensembles'] if g['fixed_c_continuum_input']]
        self.assertEqual(len(fixed), 38)
        self.assertTrue(all(g['physical_Nt'] > 1 for g in fixed))

    def test_complete_production_inventory(self) -> None:
        result = build_inventory(ROOT/'data/mg_nt_fixed')
        self.assertEqual(result['chain_count'], 12)
        self.assertEqual(result['total_sweeps'], 96000)
        for group in result['ensembles']:
            self.assertIsNone(group['target_c'])
            self.assertFalse(group['fixed_c_continuum_input'])
            self.assertEqual(len(group['chains']), 4)
            self.assertIn(r'c_{\rm ref}', title(group))
            for chain in group['chains']:
                self.assertNotEqual(chain['planned_heatbath_sha256'], chain['heatbath_sha256'])
                self.assertEqual(chain['sweeps'], 8000)

    def test_unfinished_execution_is_refused(self) -> None:
        original = json.loads
        def unfinished(text: str) -> dict:
            result = original(text)
            if result.get('status') == 'complete':
                result['status'] = 'running'
            return result
        with patch('mg_nt_fixed_nt1_inventory.json.loads', side_effect=unfinished):
            with self.assertRaisesRegex(ValueError, 'invalid completed'):
                build_inventory(ROOT/'data/mg_nt_fixed')


if __name__ == '__main__':
    unittest.main()
