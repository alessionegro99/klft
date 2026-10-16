"""Completed matched-y runs retain their untuned role in all inventories."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from mg_nt_fixed_inventory import ROOT, load_campaign_manifest
from mg_nt_fixed_scaling_inventory import build_inventory
from mg_nt_fixed_scaling_plots import effective_powers, squared_length_error


class ScalingAnalysisTest(unittest.TestCase):
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
        self.assertEqual(len(matched), 4)
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
        squared = np.linspace(1., 16., 500, dtype=np.float64)
        samples = np.zeros((500,4), dtype=np.float64)
        samples[:,3] = 1.
        samples[:,2] = 1.+4*np.sin(np.pi/ns)**2*squared
        result = dict(Ns=ns, _boot=dict(samples=samples))
        self.assertAlmostEqual(squared_length_error(result), np.std(squared, ddof=1), places=12)
        with self.assertRaisesRegex(ValueError, 'exactly 500'):
            squared_length_error(dict(Ns=ns, _boot=dict(samples=samples[:499])))


if __name__ == '__main__':
    unittest.main()
