"""Completed matched-y runs retain their untuned role in all inventories."""
import json
import unittest
from unittest.mock import patch

from mg_nt_fixed_inventory import ROOT, load_campaign_manifest
from mg_nt_fixed_scaling_inventory import build_inventory


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


if __name__ == '__main__':
    unittest.main()
