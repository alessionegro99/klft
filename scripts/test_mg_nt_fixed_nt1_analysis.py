"""Nt=1 execution provenance and untuned role must survive analysis."""
import json
import unittest
from unittest.mock import patch

from mg_nt_fixed_canonical_plots import title
from mg_nt_fixed_inventory import ROOT
from mg_nt_fixed_nt1_inventory import build_inventory


class Nt1AnalysisTest(unittest.TestCase):
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
