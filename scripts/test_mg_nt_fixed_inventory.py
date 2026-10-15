"""Verify the immutable current inventory after retiring historical stages."""
import json
import re
from pathlib import Path
import unittest

from mg_nt_fixed_inventory import ROOT, ensemble_key, sha256


class InventoryTest(unittest.TestCase):
    def test_counts_seeds_inputs_and_duration_flags(self) -> None:
        data=ROOT/'data/mg_nt_fixed'
        manifest=json.loads((data/'manifest.json').read_text())
        groups=manifest['ensembles'];chains=[c for p in groups for c in p['chains']]
        self.assertEqual(len(groups),38)
        self.assertEqual(len(chains),156)
        self.assertEqual(len({c['seed'] for c in chains}),156)
        self.assertEqual(len({c['raw_path'] for c in chains}),156)
        self.assertEqual(sum(c['sweeps'] for c in chains),1552000)
        self.assertEqual(sum('length_outlier' in c['flags'] for c in chains),4)
        self.assertEqual(sum(c['source_campaign']=='beta_nt_fixed' for c in chains),8)
        for p in groups:
            self.assertEqual(p['ensemble'],ensemble_key(p['target_c'],p['physical_Nt'],p['Ns'],p['beta']))
            self.assertEqual(p['stored_L2'],p['physical_Nt']+1)
            for c in p['chains']:
                self.assertEqual(sha256(data/c['input']),c['input_sha256'])
                parameters=dict(re.findall(r'^[ \t]+([A-Za-z_0-9]+):[ \t]*([^\n]+)',(data/c['input']).read_text(),re.MULTILINE))
                self.assertEqual((int(parameters['L0']),int(parameters['L1']),int(parameters['L2'])),(p['Ns'],p['Ns'],p['physical_Nt']+1))
                self.assertEqual(float(parameters['beta']),p['beta'])
                self.assertEqual(int(parameters['seed']),c['seed'])
                self.assertEqual(int(parameters['nSweep']),c['sweeps'])
                self.assertEqual(int(parameters['measurement_interval']),1)
                self.assertEqual(int(parameters['dirichlet_suite_measurement_interval']),5)
                self.assertEqual(c['measurement_interval'],1)
                self.assertEqual(c['dirichlet_suite_measurement_interval'],5)
                self.assertEqual('length_outlier' in c['flags'],c['sweeps']==80000)


if __name__=='__main__':
    unittest.main()
