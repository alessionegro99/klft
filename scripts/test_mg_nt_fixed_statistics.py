"""Regression checks for canonical pooling, covariance and flagged fits."""
from __future__ import annotations

import unittest
import numpy as np

from beta_nt_fixed.analysis import finite_volume_k0, tau_diagnostics
from dirichlet_analysis import block_values, integrated_autocorrelation_time
from mg_nt_fixed_inventory import ensemble_key
from mg_nt_fixed_statistics import (choose_window, fit_window, operation_seed,
                                    pooled_bootstrap, review_window)


class CanonicalStatisticsTest(unittest.TestCase):
    def test_exact_count_and_fixed_chain_weights(self) -> None:
        a, b = np.ones((20, 2)), np.full((80, 2), 3.)
        mean, samples, weights = pooled_bootstrap([a,b], 2)
        np.testing.assert_allclose(weights, [.2,.8])
        np.testing.assert_allclose(mean, [2.6,2.6])
        np.testing.assert_allclose(samples, np.full((500,2),2.6))
        with self.assertRaisesRegex(ValueError, 'exactly 500'):
            pooled_bootstrap([a,b], 2, 1000)

    def test_iid_weighted_error_and_vector_correlation(self) -> None:
        rng = np.random.default_rng(50)
        chains = [rng.normal(size=(n,1)) for n in (300,900)]
        blocks = [np.column_stack((v,2*v)) for v in chains]
        _, samples, weights = pooled_bootstrap(blocks, 6)
        np.testing.assert_allclose(samples[:,1], 2*samples[:,0], atol=1e-14)
        expected = np.sqrt(sum(w*w*np.var(v[:,0],ddof=1)/len(v) for w,v in zip(weights,blocks)))
        self.assertLess(abs(samples[:,0].std(ddof=1)/expected-1), .12)

    def test_blocks_do_not_cross_chains(self) -> None:
        a, b = block_values(np.zeros((23,1)),10), block_values(np.ones((47,1)),10)
        mean, draws, weights = pooled_bootstrap([a,b], 3)
        np.testing.assert_allclose(weights,[1/3,2/3])
        np.testing.assert_allclose(draws, np.full((500,1),2/3))

    def test_ar1_block_error_and_tau_units(self) -> None:
        rng = np.random.default_rng(52)
        innovations = rng.normal(size=(24000,2))
        values = np.zeros_like(innovations)
        for i in range(1,len(values)):
            values[i] = .7*values[i-1]+innovations[i]
        tau = integrated_autocorrelation_time(values[:,0])
        self.assertLess(abs(tau-((1+.7)/(2*(1-.7)))), .7)
        _, draws, _ = pooled_bootstrap([block_values(values[:,i:i+1],100) for i in range(2)],7)
        expected = np.sqrt(1/((1-.7)**2 * 48000))
        self.assertLess(abs(np.std(draws[:,0],ddof=1)/expected-1), .2)
        tau_record=tau_diagnostics(values, np.column_stack((values,values)), values[:,0:1], np.array([[0.,1.]]))
        self.assertEqual(tau_record['plaquette_profiles']['spatial_t_1'],5*tau)

    def test_k0_and_singular_chi2_retained(self) -> None:
        r = np.arange(1.,9.)
        mean = finite_volume_k0(r, .6,.22,32)
        rng = np.random.default_rng(62)
        draws = mean + rng.normal(size=(500,1))*.0003
        row, params = fit_window(r, mean, draws, np.full((8,8),9e-8), 32,1,8)
        self.assertAlmostEqual(row['mass'],.22,places=8)
        self.assertIsNotNone(row['mass_error'])
        self.assertIsNone(row['chi2_full'])
        self.assertIn('singular_full_covariance',row['flags'])
        self.assertEqual(params.shape,(500,2))
        self.assertNotIn('p_value',row)

    def test_stable_patch_and_manual_override_ignore_chi2(self) -> None:
        rows = [dict(r_min=lo,r_max=hi,mass=.2,mass_error=.01,chi2_full=1e10,flags=[])
                for hi in (16,14,12) for lo in range(1,hi-3)]
        index, flags = choose_window(rows)
        self.assertEqual((rows[index]['r_min'],rows[index]['r_max']),(1,16))
        self.assertEqual(flags,[])
        chosen = review_window(rows,index,{'r_min':'3','r_max':'12'})
        self.assertEqual(rows[chosen]['r_min'],3)
        with self.assertRaises(ValueError):
            review_window(rows,index,{'r_min':'90','r_max':'12'})

    def test_decimal_keys_and_seeds(self) -> None:
        self.assertEqual(ensemble_key(.5,3,80,4.40),'c_0p5/Nt_3/Ns_80/beta_4p4')
        self.assertNotEqual(operation_seed('a','b'),operation_seed('a','c'))
        self.assertEqual(operation_seed('a','b'),operation_seed('a','b'))


if __name__ == '__main__':
    unittest.main()
