"""Coupling and continuum propagation retain flags and all 500 slots."""
import unittest
import numpy as np
from mg_nt_fixed_campaign import continuum_fit, line_fit, primary_volumes, tuning_windows


class CampaignTest(unittest.TestCase):
    def test_line_and_covariance(self) -> None:
        x=np.array([.25,.333333333333,.5]);y=.7+.2*x
        rng=np.random.default_rng(17);bs=y+rng.normal(size=(500,3))*.003
        result=line_fit(x,y,np.full(3,.003),bs)
        np.testing.assert_allclose(result['parameters'],[.7,.2],atol=1e-12)
        self.assertEqual(result['bootstrap_parameters'].shape,(500,2))
        self.assertLess(result['chi2_full'],1e-20)

    def points(self) -> list[dict]:
        rng=np.random.default_rng(19)
        return [dict(beta=b,mass=float(np.exp(3-b)/3),mass_error=.003,
                     _mass_samples=np.exp(3-b)/3+rng.normal(size=500)*.003,
                     flags=['late_window_drift'],ensemble=f'beta_{b}',Ns=32,target_c=1.,physical_Nt=3)
                for b in (2.8,3.,3.2,3.4)]

    def test_all_coupling_windows_and_flags(self) -> None:
        rows,details,chosen=tuning_windows(self.points(),1.,3)
        self.assertEqual(len(rows),2)
        self.assertIsNotNone(chosen)
        self.assertAlmostEqual(rows[chosen]['beta_star'],3.,places=2)
        self.assertIn('late_window_drift',rows[chosen]['flags'])
        self.assertEqual(details[chosen]['roots'].shape,(500,))
        self.assertNotIn('p_value',rows[chosen])

    def test_no_unbracketed_extrapolation(self) -> None:
        rows,_,chosen=tuning_windows(self.points(),8.,3)
        self.assertIsNone(chosen)
        self.assertTrue(all(r['beta_star'] is None for r in rows))

    def test_defined_positive_slope_is_flagged_for_review(self) -> None:
        points=self.points()
        for point in points:
            scale=np.exp(2*(point['beta']-3))
            point['mass']*=scale
            point['_mass_samples']*=scale
            point['mass_error']*=scale
        rows,_,chosen=tuning_windows(points,1.,3)
        self.assertIsNotNone(chosen)
        self.assertIn('positive_slope_review',rows[chosen]['flags'])

    def test_volume_is_not_pooled(self) -> None:
        points=self.points()
        duplicate=dict(points[0],Ns=96)
        selected=primary_volumes(points+[duplicate])
        self.assertEqual(len(selected),4)
        self.assertEqual(selected[0]['Ns'],96)

    def test_continuum_only_p1_and_two_point_warning(self) -> None:
        rng=np.random.default_rng(21)
        rows=[dict(physical_Nt=n,beta_star=4*n*(.6+.1/n),
                   _roots=4*n*(.6+.1/n)+rng.normal(size=500)*.01,
                   flags=['provisional']) for n in (2,3,4)]
        row,_=continuum_fit(rows,1.)
        self.assertAlmostEqual(row['Q0'],.6,places=12)
        self.assertEqual(row['p'],1)
        row,_=continuum_fit(rows,1.,True)
        self.assertIn('zero_residual_dof',row['flags'])
        self.assertIn('provisional',row['flags'])


if __name__=='__main__':
    unittest.main()
