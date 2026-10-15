"""Serial and parallel schedulers must yield identical scientific results."""
import unittest
import numpy as np
from beta_nt_fixed.analysis import finite_volume_k0
from mg_nt_fixed_statistics import scan_mass_windows
from mg_nt_fixed_parallel import parallel_scan


class ParallelTest(unittest.TestCase):
    def test_same_500_replicas_same_results(self) -> None:
        r=np.arange(9,dtype=np.float64);mean=np.ones(9)
        mean[1:]=finite_volume_k0(r[1:],.5,.3,16)
        rng=np.random.default_rng(91)
        samples=mean+rng.normal(size=(500,9))*.0002
        serial=scan_mass_windows(r,mean,samples,16)
        parallel=parallel_scan(r,mean,samples,16,workers=2)
        self.assertEqual(serial[0],parallel[0])
        np.testing.assert_array_equal(serial[1],parallel[1])
        self.assertEqual(serial[2:],parallel[2:])


if __name__=='__main__':
    unittest.main()
