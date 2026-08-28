#!/usr/bin/env python3
"""Checks for the fixed-mass final statistics and volume gate."""

from __future__ import annotations

import unittest

from mg_nt_fixed_final_checks import replacement_decision


class MgNtFixedFinalChecksTest(unittest.TestCase):
    def test_expected_periodic_image_difference_does_not_reject_replacement(self) -> None:
        smaller = {"mass": 0.2, "mass_error": 0.004}
        larger = {
            "mass": 0.201,
            "mass_error": 0.002,
            "Ns_over_xi_2nd_68_percent_interval": [14.0, 15.0],
        }
        comparison = {
            "xi_2nd_consistency_pass": True,
            "full_correlator_pass": False,
        }
        decision = replacement_decision(smaller, larger, comparison)
        self.assertTrue(decision["larger_volume_replacement_accepted"])
        self.assertTrue(decision["raw_periodic_correlator_is_diagnostic_only"])

    def test_inconsistent_mass_rejects_replacement(self) -> None:
        smaller = {"mass": 0.2, "mass_error": 0.001}
        larger = {
            "mass": 0.22,
            "mass_error": 0.001,
            "Ns_over_xi_2nd_68_percent_interval": [14.0, 15.0],
        }
        decision = replacement_decision(
            smaller,
            larger,
            {"xi_2nd_consistency_pass": True, "full_correlator_pass": True},
        )
        self.assertFalse(decision["larger_volume_replacement_accepted"])


if __name__ == "__main__":
    unittest.main()
