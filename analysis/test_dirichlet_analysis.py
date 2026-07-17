#!/usr/bin/env python3
"""Deterministic checks for the Dirichlet statistical analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from dirichlet_analysis import (
    block_values,
    finite_volume_k0,
    fit_k0,
    integrated_autocorrelation_time,
    jackknife_transform,
    jackknife_k0_fit,
    load_keyed_chain,
)


class DirichletAnalysisTest(unittest.TestCase):
    def test_keyed_loader_and_thermalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correlator.out"
            rows = ["# step R C_G C_double m_eff_log"]
            for step in range(1, 7):
                for separation in range(3):
                    value = 1.0 if separation == 0 else 0.1 * step / separation
                    rows.append(f"{step} {separation} {value} {value} 0")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            chain = load_keyed_chain(path, (1,), (2, 3), discard=2)
            np.testing.assert_array_equal(chain.steps, np.arange(3, 7))
            np.testing.assert_array_equal(chain.keys[:, 0], np.arange(3))
            self.assertEqual(chain.values.shape, (4, 3, 2))

    def test_autocorrelation_and_blocking(self) -> None:
        rng = np.random.default_rng(12345)
        rho = 0.8
        noise = rng.normal(size=200_000)
        values = np.empty_like(noise)
        values[0] = noise[0]
        for index in range(1, len(values)):
            values[index] = rho * values[index - 1] + noise[index]
        # Exact AR(1): tau_int=(1+rho)/(2(1-rho))=4.5.
        self.assertAlmostEqual(integrated_autocorrelation_time(values), 4.5, delta=0.35)
        blocks = block_values(np.arange(24, dtype=np.float64), 6)
        np.testing.assert_allclose(blocks, [2.5, 8.5, 14.5, 20.5])

    def test_jackknife_ratio(self) -> None:
        blocks = np.asarray([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0], [8.0, 4.0]])
        ratio, covariance = jackknife_transform(
            blocks, lambda mean: np.asarray([mean[0] / mean[1]])
        )
        self.assertEqual(ratio[0], 2.0)
        self.assertAlmostEqual(covariance[0, 0], 0.0)

    def test_finite_volume_k0_fit(self) -> None:
        separations = np.arange(2, 13, dtype=np.float64)
        expected_amplitude = 0.7
        expected_mass = 0.18
        values = finite_volume_k0(
            separations, expected_amplitude, expected_mass, ns=64
        )
        covariance = np.diag(np.full(len(separations), 1.0e-8))
        for correlated in (True, False):
            fit = fit_k0(
                separations, values, covariance, ns=64, correlated=correlated
            )
            self.assertAlmostEqual(float(fit["amplitude"]), expected_amplitude, places=9)
            self.assertAlmostEqual(float(fit["mass"]), expected_mass, places=9)
            self.assertLess(float(fit["chi2"]), 1.0e-15)

    def test_k0_fit_jackknife(self) -> None:
        rng = np.random.default_rng(54321)
        separations = np.arange(2, 13, dtype=np.float64)
        expected = finite_volume_k0(separations, 0.7, 0.18, ns=64)
        blocks = expected + rng.normal(scale=2.0e-4, size=(24, len(expected)))
        covariance = np.atleast_2d(np.cov(blocks, rowvar=False, ddof=1)) / len(
            blocks
        )
        parameters, parameter_covariance = jackknife_k0_fit(
            blocks, separations, covariance, ns=64, correlated=False
        )
        np.testing.assert_allclose(parameters, [0.7, 0.18], rtol=0.02)
        self.assertTrue(np.all(np.isfinite(parameter_covariance)))
        self.assertTrue(np.all(np.diag(parameter_covariance) > 0.0))


if __name__ == "__main__":
    unittest.main()
