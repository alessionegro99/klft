#!/usr/bin/env python3
"""Deterministic checks for the Step-3 statistical acceptance rules."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from step3_analysis import (
    fused_replica_jackknife,
    choose_block_length_sweeps,
    finite_volume_k0,
    load_ensemble,
    prepare_chains,
    scan_k0_fits,
    second_moment_samples,
    second_moment_squared_samples,
    stable_three_length_groups,
)


class Step3AnalysisTest(unittest.TestCase):
    @staticmethod
    def _write_chain(directory: Path, seed: int) -> None:
        rng = np.random.default_rng(seed)
        summary = ["# step ReTrG ImTrG Ctilde_0 Ctilde_pmin xi_2nd_config"]
        correlator = ["# step R C_G C_double m_eff_log"]
        profile = ["# step kind t ReTrU_over_Nc"]
        for step in range(1, 2001):
            noise = rng.normal(scale=0.01)
            summary.append(f"{step} {noise} 0 1.16 1 0")
            for separation, mean in enumerate((1.0, 0.5, 0.25, 0.12, 0.06)):
                value = 1.0 if separation == 0 else mean + rng.normal(scale=0.002)
                correlator.append(f"{step} {separation} {value} {value} 0")
            if step % 5 == 0:
                for kind, coordinate in (
                    (0, 0.0), (0, 1.0), (0, 2.0), (1, 0.5), (1, 1.5)
                ):
                    value = 0.7 + rng.normal(scale=0.002)
                    profile.append(f"{step} {kind} {coordinate} {value}")
        directory.mkdir(parents=True)
        (directory / "holonomy_summary.out").write_text(
            "\n".join(summary) + "\n", encoding="utf-8"
        )
        (directory / "holonomy_correlator.out").write_text(
            "\n".join(correlator) + "\n", encoding="utf-8"
        )
        (directory / "plaquette_profile.out").write_text(
            "\n".join(profile) + "\n", encoding="utf-8"
        )

    def test_block_length_is_aligned_and_strictly_longer_than_two_tau(self) -> None:
        for tau in (0.5, 2.5, 7.1, 20.0):
            length = choose_block_length_sweeps(tau)
            self.assertEqual(length % 5, 0)
            self.assertGreater(length, 2.0 * tau)

    def test_common_plateau_requires_one_shared_three_length_group(self) -> None:
        lengths = [10, 20, 30, 40, 50]
        errors = np.asarray(
            [[1.0, 2.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
            dtype=np.float64,
        )
        groups, common = stable_three_length_groups(
            lengths, errors, np.zeros_like(errors), minimum_length=10
        )
        self.assertEqual(groups[0], [[10, 20, 30]])
        self.assertEqual(groups[1], [[30, 40, 50]])
        self.assertEqual(common, [])

    def test_second_moment_length_uses_ensemble_structure_factors(self) -> None:
        ns = 96
        expected = np.asarray([5.0, 8.0], dtype=np.float64)
        ratio = (2.0 * np.sin(np.pi / ns) * expected) ** 2
        samples = np.zeros((2, 4), dtype=np.float64)
        samples[:, 2] = 1.0 + ratio
        samples[:, 3] = 1.0
        np.testing.assert_allclose(second_moment_samples(samples, ns), expected)

    def test_second_moment_keeps_unresolved_squared_samples(self) -> None:
        ns = 48
        samples = np.zeros((2, 4), dtype=np.float64)
        samples[:, 3] = 1.0
        samples[:, 2] = [1.01, 0.99]
        squared = second_moment_squared_samples(samples, ns)
        self.assertGreater(squared[0], 0.0)
        self.assertLess(squared[1], 0.0)
        xi = second_moment_samples(samples, ns)
        self.assertTrue(np.isfinite(xi[0]))
        self.assertTrue(np.isnan(xi[1]))

    def test_fused_jackknife_blocks_replicas_separately(self) -> None:
        blocks = [
            np.arange(4, dtype=np.float64)[:, None],
            np.arange(10, 14, dtype=np.float64)[:, None],
        ]
        estimate, error, degrees_of_freedom = fused_replica_jackknife(
            blocks, lambda mean: mean
        )
        np.testing.assert_allclose(estimate, [6.5])
        np.testing.assert_allclose(error, [np.sqrt(5.0 / 24.0)])
        np.testing.assert_allclose(degrees_of_freedom, [6.0])

    def test_fused_jackknife_uses_effective_degrees_of_freedom(self) -> None:
        blocks = [
            np.arange(4, dtype=np.float64)[:, None],
            (2.0 * np.arange(4, dtype=np.float64))[:, None],
        ]
        _, error, degrees_of_freedom = fused_replica_jackknife(
            blocks, lambda mean: mean
        )
        np.testing.assert_allclose(error, [np.sqrt(25.0 / 48.0)])
        np.testing.assert_allclose(degrees_of_freedom, [75.0 / 17.0])

    def test_correlated_k0_scan_accepts_three_stable_adjacent_ranges(self) -> None:
        rng = np.random.default_rng(20260721)
        ns = 32
        separations = np.arange(ns // 2 + 1, dtype=np.int64)
        correlator = np.empty_like(separations, dtype=np.float64)
        correlator[0] = 1.0
        correlator[1:] = finite_volume_k0(
            separations[1:].astype(np.float64), 0.7, 0.2, ns
        )
        errors = np.full(len(separations), 2.0e-4, dtype=np.float64)
        errors[0] = 0.0
        covariance = np.diag(errors * errors)
        samples = np.broadcast_to(correlator, (300, len(correlator))).copy()
        samples[:, 1:] += rng.normal(
            scale=errors[1:], size=(len(samples), len(correlator) - 1)
        )
        xi, xi_error, scan = scan_k0_fits(
            separations, correlator, covariance, samples, ns
        )
        self.assertIsNotNone(xi)
        self.assertIsNotNone(xi_error)
        self.assertAlmostEqual(float(xi), 5.0, delta=0.05)
        self.assertTrue(any(entry.get("selected", False) for entry in scan))

    def test_end_to_end_grid_keeps_chains_and_profile_cadence_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for chain_index in range(4):
                self._write_chain(root / f"seed_{100 + chain_index}", chain_index)
            raw = load_ensemble(root)
            chains, diagnostics = prepare_chains(raw, minimum_cut=100, ns=8)
            self.assertEqual(len(chains), 4)
            self.assertEqual(len(diagnostics["chains"]), 4)
            for chain in chains:
                self.assertEqual(chain.cut % 5, 0)
                self.assertEqual(
                    len(chain.summary), 5 * len(chain.profile)
                )
                self.assertGreaterEqual(len(chain.primary_blocks), 30)
            plateau = diagnostics["blocking_error_plateau"]
            self.assertEqual(
                plateau["candidate_block_lengths_sweeps"],
                [5, 10, 15, 20, 30, 40],
            )
            self.assertTrue(
                plateau["all_primary_and_secondary_plateaus_found"]
            )
            self.assertEqual(
                plateau["selected_common_block_length_sweeps"],
                plateau["common_plateau_group_sweeps"][-1],
            )

    def test_explicit_chain_directories_ignore_other_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = []
            for chain_index in range(4):
                directory = root / f"seed_{100 + chain_index}"
                self._write_chain(directory, chain_index)
                selected.append(directory)
            (root / "seed_999").mkdir()
            with self.assertRaisesRegex(ValueError, "expected four"):
                load_ensemble(root)
            chains = load_ensemble(root, selected)
            self.assertEqual(
                [chain.directory for chain in chains], selected
            )


if __name__ == "__main__":
    unittest.main()
