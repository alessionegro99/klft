#!/usr/bin/env python3
"""Chain-aware analysis for the Step-3 Dirichlet-slab pilot.

MSER-5 and the Sokal-window autocorrelation estimator follow the references in
``thermalization_diagnostics.py``. Independent chains are blocked separately;
the seeded two-level resampling follows the references in
``pcm_beta_pilot_analysis.py``. The correlated periodic K0 form is Eq. (4.14)
of M. Nebbia, *Effects of Dirichlet Boundary Conditions in Yang-Mills
Theories* (2024/2025 MSc thesis). No raw histories are concatenated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, special, stats

from dirichlet_analysis import (
    FloatArray,
    KeyedChain,
    block_values,
    integrated_autocorrelation_time,
    load_keyed_chain,
    require_matching_grids,
)
from pcm_beta_pilot_analysis import hierarchical_bootstrap
from thermalization_diagnostics import mser5_cut, split_z_score


IntArray = NDArray[np.int64]
PROFILE_CADENCE = 5
MINIMUM_BLOCKS = 30
TARGET_BLOCKS = 50
PLATEAU_RELATIVE_TOLERANCE = 0.05
PLATEAU_SIGMA_TOLERANCE = 2.0
BLOCKING_CANDIDATE_UNITS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


@dataclass(frozen=True)
class ChainData:
    """Validated raw observable tables for one independent chain."""

    directory: Path
    summary: KeyedChain
    correlator: KeyedChain
    profile: KeyedChain


@dataclass(frozen=True)
class PostCutChain:
    """Post-thermalization arrays and chain-specific blocking metadata."""

    raw: ChainData
    cut: int
    summary: FloatArray
    correlator: FloatArray
    profile: FloatArray
    tau: dict[str, object]
    block_length_sweeps: int
    primary_blocks: FloatArray
    profile_blocks: FloatArray


def load_chain(directory: Path) -> ChainData:
    """Load one chain and require the Step-3 1/5 measurement grids."""
    summary = load_keyed_chain(
        directory / "holonomy_summary.out", (), (1, 2, 3, 4), 0
    )
    correlator = load_keyed_chain(
        directory / "holonomy_correlator.out", (1,), (2,), 0
    )
    profile = load_keyed_chain(
        directory / "plaquette_profile.out", (1, 2), (3,), 0
    )
    if not np.array_equal(summary.steps, correlator.steps):
        raise ValueError(f"{directory}: holonomy measurement grids differ")
    if len(summary.steps) < 4 or not np.all(np.diff(summary.steps) == 1):
        raise ValueError(f"{directory}: holonomy steps are not consecutive sweeps")
    expected_profile_steps = summary.steps[
        np.asarray(summary.steps, dtype=np.int64) % PROFILE_CADENCE == 0
    ]
    if not np.array_equal(profile.steps, expected_profile_steps):
        raise ValueError(f"{directory}: profile steps do not have 5-sweep cadence")
    separations = correlator.keys[:, 0].astype(np.int64)
    if not np.array_equal(separations, np.arange(separations[-1] + 1)):
        raise ValueError(f"{directory}: incomplete correlator separation grid")
    if not np.all(correlator.values[:, 0, 0] == 1.0):
        raise ValueError(f"{directory}: C_G(0) is not exactly one")
    return ChainData(directory, summary, correlator, profile)


def load_ensemble(
    directory: Path, chain_directories: Sequence[Path] | None = None
) -> list[ChainData]:
    """Load exactly four independent chains from one ensemble directory."""
    if chain_directories is None:
        selected_directories = sorted(
            (path for path in directory.glob("seed_*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("seed_")),
        )
    else:
        selected_directories = list(chain_directories)
        if len(set(selected_directories)) != len(selected_directories):
            raise ValueError(f"{directory}: duplicate chain directory")
        ensemble = directory.resolve()
        if any(path.resolve().parent != ensemble for path in selected_directories):
            raise ValueError(f"{directory}: chain outside ensemble directory")
    if len(selected_directories) != 4:
        raise ValueError(f"{directory}: expected four seed directories")
    chains = [load_chain(path) for path in selected_directories]
    require_matching_grids([chain.correlator for chain in chains])
    require_matching_grids([chain.profile for chain in chains])
    return chains


def _largest_named(values: dict[str, float]) -> tuple[str, float]:
    """Return the key and value of the largest finite mapping entry."""
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    if not finite:
        raise ValueError("no finite diagnostic values")
    key = max(finite, key=finite.__getitem__)
    return key, float(finite[key])


def candidate_thermalization_cut(
    chain: ChainData, minimum_cut: int
) -> tuple[int, dict[str, object]]:
    """Return the per-chain MSER-5 cut in sweeps, rounded to profile cadence."""
    summary = chain.summary.values[:, 0, :]
    correlator = chain.correlator.values[:, :, 0]
    profile = chain.profile.values[:, :, 0]
    scalar_candidates = {
        "Tr_G_real": mser5_cut(summary[:, 0]),
        "Ctilde_0": mser5_cut(summary[:, 2]),
        "Ctilde_pmin": mser5_cut(summary[:, 3]),
        "C_G_R1": mser5_cut(correlator[:, 1]),
    }
    correlator_candidates = {
        f"C_G_R{index}": mser5_cut(correlator[:, index])
        for index in range(1, correlator.shape[1])
    }
    profile_candidates = {
        profile_label(chain.profile.keys[index]): PROFILE_CADENCE
        * mser5_cut(profile[:, index])
        for index in range(profile.shape[1])
    }
    largest_correlator = _largest_named(correlator_candidates)
    largest_profile = _largest_named(profile_candidates)
    raw_cut = max(
        minimum_cut,
        *scalar_candidates.values(),
        largest_correlator[1],
        largest_profile[1],
    )
    cut = int(PROFILE_CADENCE * math.ceil(raw_cut / PROFILE_CADENCE))
    if cut >= int(chain.summary.steps[-1]) - 4:
        raise ValueError(f"{chain.directory}: thermalization leaves too little data")
    return cut, {
        "selected_scalar_candidates_sweeps": scalar_candidates,
        "largest_correlator_candidate": {
            "observable": largest_correlator[0],
            "sweeps": largest_correlator[1],
        },
        "largest_profile_candidate": {
            "observable": largest_profile[0],
            "sweeps": largest_profile[1],
        },
    }


def profile_label(key: FloatArray) -> str:
    """Return an unambiguous profile-coordinate label."""
    kind = "spatial" if int(key[0]) == 0 else "temporal"
    return f"{kind}_t_{float(key[1]):g}"


def tau_diagnostics(
    correlator: FloatArray, summary: FloatArray, profile: FloatArray,
    profile_keys: FloatArray,
) -> dict[str, object]:
    """Estimate every required autocorrelation time in composite sweeps."""
    correlator_taus = {
        f"C_G_R{index}": integrated_autocorrelation_time(correlator[:, index])
        for index in range(1, correlator.shape[1])
    }
    largest_correlator = _largest_named(correlator_taus)
    profile_taus = {
        profile_label(profile_keys[index]): PROFILE_CADENCE
        * integrated_autocorrelation_time(profile[:, index])
        for index in range(profile.shape[1])
    }
    largest_profile = _largest_named(profile_taus)
    return {
        "Tr_G_real": integrated_autocorrelation_time(summary[:, 0]),
        "Ctilde_0": integrated_autocorrelation_time(summary[:, 2]),
        "Ctilde_pmin": integrated_autocorrelation_time(summary[:, 3]),
        "C_G_R1": integrated_autocorrelation_time(correlator[:, 1]),
        "largest_positive_distance_C_G": {
            "observable": largest_correlator[0],
            "tau_int_sweeps": largest_correlator[1],
        },
        "plaquette_profiles": profile_taus,
        "largest_plaquette_profile": {
            "observable": largest_profile[0],
            "tau_int_sweeps": largest_profile[1],
        },
    }


def maximum_required_tau(tau: dict[str, object]) -> float:
    """Return the largest required autocorrelation time in sweeps."""
    scalar = [
        float(tau["Tr_G_real"]),
        float(tau["Ctilde_0"]),
        float(tau["Ctilde_pmin"]),
        float(tau["C_G_R1"]),
        float(tau["largest_positive_distance_C_G"]["tau_int_sweeps"]),
        float(tau["largest_plaquette_profile"]["tau_int_sweeps"]),
    ]
    return max(scalar)


def choose_block_length_sweeps(tau_int_sweeps: float) -> int:
    """Choose the smallest 5-sweep multiple strictly exceeding ``2 tau``."""
    if not np.isfinite(tau_int_sweeps) or tau_int_sweeps < 0.5:
        raise ValueError("tau_int must be finite and at least 0.5 sweeps")
    minimum = math.floor(2.0 * tau_int_sweeps) + 1
    length = PROFILE_CADENCE * math.ceil(minimum / PROFILE_CADENCE)
    if length <= 2.0 * tau_int_sweeps:
        length += PROFILE_CADENCE
    return int(length)


def _post_cut(chain: ChainData, cut: int) -> PostCutChain:
    """Apply one cut and form complete within-chain block vectors."""
    summary_mask = chain.summary.steps > cut
    correlator_mask = chain.correlator.steps > cut
    profile_mask = chain.profile.steps > cut
    summary = chain.summary.values[summary_mask, 0, :]
    correlator = chain.correlator.values[correlator_mask, :, 0]
    profile = chain.profile.values[profile_mask, :, 0]
    if len(summary) != len(correlator) or len(summary) < 4:
        raise ValueError(f"{chain.directory}: invalid post-cut holonomy grid")
    if len(summary) != PROFILE_CADENCE * len(profile):
        raise ValueError(f"{chain.directory}: post-cut 1/5 grids are misaligned")

    tau = tau_diagnostics(correlator, summary, profile, chain.profile.keys)
    block_length = choose_block_length_sweeps(maximum_required_tau(tau))
    combined = np.concatenate((summary, correlator), axis=1)
    primary_blocks = block_values(combined, block_length)
    profile_blocks = block_values(profile, block_length // PROFILE_CADENCE)
    if len(primary_blocks) != len(profile_blocks):
        raise ValueError(f"{chain.directory}: inconsistent complete block counts")
    if len(primary_blocks) < MINIMUM_BLOCKS:
        raise ValueError(
            f"{chain.directory}: only {len(primary_blocks)} complete blocks; "
            f"need at least {MINIMUM_BLOCKS}"
        )
    return PostCutChain(
        chain,
        cut,
        summary,
        correlator,
        profile,
        tau,
        block_length,
        primary_blocks,
        profile_blocks,
    )


def _reblock_chain(chain: PostCutChain, block_length_sweeps: int) -> PostCutChain:
    """Return one chain reblocked without crossing replica boundaries."""
    if block_length_sweeps % PROFILE_CADENCE != 0:
        raise ValueError("block length must be a multiple of profile cadence")
    combined = np.concatenate((chain.summary, chain.correlator), axis=1)
    primary_blocks = block_values(combined, block_length_sweeps)
    profile_blocks = block_values(
        chain.profile, block_length_sweeps // PROFILE_CADENCE
    )
    if len(primary_blocks) != len(profile_blocks):
        raise ValueError(f"{chain.raw.directory}: inconsistent reblocked grids")
    if len(primary_blocks) < MINIMUM_BLOCKS:
        raise ValueError(
            f"{chain.raw.directory}: only {len(primary_blocks)} complete blocks; "
            f"need at least {MINIMUM_BLOCKS}"
        )
    return replace(
        chain,
        block_length_sweeps=block_length_sweeps,
        primary_blocks=primary_blocks,
        profile_blocks=profile_blocks,
    )


def fused_replica_jackknife(
    per_chain_blocks: Sequence[FloatArray],
    transform: Callable[[FloatArray], FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a delete-one-block error for an equal-weight replica mean.

    Each deletion changes only one already-blocked chain mean. The independent
    replica contributions are then summed, so no block crosses a chain
    boundary. This is the replica-stratified form of the delete-one-block
    jackknife used for the blocking curves.
    """
    if len(per_chain_blocks) < 2:
        raise ValueError("fused blocking needs at least two replicas")
    if any(len(blocks) < 3 for blocks in per_chain_blocks):
        raise ValueError("every replica needs at least three complete blocks")
    vector_shape = per_chain_blocks[0].shape[1:]
    if any(blocks.shape[1:] != vector_shape for blocks in per_chain_blocks):
        raise ValueError("replica block vectors have inconsistent shapes")

    chain_means = np.asarray(
        [np.mean(blocks, axis=0) for blocks in per_chain_blocks],
        dtype=np.float64,
    )
    n_chains = len(chain_means)
    chain_sum = np.sum(chain_means, axis=0)
    estimate = np.asarray(transform(chain_sum / n_chains), dtype=np.float64)
    variance = np.zeros_like(estimate)
    variance_df_denominator = np.zeros_like(estimate)
    nominal_degrees_of_freedom = 0
    for chain_index, blocks in enumerate(per_chain_blocks):
        n_blocks = len(blocks)
        deleted_chain_means = (
            np.sum(blocks, axis=0)[None, :] - blocks
        ) / (n_blocks - 1)
        deleted_fused_means = (
            chain_sum[None, :]
            - chain_means[chain_index][None, :]
            + deleted_chain_means
        ) / n_chains
        samples = np.asarray(
            [transform(mean) for mean in deleted_fused_means],
            dtype=np.float64,
        )
        centered = samples - np.mean(samples, axis=0)
        contribution = (n_blocks - 1) / n_blocks * np.sum(
            centered * centered, axis=0
        )
        variance += contribution
        variance_df_denominator += contribution * contribution / (
            n_blocks - 1
        )
        nominal_degrees_of_freedom += n_blocks - 1
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(variance)):
        raise ValueError("non-finite fused blocking estimate")
    error = np.sqrt(np.maximum(variance, 0.0))
    effective_degrees_of_freedom = np.full_like(
        variance, float(nominal_degrees_of_freedom)
    )
    np.divide(
        variance * variance,
        variance_df_denominator,
        out=effective_degrees_of_freedom,
        where=variance_df_denominator > 0.0,
    )
    roundoff = 1.0e-12 * np.maximum(np.abs(estimate), 1.0)
    error[error <= roundoff] = 0.0
    return estimate, error, effective_degrees_of_freedom


def _blocking_labels(chains: Sequence[PostCutChain]) -> tuple[list[str], int]:
    """Return primary/secondary labels and the number of primary entries."""
    separations = chains[0].raw.correlator.keys[:, 0].astype(np.int64)
    labels = ["Tr_G_real", "Tr_G_imag", "Ctilde_0", "Ctilde_pmin"]
    labels.extend(f"C_G_R{separation}" for separation in separations)
    labels.extend(
        f"plaquette_{profile_label(key)}" for key in chains[0].raw.profile.keys
    )
    n_primary = len(labels)
    labels.extend(("xi_2nd", "Ns_over_xi_2nd"))
    return labels, n_primary


def blocking_error_plateau(
    chains: Sequence[PostCutChain], ns: int
) -> dict[str, object]:
    """Check primary and secondary error plateaus over whole histories.

    Candidate block lengths use a fixed five-sweep-aligned grid. At every
    length, chains are blocked separately and fused only through equal-weight
    replica means. The standard-error uncertainty ``sigma/sqrt(2 dof)`` is the
    large-sample variance-estimator result used in the blocking analysis of
    Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989), DOI
    10.1063/1.457480. Unequal replica contributions use Satterthwaite effective
    degrees of freedom, Biometrics Bulletin 2, 110 (1946), DOI
    10.2307/3002019.
    """
    if not chains:
        raise ValueError("blocking plateau needs at least one chain")
    labels, n_primary = _blocking_labels(chains)
    suite_series = []
    for chain in chains:
        holonomy = np.concatenate((chain.summary, chain.correlator), axis=1)
        holonomy_five_sweep = block_values(holonomy, PROFILE_CADENCE)
        if len(holonomy_five_sweep) != len(chain.profile):
            raise ValueError(f"{chain.raw.directory}: fused 1/5 grids differ")
        suite_series.append(
            np.concatenate((holonomy_five_sweep, chain.profile), axis=1)
        )

    minimum_units = min(len(series) for series in suite_series)
    candidate_units = [
        units
        for units in BLOCKING_CANDIDATE_UNITS
        if minimum_units // units >= MINIMUM_BLOCKS
    ]
    if len(candidate_units) < 3:
        raise ValueError("too few blocking lengths for a three-point plateau")

    def transform(mean: FloatArray) -> FloatArray:
        ratio = mean[2] / mean[3] - 1.0
        if mean[3] <= 0.0 or ratio <= 0.0:
            raise ValueError("xi_2nd is unresolved in fused blocking")
        xi = math.sqrt(ratio) / (2.0 * math.sin(math.pi / ns))
        return np.concatenate((mean, np.asarray([xi, ns / xi])))

    lengths = [PROFILE_CADENCE * units for units in candidate_units]
    errors = np.empty((len(lengths), len(labels)), dtype=np.float64)
    error_uncertainties = np.empty_like(errors)
    effective_degrees_of_freedom = np.empty_like(errors)
    estimates = np.empty_like(errors)
    blocks_per_chain: list[list[int]] = []
    discarded_sweeps_per_chain: list[list[int]] = []
    for length_index, units in enumerate(candidate_units):
        blocks = [block_values(series, units) for series in suite_series]
        estimate, error, degrees_of_freedom = fused_replica_jackknife(
            blocks, transform
        )
        if len(estimate) != len(labels):
            raise ValueError("blocking transform/label size mismatch")
        estimates[length_index] = estimate
        errors[length_index] = error
        effective_degrees_of_freedom[length_index] = degrees_of_freedom
        error_uncertainties[length_index] = error / np.sqrt(
            2.0 * degrees_of_freedom
        )
        blocks_per_chain.append([len(chain_blocks) for chain_blocks in blocks])
        discarded_sweeps_per_chain.append(
            [
                PROFILE_CADENCE * (len(series) % units)
                for series in suite_series
            ]
        )

    tau_minimum = max(chain.block_length_sweeps for chain in chains)
    observable_results: dict[str, object] = {}
    plateau_ends: list[int] = []
    for observable_index, label in enumerate(labels):
        plateau_group: list[int] | None = None
        for start in range(len(lengths) - 2):
            group_lengths = lengths[start : start + 3]
            if group_lengths[0] < tau_minimum:
                continue
            stable = True
            for left in range(start, start + 3):
                for right in range(left + 1, start + 3):
                    scale = max(
                        errors[left, observable_index],
                        errors[right, observable_index],
                    )
                    if scale == 0.0:
                        continue
                    tolerance = max(
                        PLATEAU_RELATIVE_TOLERANCE * scale,
                        PLATEAU_SIGMA_TOLERANCE
                        * math.hypot(
                            error_uncertainties[left, observable_index],
                            error_uncertainties[right, observable_index],
                        ),
                    )
                    if (
                        abs(
                            errors[left, observable_index]
                            - errors[right, observable_index]
                        )
                        > tolerance
                    ):
                        stable = False
            if stable:
                plateau_group = group_lengths
                plateau_ends.append(group_lengths[-1])
                break
        observable_results[label] = {
            "kind": "primary" if observable_index < n_primary else "secondary",
            "estimate": _strict_float_list(estimates[:, observable_index]),
            "error": _strict_float_list(errors[:, observable_index]),
            "error_uncertainty": _strict_float_list(
                error_uncertainties[:, observable_index]
            ),
            "effective_degrees_of_freedom": _strict_float_list(
                effective_degrees_of_freedom[:, observable_index]
            ),
            "plateau_group_sweeps": plateau_group,
            "plateau_found": plateau_group is not None,
        }

    all_found = len(plateau_ends) == len(labels)
    selected_length = max(plateau_ends) if all_found else None
    return {
        "method": (
            "whole post-cut histories; five-sweep elementary vectors; "
            "separate complete blocks per replica; replica-stratified "
            "delete-one-block jackknife; three-length error plateau"
        ),
        "candidate_block_lengths_sweeps": lengths,
        "minimum_tau_block_length_sweeps": tau_minimum,
        "minimum_complete_blocks_per_replica": MINIMUM_BLOCKS,
        "complete_blocks_per_replica": blocks_per_chain,
        "discarded_sweeps_per_replica": discarded_sweeps_per_chain,
        "plateau_relative_tolerance": PLATEAU_RELATIVE_TOLERANCE,
        "plateau_sigma_tolerance": PLATEAU_SIGMA_TOLERANCE,
        "observables": observable_results,
        "all_primary_and_secondary_plateaus_found": all_found,
        "selected_common_block_length_sweeps": selected_length,
    }


def _split_diagnostics(chain: PostCutChain) -> dict[str, object]:
    """Return selected and worst-vector split-history checks."""
    core = {
        "Tr_G_real": split_z_score(chain.summary[:, 0]),
        "Ctilde_0": split_z_score(chain.summary[:, 2]),
        "Ctilde_pmin": split_z_score(chain.summary[:, 3]),
        "C_G_R1": split_z_score(chain.correlator[:, 1]),
    }
    all_values = dict(core)
    all_values.update(
        {
            f"C_G_R{index}": split_z_score(chain.correlator[:, index])
            for index in range(1, chain.correlator.shape[1])
        }
    )
    all_values.update(
        {
            profile_label(chain.raw.profile.keys[index]): split_z_score(
                chain.profile[:, index]
            )
            for index in range(chain.profile.shape[1])
        }
    )
    worst = _largest_named(all_values)
    return {
        "selected": core,
        "maximum": {"observable": worst[0], "z": worst[1]},
    }


def _pairwise_block_diagnostic(
    block_sets: Sequence[FloatArray], labels: Sequence[str]
) -> dict[str, object]:
    """Return the largest autocorrelation-aware pairwise chain discrepancy."""
    maximum = 0.0
    maximum_label = ""
    maximum_pair = ""
    for left_index, left in enumerate(block_sets):
        left_mean = np.mean(left, axis=0)
        left_var = np.var(left, axis=0, ddof=1) / len(left)
        for right_index in range(left_index + 1, len(block_sets)):
            right = block_sets[right_index]
            right_mean = np.mean(right, axis=0)
            right_var = np.var(right, axis=0, ddof=1) / len(right)
            variance = left_var + right_var
            valid = variance > 0.0
            z = np.zeros_like(variance)
            z[valid] = np.abs(left_mean[valid] - right_mean[valid]) / np.sqrt(
                variance[valid]
            )
            index = int(np.argmax(z))
            if z[index] > maximum:
                maximum = float(z[index])
                maximum_label = labels[index]
                maximum_pair = f"{left_index}-{right_index}"
    return {"z": maximum, "observable": maximum_label, "chain_pair": maximum_pair}


def prepare_chains(
    raw_chains: Sequence[ChainData], minimum_cut: int, ns: int,
    prescribed_cuts: Sequence[int] | None = None,
) -> tuple[list[PostCutChain], dict[str, object]]:
    """Select cuts and blocks using tau and fused-replica error plateaus."""
    candidates = [
        candidate_thermalization_cut(chain, minimum_cut) for chain in raw_chains
    ]
    automatic_cuts = [entry[0] for entry in candidates]
    if prescribed_cuts is None:
        cuts = automatic_cuts
    else:
        if len(prescribed_cuts) != len(raw_chains):
            raise ValueError("prescribed cut count does not match chain count")
        cuts = [int(cut) for cut in prescribed_cuts]
        if any(cut < automatic for cut, automatic in zip(cuts, automatic_cuts)):
            raise ValueError("a prescribed cut is below its MSER-5 minimum")
        if any(cut % PROFILE_CADENCE != 0 for cut in cuts):
            raise ValueError("every prescribed cut must be a multiple of five")
    chains = [
        _post_cut(chain, cut)
        for chain, cut in zip(raw_chains, cuts, strict=True)
    ]
    blocking_plateau = blocking_error_plateau(chains, ns)
    selected_block_length = blocking_plateau[
        "selected_common_block_length_sweeps"
    ]
    if selected_block_length is not None:
        chains = [
            _reblock_chain(chain, int(selected_block_length)) for chain in chains
        ]
    separations = raw_chains[0].correlator.keys[:, 0].astype(np.int64)
    labels = ["Tr_G_real", "Tr_G_imag", "Ctilde_0", "Ctilde_pmin"] + [
        f"C_G_R{separation}" for separation in separations
    ]
    profile_labels = [
        profile_label(key) for key in raw_chains[0].profile.keys
    ]
    chain_entries = []
    for index, (chain, candidate) in enumerate(zip(chains, candidates, strict=True)):
        chain_entries.append(
            {
                "chain_index": index,
                "chain": str(chain.raw.directory),
                "automatic_cut_sweeps": automatic_cuts[index],
                "thermalization_cut_sweeps": chain.cut,
                "mser5": candidate[1],
                "retained_holonomy_measurements": len(chain.summary),
                "retained_profile_measurements": len(chain.profile),
                "tau_int_sweeps": chain.tau,
                "block_length_sweeps": chain.block_length_sweeps,
                "n_complete_blocks": len(chain.primary_blocks),
                "target_50_blocks_met": len(chain.primary_blocks) >= TARGET_BLOCKS,
                "discarded_partial_sweeps": len(chain.summary)
                - len(chain.primary_blocks) * chain.block_length_sweeps,
                "split_history": _split_diagnostics(chain),
            }
        )
    holonomy_between = _pairwise_block_diagnostic(
        [chain.primary_blocks for chain in chains], labels
    )
    profile_between = _pairwise_block_diagnostic(
        [chain.profile_blocks for chain in chains], profile_labels
    )
    maximum_between = max(
        (holonomy_between, profile_between), key=lambda entry: float(entry["z"])
    )
    diagnostics = {
        "method": (
            "per-chain MSER-5 with a 500-sweep minimum; Sokal-window tau_int; "
            "whole-history fused-replica error plateaus; separate complete "
            "within-chain blocks longer than 2 tau_int"
        ),
        "minimum_cut_sweeps": minimum_cut,
        "profile_cadence_sweeps": PROFILE_CADENCE,
        "recommended_per_chain_cuts": cuts,
        "chains": chain_entries,
        "between_chain": {
            "holonomy": holonomy_between,
            "plaquette_profiles": profile_between,
            "maximum": maximum_between,
        },
        "blocking_error_plateau": blocking_plateau,
    }
    return chains, diagnostics


def second_moment_samples(samples: FloatArray, ns: int) -> FloatArray:
    """Return xi_2nd for bootstrap means containing the four summary fields."""
    c_zero = samples[:, 2]
    c_minimum = samples[:, 3]
    ratio = c_zero / c_minimum - 1.0
    if np.any(c_minimum <= 0.0) or np.any(ratio <= 0.0):
        raise ValueError("xi_2nd is unresolved in at least one bootstrap sample")
    return np.sqrt(ratio) / (2.0 * np.sin(np.pi / ns))


def finite_volume_k0(
    separation: FloatArray, amplitude: float, mass: float, ns: int
) -> FloatArray:
    """Return ``A[K0(mR)+K0(m(Ns-R))]``."""
    return amplitude * (
        special.k0(mass * separation)
        + special.k0(mass * (ns - separation))
    )


def correlated_k0_fit(
    separation: FloatArray,
    values: FloatArray,
    covariance: FloatArray,
    ns: int,
    initial: FloatArray | None = None,
) -> dict[str, float]:
    """Fit the periodic K0 form using the full fixed covariance matrix."""
    if len(separation) < 5 or np.any(separation <= 0.0):
        raise ValueError("a K0 fit requires at least five positive distances")
    scale = linalg.cholesky(covariance, lower=True)
    return _correlated_k0_fit_with_scale(separation, values, scale, ns, initial)


def _correlated_k0_fit_with_scale(
    separation: FloatArray,
    values: FloatArray,
    scale: FloatArray,
    ns: int,
    initial: FloatArray | None = None,
) -> dict[str, float]:
    """Fit with a precomputed covariance Cholesky factor."""

    def residual(parameters: FloatArray) -> FloatArray:
        model = finite_volume_k0(separation, parameters[0], parameters[1], ns)
        return linalg.solve_triangular(scale, values - model, lower=True)

    if initial is None:
        initial_mass = max(1.0 / ns, 0.05)
        normalization = finite_volume_k0(
            np.asarray([separation[0]]), 1.0, initial_mass, ns
        )[0]
        initial = np.asarray(
            [max(abs(values[0]) / normalization, 1.0e-8), initial_mass]
        )
    fit = optimize.least_squares(
        residual,
        x0=np.asarray(initial, dtype=np.float64),
        bounds=(np.asarray([1.0e-15, 1.0e-15]), np.asarray([np.inf, np.inf])),
        max_nfev=300,
    )
    if not fit.success or np.any(~np.isfinite(fit.x)) or np.any(fit.x <= 0.0):
        raise RuntimeError(f"K0 fit failed: {fit.message}")
    chi2 = float(np.dot(fit.fun, fit.fun))
    dof = len(separation) - 2
    return {
        "amplitude": float(fit.x[0]),
        "mass": float(fit.x[1]),
        "chi2": chi2,
        "dof": dof,
        "p_value": float(stats.chi2.sf(chi2, dof)),
    }


def scan_k0_fits(
    separations: IntArray,
    correlator: FloatArray,
    covariance: FloatArray,
    bootstrap_correlators: FloatArray,
    ns: int,
) -> tuple[float | None, float | None, list[dict[str, object]]]:
    """Apply the preregistered correlated-fit and stability acceptance rules."""
    r_max = int(separations[-1])
    maximum_r_min = min(8, r_max - 4)
    entries: list[dict[str, object]] = []
    for r_min in range(1, maximum_r_min + 1):
        selected = (separations >= r_min) & (separations <= r_max)
        entry: dict[str, object] = {
            "r_min": r_min,
            "r_max": r_max,
            "n_points": int(np.count_nonzero(selected)),
            "correlated": True,
        }
        selected_covariance = covariance[np.ix_(selected, selected)]
        try:
            scale = linalg.cholesky(selected_covariance, lower=True)
            central = _correlated_k0_fit_with_scale(
                separations[selected].astype(np.float64),
                correlator[selected],
                scale,
                ns,
            )
        except (ValueError, RuntimeError, linalg.LinAlgError) as error:
            entry["error"] = str(error)
            entry["quality_pass"] = False
            entries.append(entry)
            continue

        initial = np.asarray([central["amplitude"], central["mass"]])
        if central["p_value"] < 0.05:
            entry.update(
                {
                    **central,
                    "bootstrap_success_fraction": 0.0,
                    "quality_pass": False,
                    "stable_three_range_group": False,
                }
            )
            entries.append(entry)
            continue
        xi_samples: list[float] = []
        for sample in bootstrap_correlators:
            try:
                fit = _correlated_k0_fit_with_scale(
                    separations[selected].astype(np.float64),
                    sample[selected],
                    scale,
                    ns,
                    initial,
                )
            except (ValueError, RuntimeError, linalg.LinAlgError):
                continue
            xi_samples.append(1.0 / fit["mass"])
        success_fraction = len(xi_samples) / len(bootstrap_correlators)
        xi = 1.0 / central["mass"]
        xi_error = (
            float(np.std(np.asarray(xi_samples), ddof=1))
            if len(xi_samples) >= 2
            else math.inf
        )
        relative_error = xi_error / xi
        quality_pass = bool(
            entry["n_points"] >= 5
            and central["p_value"] >= 0.05
            and central["mass"] > 0.0
            and success_fraction >= 0.90
            and relative_error < 0.25
        )
        entry.update(
            {
                **central,
                "xi_exp": xi,
                "xi_exp_bootstrap_error": xi_error,
                "relative_error": relative_error,
                "bootstrap_success_fraction": success_fraction,
                "quality_pass": quality_pass,
                "stable_three_range_group": False,
            }
        )
        entries.append(entry)

    selected_index: int | None = None
    for start in range(len(entries) - 2):
        group = entries[start : start + 3]
        if not all(bool(entry.get("quality_pass", False)) for entry in group):
            continue
        stable = True
        for left in range(3):
            for right in range(left + 1, 3):
                difference = abs(float(group[left]["xi_exp"]) - float(group[right]["xi_exp"]))
                combined = math.hypot(
                    float(group[left]["xi_exp_bootstrap_error"]),
                    float(group[right]["xi_exp_bootstrap_error"]),
                )
                if combined <= 0.0 or difference > 2.0 * combined:
                    stable = False
        if stable:
            for entry in group:
                entry["stable_three_range_group"] = True
            selected_index = start + 1
            break
    if selected_index is None:
        return None, None, entries
    selected = entries[selected_index]
    selected["selected"] = True
    return (
        float(selected["xi_exp"]),
        float(selected["xi_exp_bootstrap_error"]),
        entries,
    )


def _strict_float_list(values: FloatArray) -> list[float]:
    """Convert a finite float64 array to strict JSON values."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot serialize non-finite values")
    return [float(value) for value in array]


def analyse_ensemble(
    chains: Sequence[PostCutChain],
    diagnostics: dict[str, object],
    nt: int,
    ns: int,
    beta: float,
    purpose: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    covariance_output: Path,
) -> dict[str, object]:
    """Produce the full Step-3 result for one four-chain ensemble."""
    if bootstrap_samples < 2:
        raise ValueError("at least two bootstrap samples are required")
    plateau = diagnostics["blocking_error_plateau"]
    if not bool(plateau["all_primary_and_secondary_plateaus_found"]):
        raise ValueError(
            "full analysis requires blocking-error plateaus for every "
            "primary and secondary observable"
        )
    separations = chains[0].raw.correlator.keys[:, 0].astype(np.int64)
    if not np.array_equal(separations, np.arange(ns // 2 + 1)):
        raise ValueError("correlator grid must be R=0,...,floor(Ns/2)")

    mean_vector, mean_samples, _ = hierarchical_bootstrap(
        [chain.primary_blocks for chain in chains],
        lambda mean: mean,
        bootstrap_samples,
        np.random.default_rng(bootstrap_seed),
    )
    correlator = mean_vector[4:]
    bootstrap_correlators = mean_samples[:, 4:]
    correlator_covariance = np.atleast_2d(
        np.cov(bootstrap_correlators, rowvar=False, ddof=1)
    )
    correlator_error = np.sqrt(np.maximum(np.diag(correlator_covariance), 0.0))
    if correlator[0] != 1.0 or correlator_error[0] > 1.0e-14:
        raise ValueError("bootstrap did not preserve exact C_G(0)=1")

    xi_samples = second_moment_samples(mean_samples, ns)
    xi = math.sqrt(mean_vector[2] / mean_vector[3] - 1.0) / (
        2.0 * math.sin(math.pi / ns)
    )
    ratio_samples = ns / xi_samples
    ratio = ns / xi
    xi_interval = np.quantile(xi_samples, [0.16, 0.84])
    ratio_interval = np.quantile(ratio_samples, [0.16, 0.84])

    profile_mean, profile_samples, profile_covariance = hierarchical_bootstrap(
        [chain.profile_blocks for chain in chains],
        lambda mean: mean,
        bootstrap_samples,
        np.random.default_rng(bootstrap_seed + 1),
    )
    xi_exp, xi_exp_error, fit_scan = scan_k0_fits(
        separations,
        correlator,
        correlator_covariance,
        bootstrap_correlators,
        ns,
    )

    covariance_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        covariance_output,
        R=separations,
        C_G=correlator,
        C_G_covariance=correlator_covariance,
        profile_kind_t=chains[0].raw.profile.keys,
        profile_mean=profile_mean,
        profile_covariance=profile_covariance,
        xi_2nd_bootstrap=xi_samples,
        Ns_over_xi_2nd_bootstrap=ratio_samples,
    )
    return {
        "purpose": purpose,
        "physical_Nt": nt,
        "stored_L2": nt + 1,
        "Ns": ns,
        "beta": beta,
        "lambda_squared_tree": 4.0 * nt / beta,
        "boundary_conditions": "temporal Dirichlet; periodic spatial",
        "sweep_definition": "one heatbath sweep plus five overrelaxation sweeps",
        "measurement_interval_sweeps": 1,
        "dirichlet_suite_measurement_interval_sweeps": PROFILE_CADENCE,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "thermalization_and_blocking": diagnostics,
        "R": separations.tolist(),
        "C_G": _strict_float_list(correlator),
        "C_G_error": _strict_float_list(correlator_error),
        "mean_Tr_G_real": float(mean_vector[0]),
        "mean_Tr_G_real_error": float(np.std(mean_samples[:, 0], ddof=1)),
        "Ctilde_0": float(mean_vector[2]),
        "Ctilde_pmin": float(mean_vector[3]),
        "xi_2nd": xi,
        "xi_2nd_error": float(np.std(xi_samples, ddof=1)),
        "xi_2nd_68_percent_interval": _strict_float_list(xi_interval),
        "Ns_over_xi_2nd": ratio,
        "Ns_over_xi_2nd_error": float(np.std(ratio_samples, ddof=1)),
        "Ns_over_xi_2nd_68_percent_interval": _strict_float_list(
            ratio_interval
        ),
        "xi_exp": xi_exp,
        "xi_exp_error": xi_exp_error,
        "mass": None if xi_exp is None else 1.0 / xi_exp,
        "mass_error": (
            None
            if xi_exp is None or xi_exp_error is None
            else xi_exp_error / (xi_exp * xi_exp)
        ),
        "K0_correlated_fit_scan": fit_scan,
        "plaquette_profile": {
            "kind_t": chains[0].raw.profile.keys.tolist(),
            "mean": _strict_float_list(profile_mean),
            "error": _strict_float_list(
                np.sqrt(np.maximum(np.diag(profile_covariance), 0.0))
            ),
        },
    }


def _load_prescribed_cuts(path: Path, chains: Sequence[ChainData]) -> list[int]:
    """Load inspected cuts and verify their chain ordering."""
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    recorded = [Path(entry["chain"]) for entry in diagnostics["chains"]]
    expected = [chain.directory for chain in chains]
    if recorded != expected:
        raise ValueError("thermalization diagnostic chain order does not match")
    return [int(cut) for cut in diagnostics["recommended_per_chain_cuts"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--chain-directory", type=Path, action="append")
    parser.add_argument("--nt", type=int, required=True)
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--minimum-cut", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_721)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--thermalization-diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--covariance-output", type=Path)
    args = parser.parse_args()
    if args.nt < 1 or args.ns < 2 or args.beta <= 0.0 or args.minimum_cut < 0:
        raise ValueError("invalid ensemble parameters")

    raw_chains = load_ensemble(args.ensemble_dir, args.chain_directory)
    prescribed = (
        _load_prescribed_cuts(args.thermalization_diagnostics, raw_chains)
        if args.thermalization_diagnostics is not None
        else None
    )
    chains, diagnostics = prepare_chains(
        raw_chains, args.minimum_cut, args.ns, prescribed
    )
    if args.diagnostics_only:
        result = diagnostics
    else:
        if args.covariance_output is None:
            parser.error("--covariance-output is required for full analysis")
        result = analyse_ensemble(
            chains,
            diagnostics,
            args.nt,
            args.ns,
            args.beta,
            args.purpose,
            args.bootstrap_samples,
            args.bootstrap_seed,
            args.covariance_output,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
