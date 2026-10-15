#!/usr/bin/env python3
"""Chain-aware analysis for the fixed-beta/Nt Dirichlet-slab campaign.

MSER-5 and the Sokal-window autocorrelation estimator follow the references in
``thermalization_diagnostics.py``. Independent chains are blocked separately;
the seeded two-level resampling follows the references in
``pcm_beta_pilot_analysis.py``. Periodic K0 fits use Eq. (4.14) of M. Nebbia,
*Effects of Dirichlet Boundary Conditions in Yang-Mills Theories* (2024/2025
MSc thesis). Fit parameters and bootstrap refits use diagonal covariance,
while the complete covariance enters a recentered-bootstrap goodness-of-fit
statistic. No raw histories are concatenated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, special


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from dirichlet_analysis import (
    FloatArray,
    KeyedChain,
    block_values,
    integrated_autocorrelation_time,
    load_keyed_chain,
    require_matching_grids,
)
from pcm_beta_pilot_analysis import chainwise_bootstrap
from thermalization_diagnostics import mser5_cut, split_z_score


IntArray = NDArray[np.int64]
PROFILE_CADENCE = 5
MINIMUM_BLOCKS = 30
TARGET_BLOCKS = 50
PLATEAU_RELATIVE_TOLERANCE = 0.05
PLATEAU_SIGMA_TOLERANCE = 2.0
BLOCKING_CANDIDATE_UNITS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
)
RANGE_SCAN_BOOTSTRAP_SAMPLES = 500
# User policy updated 2026-10-12: exactly 500 replicas for new operations.
# Historical 10000-replica products keep their original provenance on disk.
ANALYSIS_BOOTSTRAP_SAMPLES = (500,)


@dataclass(frozen=True)
class ChainData:
    """Validated raw observable tables for one independent chain."""

    directory: Path
    summary: KeyedChain
    correlator: KeyedChain
    profile: KeyedChain
    boundary: KeyedChain | None


@dataclass(frozen=True)
class PostCutChain:
    """Post-thermalization arrays and chain-specific blocking metadata."""

    raw: ChainData
    cut: int
    summary: FloatArray
    correlator: FloatArray
    profile: FloatArray
    boundary: FloatArray | None
    tau: dict[str, object]
    block_length_sweeps: int
    primary_blocks: FloatArray
    profile_blocks: FloatArray
    boundary_blocks: FloatArray | None


@dataclass(frozen=True)
class K0ScanResult:
    """Selected periodic-K0 fit and its complete bootstrap replicas."""

    xi_exp: float | None
    xi_exp_error: float | None
    mass: float | None
    mass_error: float | None
    xi_exp_bootstrap: FloatArray
    mass_bootstrap: FloatArray
    entries: list[dict[str, object]]


def load_chain(directory: Path, maximum_sweep: int | None = None) -> ChainData:
    """Load one chain, optionally frozen at a completed common sweep."""
    summary = load_keyed_chain(
        directory / "holonomy_summary.out",
        (),
        (1, 2, 3, 4),
        0,
        maximum_step=maximum_sweep,
    )
    correlator = load_keyed_chain(
        directory / "holonomy_correlator.out",
        (1,),
        (2,),
        0,
        maximum_step=maximum_sweep,
    )
    profile = load_keyed_chain(
        directory / "plaquette_profile.out",
        (1, 2),
        (3,),
        0,
        maximum_step=maximum_sweep,
    )
    boundary_path = directory / "boundary_wilson.out"
    boundary = (
        load_keyed_chain(
            boundary_path,
            (1, 2),
            (3,),
            0,
            maximum_step=maximum_sweep,
        )
        if boundary_path.exists()
        else None
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
    if boundary is not None:
        cadence = regular_measurement_cadence(boundary)
        if cadence % PROFILE_CADENCE != 0:
            raise ValueError(
                f"{directory}: boundary cadence is not a multiple of five"
            )
        expected_boundary_steps = summary.steps[
            np.asarray(summary.steps, dtype=np.int64) % cadence == 0
        ]
        if not np.array_equal(boundary.steps, expected_boundary_steps):
            raise ValueError(f"{directory}: incomplete boundary-loop cadence")
        r_h = boundary.keys.astype(np.int64)
        zero_r = r_h[:, 0] == 0
        if not np.all(boundary.values[:, zero_r, 0] == 1.0):
            raise ValueError(f"{directory}: W_b(0,H) is not exactly one")
        maximum_height = int(np.max(r_h[:, 1]))
        full_height = np.flatnonzero(r_h[:, 1] == maximum_height)
        summary_indices = np.searchsorted(summary.steps, boundary.steps)
        for boundary_index in full_height:
            separation = int(r_h[boundary_index, 0])
            if separation >= correlator.values.shape[1]:
                continue
            if not np.allclose(
                boundary.values[:, boundary_index, 0],
                correlator.values[summary_indices, separation, 0],
                rtol=0.0,
                atol=2.0e-11,
            ):
                raise ValueError(
                    f"{directory}: W_b(R,Nt) does not match C_G(R)"
                )
    return ChainData(directory, summary, correlator, profile, boundary)


def load_ensemble(
    directory: Path,
    chain_directories: Sequence[Path] | None = None,
    maximum_sweep: int | None = None,
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
    chains = [load_chain(path, maximum_sweep) for path in selected_directories]
    require_matching_grids([chain.correlator for chain in chains])
    require_matching_grids([chain.profile for chain in chains])
    boundary_presence = [chain.boundary is not None for chain in chains]
    if any(boundary_presence) and not all(boundary_presence):
        raise ValueError(f"{directory}: boundary output is missing for a chain")
    if all(boundary_presence):
        require_matching_grids(
            [chain.boundary for chain in chains if chain.boundary is not None]
        )
    return chains


def regular_measurement_cadence(chain: KeyedChain) -> int:
    """Return a positive regular cadence whose first measurement is on-grid."""
    steps = np.asarray(chain.steps, dtype=np.int64)
    if len(steps) < 2:
        raise ValueError(f"{chain.path}: need at least two measurements")
    differences = np.diff(steps)
    cadence = int(differences[0])
    if cadence <= 0 or not np.all(differences == cadence):
        raise ValueError(f"{chain.path}: measurement cadence is not regular")
    if np.any(steps % cadence != 0):
        raise ValueError(f"{chain.path}: measurement steps are off cadence")
    return cadence


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
    """Return the per-chain MSER-5 cut on every enabled observable grid."""
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
    boundary_cadence = (
        regular_measurement_cadence(chain.boundary)
        if chain.boundary is not None
        else PROFILE_CADENCE
    )
    boundary_candidates = (
        {
            boundary_label(chain.boundary.keys[index]): boundary_cadence
            * mser5_cut(chain.boundary.values[:, index, 0])
            for index in range(chain.boundary.values.shape[1])
            if int(chain.boundary.keys[index, 0]) > 0
        }
        if chain.boundary is not None
        else {}
    )
    largest_boundary = (
        _largest_named(boundary_candidates) if boundary_candidates else None
    )
    raw_cut = max(
        minimum_cut,
        *scalar_candidates.values(),
        largest_correlator[1],
        largest_profile[1],
        largest_boundary[1] if largest_boundary is not None else 0.0,
    )
    cut_cadence = math.lcm(PROFILE_CADENCE, boundary_cadence)
    cut = int(cut_cadence * math.ceil(raw_cut / cut_cadence))
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
        "largest_boundary_candidate": (
            {
                "observable": largest_boundary[0],
                "sweeps": largest_boundary[1],
            }
            if largest_boundary is not None
            else None
        ),
        "cut_cadence_sweeps": cut_cadence,
    }


def profile_label(key: FloatArray) -> str:
    """Return an unambiguous profile-coordinate label."""
    kind = "spatial" if int(key[0]) == 0 else "temporal"
    return f"{kind}_t_{float(key[1]):g}"


def boundary_label(key: FloatArray) -> str:
    """Return an unambiguous boundary-loop label."""
    return f"W_b_R_{int(key[0])}_H_{int(key[1])}"


def tau_diagnostics(
    correlator: FloatArray, summary: FloatArray, profile: FloatArray,
    profile_keys: FloatArray, boundary: FloatArray | None = None,
    boundary_keys: FloatArray | None = None, boundary_cadence: int = 0,
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
    boundary_taus = (
        {
            boundary_label(boundary_keys[index]): boundary_cadence
            * integrated_autocorrelation_time(boundary[:, index])
            for index in range(boundary.shape[1])
            if int(boundary_keys[index, 0]) > 0
        }
        if boundary is not None and boundary_keys is not None
        else {}
    )
    largest_boundary = _largest_named(boundary_taus) if boundary_taus else None
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
        "boundary_wilson_loops": boundary_taus,
        "largest_boundary_wilson_loop": (
            {
                "observable": largest_boundary[0],
                "tau_int_sweeps": largest_boundary[1],
            }
            if largest_boundary is not None
            else None
        ),
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
    if tau["largest_boundary_wilson_loop"] is not None:
        scalar.append(
            float(
                tau["largest_boundary_wilson_loop"]["tau_int_sweeps"]
            )
        )
    return max(scalar)


def choose_block_length_sweeps(
    tau_int_sweeps: float, cadence_sweeps: int = PROFILE_CADENCE
) -> int:
    """Choose the smallest cadence multiple strictly exceeding ``2 tau``."""
    if not np.isfinite(tau_int_sweeps) or tau_int_sweeps < 0.5:
        raise ValueError("tau_int must be finite and at least 0.5 sweeps")
    if cadence_sweeps <= 0 or cadence_sweeps % PROFILE_CADENCE != 0:
        raise ValueError("block cadence must be a positive multiple of five")
    minimum = math.floor(2.0 * tau_int_sweeps) + 1
    length = cadence_sweeps * math.ceil(minimum / cadence_sweeps)
    if length <= 2.0 * tau_int_sweeps:
        length += cadence_sweeps
    return int(length)


def _post_cut(chain: ChainData, cut: int) -> PostCutChain:
    """Apply one cut and form complete within-chain block vectors."""
    summary_mask = chain.summary.steps > cut
    correlator_mask = chain.correlator.steps > cut
    profile_mask = chain.profile.steps > cut
    boundary_mask = (
        chain.boundary.steps > cut if chain.boundary is not None else None
    )
    summary = chain.summary.values[summary_mask, 0, :]
    correlator = chain.correlator.values[correlator_mask, :, 0]
    profile = chain.profile.values[profile_mask, :, 0]
    boundary = (
        chain.boundary.values[boundary_mask, :, 0]
        if chain.boundary is not None and boundary_mask is not None
        else None
    )
    if len(summary) != len(correlator) or len(summary) < 4:
        raise ValueError(f"{chain.directory}: invalid post-cut holonomy grid")
    if len(summary) != PROFILE_CADENCE * len(profile):
        raise ValueError(f"{chain.directory}: post-cut 1/5 grids are misaligned")

    boundary_cadence = (
        regular_measurement_cadence(chain.boundary)
        if chain.boundary is not None
        else PROFILE_CADENCE
    )
    if boundary is not None and len(summary) != boundary_cadence * len(boundary):
        raise ValueError(f"{chain.directory}: post-cut boundary grid is misaligned")
    tau = tau_diagnostics(
        correlator,
        summary,
        profile,
        chain.profile.keys,
        boundary,
        chain.boundary.keys if chain.boundary is not None else None,
        boundary_cadence,
    )
    block_cadence = math.lcm(PROFILE_CADENCE, boundary_cadence)
    block_length = choose_block_length_sweeps(
        maximum_required_tau(tau), block_cadence
    )
    combined = np.concatenate((summary, correlator), axis=1)
    primary_blocks = block_values(combined, block_length)
    profile_blocks = block_values(profile, block_length // PROFILE_CADENCE)
    boundary_blocks = (
        block_values(boundary, block_length // boundary_cadence)
        if boundary is not None
        else None
    )
    if len(primary_blocks) != len(profile_blocks) or (
        boundary_blocks is not None and len(primary_blocks) != len(boundary_blocks)
    ):
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
        boundary,
        tau,
        block_length,
        primary_blocks,
        profile_blocks,
        boundary_blocks,
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
    boundary_cadence = (
        regular_measurement_cadence(chain.raw.boundary)
        if chain.raw.boundary is not None
        else PROFILE_CADENCE
    )
    if block_length_sweeps % boundary_cadence != 0:
        raise ValueError("block length must be a multiple of boundary cadence")
    boundary_blocks = (
        block_values(
            chain.boundary, block_length_sweeps // boundary_cadence
        )
        if chain.boundary is not None
        else None
    )
    if len(primary_blocks) != len(profile_blocks) or (
        boundary_blocks is not None and len(primary_blocks) != len(boundary_blocks)
    ):
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
        boundary_blocks=boundary_blocks,
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
    if chains[0].raw.boundary is not None:
        labels.extend(
            boundary_label(key) for key in chains[0].raw.boundary.keys
        )
    n_primary = len(labels)
    labels.extend(("xi_2nd", "Ns_over_xi_2nd"))
    return labels, n_primary


def stable_three_length_groups(
    lengths: Sequence[int],
    errors: FloatArray,
    error_uncertainties: FloatArray,
    minimum_length: int,
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Return per-observable and common stable three-length groups."""
    if errors.shape != error_uncertainties.shape:
        raise ValueError("blocking error arrays have different shapes")
    if errors.shape[0] != len(lengths):
        raise ValueError("blocking lengths and error arrays differ")
    groups_by_observable: list[list[list[int]]] = [
        [] for _ in range(errors.shape[1])
    ]
    common_groups: list[list[int]] = []
    for start in range(len(lengths) - 2):
        group = list(lengths[start : start + 3])
        if group[0] < minimum_length:
            continue
        stable_observables = np.ones(errors.shape[1], dtype=bool)
        for left in range(start, start + 3):
            for right in range(left + 1, start + 3):
                scale = np.maximum(errors[left], errors[right])
                tolerance = np.maximum(
                    PLATEAU_RELATIVE_TOLERANCE * scale,
                    PLATEAU_SIGMA_TOLERANCE
                    * np.hypot(
                        error_uncertainties[left],
                        error_uncertainties[right],
                    ),
                )
                stable_observables &= (
                    np.abs(errors[left] - errors[right]) <= tolerance
                ) | (scale == 0.0)
        for observable_index in np.flatnonzero(stable_observables):
            groups_by_observable[int(observable_index)].append(group)
        if np.all(stable_observables):
            common_groups.append(group)
    return groups_by_observable, common_groups


def blocking_error_plateau(
    chains: Sequence[PostCutChain], ns: int
) -> dict[str, object]:
    """Check primary and secondary error plateaus over whole histories.

    Candidate block lengths use a fixed grid in units of the least common
    multiple of every enabled measurement cadence. At every length, chains
    are blocked separately and fused only through equal-weight replica means.
    The standard-error uncertainty ``sigma/sqrt(2 dof)`` is the
    large-sample variance-estimator result used in the blocking analysis of
    Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989), DOI
    10.1063/1.457480. Unequal replica contributions use Satterthwaite effective
    degrees of freedom, Biometrics Bulletin 2, 110 (1946), DOI
    10.2307/3002019.
    """
    if not chains:
        raise ValueError("blocking plateau needs at least one chain")
    labels, n_primary = _blocking_labels(chains)
    boundary_cadence = (
        regular_measurement_cadence(chains[0].raw.boundary)
        if chains[0].raw.boundary is not None
        else PROFILE_CADENCE
    )
    elementary_cadence = math.lcm(PROFILE_CADENCE, boundary_cadence)
    suite_series = []
    for chain in chains:
        holonomy = np.concatenate((chain.summary, chain.correlator), axis=1)
        holonomy_elementary = block_values(holonomy, elementary_cadence)
        profile_elementary = block_values(
            chain.profile, elementary_cadence // PROFILE_CADENCE
        )
        parts = [holonomy_elementary, profile_elementary]
        if chain.boundary is not None:
            parts.append(
                block_values(
                    chain.boundary,
                    elementary_cadence // boundary_cadence,
                )
            )
        if len({len(part) for part in parts}) != 1:
            raise ValueError(
                f"{chain.raw.directory}: fused observable grids differ"
            )
        suite_series.append(np.concatenate(parts, axis=1))

    candidate_lengths = [
        elementary_cadence * units
        for units in BLOCKING_CANDIDATE_UNITS
        if min(len(series) for series in suite_series) // units
        >= MINIMUM_BLOCKS
    ]
    if len(candidate_lengths) < 3:
        raise ValueError("too few blocking lengths for a three-point plateau")

    def transform(mean: FloatArray) -> FloatArray:
        ratio = mean[2] / mean[3] - 1.0
        if mean[3] <= 0.0 or ratio <= 0.0:
            raise ValueError("xi_2nd is unresolved in fused blocking")
        xi = math.sqrt(ratio) / (2.0 * math.sin(math.pi / ns))
        return np.concatenate((mean, np.asarray([xi, ns / xi])))

    lengths = candidate_lengths
    errors = np.empty((len(lengths), len(labels)), dtype=np.float64)
    error_uncertainties = np.empty_like(errors)
    effective_degrees_of_freedom = np.empty_like(errors)
    estimates = np.empty_like(errors)
    blocks_per_chain: list[list[int]] = []
    discarded_sweeps_per_chain: list[list[int]] = []
    for length_index, length in enumerate(lengths):
        units = length // elementary_cadence
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
                elementary_cadence * (len(series) % units)
                for series in suite_series
            ]
        )

    tau_minimum = max(chain.block_length_sweeps for chain in chains)
    groups_by_observable, common_groups = stable_three_length_groups(
        lengths, errors, error_uncertainties, tau_minimum
    )
    observable_results: dict[str, object] = {}
    for observable_index, label in enumerate(labels):
        groups = groups_by_observable[observable_index]
        plateau_group = groups[0] if groups else None
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
            "all_plateau_groups_sweeps": groups,
            "plateau_found": plateau_group is not None,
        }

    common_group = common_groups[0] if common_groups else None
    all_found = common_group is not None
    selected_length = common_group[-1] if common_group is not None else None
    return {
        "method": (
            "whole post-cut histories; common-cadence elementary vectors; "
            "separate complete blocks per replica; replica-stratified "
            "delete-one-block jackknife; three-length error plateau"
        ),
        "elementary_cadence_sweeps": elementary_cadence,
        "candidate_block_lengths_sweeps": lengths,
        "minimum_tau_block_length_sweeps": tau_minimum,
        "minimum_complete_blocks_per_replica": MINIMUM_BLOCKS,
        "complete_blocks_per_replica": blocks_per_chain,
        "discarded_sweeps_per_replica": discarded_sweeps_per_chain,
        "plateau_relative_tolerance": PLATEAU_RELATIVE_TOLERANCE,
        "plateau_sigma_tolerance": PLATEAU_SIGMA_TOLERANCE,
        "common_plateau_group_sweeps": common_group,
        "all_common_plateau_groups_sweeps": common_groups,
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
    if chain.boundary is not None and chain.raw.boundary is not None:
        all_values.update(
            {
                boundary_label(chain.raw.boundary.keys[index]): split_z_score(
                    chain.boundary[:, index]
                )
                for index in range(chain.boundary.shape[1])
                if int(chain.raw.boundary.keys[index, 0]) > 0
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
        for cut, chain in zip(cuts, raw_chains, strict=True):
            boundary_cadence = (
                regular_measurement_cadence(chain.boundary)
                if chain.boundary is not None
                else PROFILE_CADENCE
            )
            if cut % math.lcm(PROFILE_CADENCE, boundary_cadence) != 0:
                raise ValueError(
                    "a prescribed cut is off an enabled measurement cadence"
                )
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
    boundary_labels = (
        [boundary_label(key) for key in raw_chains[0].boundary.keys]
        if raw_chains[0].boundary is not None
        else []
    )
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
                "retained_boundary_measurements": (
                    len(chain.boundary) if chain.boundary is not None else 0
                ),
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
    boundary_between = (
        _pairwise_block_diagnostic(
            [
                chain.boundary_blocks
                for chain in chains
                if chain.boundary_blocks is not None
            ],
            boundary_labels,
        )
        if boundary_labels
        else None
    )
    between_results = [holonomy_between, profile_between]
    if boundary_between is not None:
        between_results.append(boundary_between)
    maximum_between = max(
        between_results, key=lambda entry: float(entry["z"])
    )
    boundary_cadence = (
        regular_measurement_cadence(raw_chains[0].boundary)
        if raw_chains[0].boundary is not None
        else None
    )
    diagnostics = {
        "method": (
            "per-chain MSER-5 with a 500-sweep minimum; Sokal-window tau_int; "
            "whole-history fused-replica error plateaus; separate complete "
            "within-chain blocks longer than 2 tau_int"
        ),
        "minimum_cut_sweeps": minimum_cut,
        "profile_cadence_sweeps": PROFILE_CADENCE,
        "boundary_wilson_cadence_sweeps": boundary_cadence,
        "recommended_per_chain_cuts": cuts,
        "chains": chain_entries,
        "between_chain": {
            "holonomy": holonomy_between,
            "plaquette_profiles": profile_between,
            "boundary_wilson_loops": boundary_between,
            "maximum": maximum_between,
        },
        "blocking_error_plateau": blocking_plateau,
    }
    return chains, diagnostics


def second_moment_squared_samples(samples: FloatArray, ns: int) -> FloatArray:
    """Return signed bootstrap estimates of ``xi_2nd**2``."""
    c_zero = samples[:, 2]
    c_minimum = samples[:, 3]
    if np.any(c_minimum <= 0.0):
        raise ValueError("Ctilde_pmin is nonpositive in a bootstrap sample")
    ratio = c_zero / c_minimum - 1.0
    return ratio / (2.0 * np.sin(np.pi / ns)) ** 2


def second_moment_samples(samples: FloatArray, ns: int) -> FloatArray:
    """Return ``xi_2nd``, with unresolved bootstrap samples marked NaN."""
    squared = second_moment_squared_samples(samples, ns)
    result = np.full_like(squared, np.nan)
    positive = squared > 0.0
    result[positive] = np.sqrt(squared[positive])
    return result


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
    """Fit the periodic K0 form using the complete fixed covariance."""
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
    """Fit using a precomputed complete-covariance Cholesky factor."""

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
    return {
        "amplitude": float(fit.x[0]),
        "mass": float(fit.x[1]),
        "chi2": float(np.dot(fit.fun, fit.fun)),
        "dof": len(separation) - 2,
    }


def uncorrelated_k0_fit(
    separation: FloatArray,
    values: FloatArray,
    covariance: FloatArray,
    ns: int,
    initial: FloatArray | None = None,
) -> dict[str, float]:
    """Fit the periodic K0 form using only covariance diagonal errors."""
    if len(separation) < 5 or np.any(separation <= 0.0):
        raise ValueError("a K0 fit requires at least five positive distances")
    if covariance.shape != (len(separation), len(separation)):
        raise ValueError("K0 covariance has the wrong shape")
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return _uncorrelated_k0_fit_with_errors(
        separation, values, errors, ns, initial
    )


def _uncorrelated_k0_fit_with_errors(
    separation: FloatArray,
    values: FloatArray,
    errors: FloatArray,
    ns: int,
    initial: FloatArray | None = None,
) -> dict[str, float]:
    """Fit with fixed marginal errors from blocked vector estimates."""
    if errors.shape != separation.shape:
        raise ValueError("K0 marginal errors have the wrong shape")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0.0):
        raise ValueError("K0 marginal errors must be finite and positive")

    def residual(parameters: FloatArray) -> FloatArray:
        model = finite_volume_k0(separation, parameters[0], parameters[1], ns)
        return (values - model) / errors

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
        "diagonal_chi2": chi2,
        "dof": dof,
        "diagonal_chi2_per_dof": chi2 / dof,
    }


def full_covariance_chi2(
    separation: FloatArray,
    values: FloatArray,
    covariance_factor: FloatArray,
    amplitude: float,
    mass: float,
    ns: int,
) -> float:
    """Return ``r.T @ Cov^-1 @ r`` without explicitly inverting Cov."""
    residual = values - finite_volume_k0(
        separation, amplitude, mass, ns
    )
    whitened = linalg.solve_triangular(
        covariance_factor, residual, lower=True
    )
    return float(np.dot(whitened, whitened))


def k0_r_max_values(ns: int, scan_to_ns_third: bool) -> list[int]:
    """Return the campaign-specific periodic-K0 endpoint grid."""
    maximum = ns // 2
    if maximum < 8:
        raise ValueError("the K0 scan needs Ns/2 >= 8")
    if scan_to_ns_third:
        return list(range(maximum, math.ceil(ns / 3) - 1, -2))
    return [maximum - 2 * index for index in range(3)]


def _bootstrap_k0_window(
    separation: FloatArray,
    central_values: FloatArray,
    covariance_factor: FloatArray,
    errors: FloatArray,
    bootstrap_values: FloatArray,
    ns: int,
    central: dict[str, float],
) -> tuple[dict[str, float | bool], FloatArray, FloatArray]:
    """Bootstrap one diagonal fit and calibrate its full-covariance statistic.

    Each row of ``bootstrap_values`` is a jointly resampled vector over all
    fitted separations. The recentered samples impose the fitted K0 mean while
    retaining those joint fluctuations, which gives the null distribution of
    the requested post-fit full-covariance statistic.
    """
    initial = np.asarray([central["amplitude"], central["mass"]])
    central_model = finite_volume_k0(
        separation, central["amplitude"], central["mass"], ns
    )
    observed_chi2 = full_covariance_chi2(
        separation,
        central_values,
        covariance_factor,
        central["amplitude"],
        central["mass"],
        ns,
    )
    n_samples = len(bootstrap_values)
    mass_array = np.full(n_samples, np.nan, dtype=np.float64)
    xi_array = np.full(n_samples, np.nan, dtype=np.float64)
    null_chi2_samples: list[float] = []
    for sample_index, sample in enumerate(bootstrap_values):
        try:
            fit = _uncorrelated_k0_fit_with_errors(
                separation, sample, errors, ns, initial
            )
        except (ValueError, RuntimeError):
            pass
        else:
            mass_array[sample_index] = float(fit["mass"])
            xi_array[sample_index] = 1.0 / float(fit["mass"])

        null_values = central_model + sample - central_values
        try:
            null_fit = _uncorrelated_k0_fit_with_errors(
                separation, null_values, errors, ns, initial
            )
            null_chi2_samples.append(
                full_covariance_chi2(
                    separation,
                    null_values,
                    covariance_factor,
                    float(null_fit["amplitude"]),
                    float(null_fit["mass"]),
                    ns,
                )
            )
        except (ValueError, RuntimeError, linalg.LinAlgError):
            continue

    null_chi2_array = np.asarray(null_chi2_samples, dtype=np.float64)
    valid_mass = mass_array[np.isfinite(mass_array)]
    valid_xi = xi_array[np.isfinite(xi_array)]
    fit_success = len(valid_mass) / n_samples
    gof_success = len(null_chi2_array) / n_samples
    mass_error = (
        float(np.std(valid_mass, ddof=1))
        if len(valid_mass) >= 2
        else math.inf
    )
    xi_error = (
        float(np.std(valid_xi, ddof=1))
        if len(valid_xi) >= 2
        else math.inf
    )
    bootstrap_p_value = (
        (1.0 + float(np.count_nonzero(null_chi2_array >= observed_chi2)))
        / (1.0 + len(null_chi2_array))
        if len(null_chi2_array) >= 2
        else 0.0
    )
    relative_error = mass_error / central["mass"]
    quality_pass = bool(
        central["mass"] > 0.0
        and fit_success >= 0.90
        and gof_success >= 0.90
        and relative_error < 0.25
        and bootstrap_p_value >= 0.05
    )
    statistics: dict[str, float | bool] = {
        "mass_bootstrap_error": mass_error,
        "xi_exp": 1.0 / central["mass"],
        "xi_exp_bootstrap_error": xi_error,
        "relative_error": relative_error,
        "bootstrap_success_fraction": fit_success,
        "gof_bootstrap_success_fraction": gof_success,
        "bootstrap_gof_p_value": bootstrap_p_value,
        "quality_pass": quality_pass,
    }
    return statistics, xi_array, mass_array


def scan_k0_fits(
    separations: IntArray,
    correlator: FloatArray,
    covariance: FloatArray,
    bootstrap_correlators: FloatArray,
    ns: int,
    scan_r_max_to_ns_third: bool = False,
) -> K0ScanResult:
    """Scan joint-bootstrap K0 windows and select the earliest stable 3x3 patch.

    The fixed-beta/Nt default uses the three endpoint choices
    ``R_max = Ns/2, Ns/2-2, Ns/2-4``. The fixed-mass campaign instead scans
    that grid down to the last endpoint not below ``ceil(Ns/3)``. Every
    positive ``R_min`` leaving at least five points is tested. A stable patch
    contains three adjacent ``R_min`` values at three adjacent endpoints.
    Stability retains the pairwise two-combined-standard-deviation rule,
    applied to all nine joint-bootstrap mass estimates. The selected window
    is the earliest (smallest) ``R_min`` anchor, with the largest ``R_max``
    breaking ties.

    Range selection uses at most ``RANGE_SCAN_BOOTSTRAP_SAMPLES`` evenly
    spaced replicas from the reproducible joint bootstrap. The selected
    anchor is then refit with every replica for its final uncertainty and
    recentered-bootstrap goodness-of-fit p-value.
    """
    if len(bootstrap_correlators) < 2:
        raise ValueError("the K0 scan needs at least two bootstrap samples")
    maximum_r_max = int(separations[-1])
    if maximum_r_max != ns // 2:
        raise ValueError("the K0 scan requires separations through floor(Ns/2)")
    r_max_values = k0_r_max_values(ns, scan_r_max_to_ns_third)
    n_range_samples = min(
        len(bootstrap_correlators), RANGE_SCAN_BOOTSTRAP_SAMPLES
    )
    range_indices = np.linspace(
        0, len(bootstrap_correlators) - 1, n_range_samples, dtype=np.int64
    )
    range_bootstrap = bootstrap_correlators[range_indices]
    entries: list[dict[str, object]] = []
    entry_by_window: dict[tuple[int, int], dict[str, object]] = {}
    for r_max in r_max_values:
        for r_min in range(1, r_max - 3):
            selected = (separations >= r_min) & (separations <= r_max)
            selected_separations = separations[selected].astype(np.float64)
            selected_covariance = covariance[np.ix_(selected, selected)]
            entry: dict[str, object] = {
                "r_min": r_min,
                "r_max": r_max,
                "n_points": int(np.count_nonzero(selected)),
                "parameter_fit_covariance": "diagonal",
                "goodness_of_fit_covariance": "complete",
                "range_scan_bootstrap_samples": n_range_samples,
                "stable_three_by_three_patch": False,
                "stable_patch_anchor": False,
            }
            entry_by_window[(r_min, r_max)] = entry
            entries.append(entry)
            try:
                scale = linalg.cholesky(selected_covariance, lower=True)
                errors = np.sqrt(np.diag(selected_covariance))
                central = _uncorrelated_k0_fit_with_errors(
                    selected_separations,
                    correlator[selected],
                    errors,
                    ns,
                )
                complete_chi2 = full_covariance_chi2(
                    selected_separations,
                    correlator[selected],
                    scale,
                    central["amplitude"],
                    central["mass"],
                    ns,
                )
            except (ValueError, RuntimeError, linalg.LinAlgError) as error:
                entry["error"] = str(error)
                entry["quality_pass"] = False
                continue
            statistics, _, _ = _bootstrap_k0_window(
                selected_separations,
                correlator[selected],
                scale,
                errors,
                range_bootstrap[:, selected],
                ns,
                central,
            )
            entry.update(
                {
                    **central,
                    "full_covariance_chi2": complete_chi2,
                    "full_covariance_chi2_per_dof": (
                        complete_chi2 / central["dof"]
                    ),
                    "full_covariance_condition_number": float(
                        np.linalg.cond(selected_covariance)
                    ),
                    **statistics,
                }
            )

    stable_anchors: list[tuple[int, int]] = []
    for r_max_index in range(len(r_max_values) - 2):
        patch_r_max = r_max_values[r_max_index : r_max_index + 3]
        smallest_r_max = patch_r_max[-1]
        for r_min in range(1, smallest_r_max - 5):
            windows = [
                entry_by_window[(patch_r_min, patch_max)]
                for patch_max in patch_r_max
                for patch_r_min in range(r_min, r_min + 3)
            ]
            if not all(
                bool(entry.get("quality_pass", False)) for entry in windows
            ):
                continue
            stable = True
            for left_index, left in enumerate(windows):
                for right in windows[left_index + 1 :]:
                    difference = abs(float(left["mass"]) - float(right["mass"]))
                    combined = math.hypot(
                        float(left["mass_bootstrap_error"]),
                        float(right["mass_bootstrap_error"]),
                    )
                    if difference > 2.0 * combined:
                        stable = False
                        break
                if not stable:
                    break
            if not stable:
                continue
            for entry in windows:
                entry["stable_three_by_three_patch"] = True
            anchor = entry_by_window[(r_min, patch_r_max[0])]
            anchor["stable_patch_anchor"] = True
            anchor["stable_patch_r_min_values"] = list(range(r_min, r_min + 3))
            anchor["stable_patch_r_max_values"] = patch_r_max
            stable_anchors.append((r_min, patch_r_max[0]))

    for selected_window in sorted(
        stable_anchors, key=lambda window: (window[0], -window[1])
    ):
        selected_entry = entry_by_window[selected_window]
        selected = (
            (separations >= selected_window[0])
            & (separations <= selected_window[1])
        )
        selected_separations = separations[selected].astype(np.float64)
        selected_covariance = covariance[np.ix_(selected, selected)]
        scale = linalg.cholesky(selected_covariance, lower=True)
        errors = np.sqrt(np.diag(selected_covariance))
        central = {
            key: float(selected_entry[key])
            for key in ("amplitude", "mass")
        }
        range_statistics = {
            key: selected_entry[key]
            for key in (
                "mass_bootstrap_error",
                "xi_exp_bootstrap_error",
                "relative_error",
                "bootstrap_success_fraction",
                "gof_bootstrap_success_fraction",
                "bootstrap_gof_p_value",
                "quality_pass",
            )
        }
        final_statistics, xi_samples, mass_samples = _bootstrap_k0_window(
            selected_separations,
            correlator[selected],
            scale,
            errors,
            bootstrap_correlators[:, selected],
            ns,
            central,
        )
        selected_entry.update(
            {
                **{
                    f"range_scan_{key}": value
                    for key, value in range_statistics.items()
                },
                **final_statistics,
                "final_bootstrap_samples": len(bootstrap_correlators),
                "final_quality_pass": final_statistics["quality_pass"],
            }
        )
        if not bool(final_statistics["quality_pass"]):
            continue
        selected_entry["selected"] = True
        return K0ScanResult(
            float(selected_entry["xi_exp"]),
            float(selected_entry["xi_exp_bootstrap_error"]),
            float(selected_entry["mass"]),
            float(selected_entry["mass_bootstrap_error"]),
            xi_samples,
            mass_samples,
            entries,
        )

    empty = np.empty(0, dtype=np.float64)
    return K0ScanResult(None, None, None, None, empty, empty, entries)


def k0_fit_blocking_error_plateau(
    chains: Sequence[PostCutChain],
    ns: int,
    covariance: FloatArray,
    selected_fit: dict[str, object],
    primary_plateau: dict[str, object],
) -> dict[str, object]:
    """Check whole-history blocking plateaus for ``xi_exp`` and ``mass``."""
    separations = chains[0].raw.correlator.keys[:, 0].astype(np.int64)
    selected = (separations >= int(selected_fit["r_min"])) & (
        separations <= int(selected_fit["r_max"])
    )
    selected_covariance = covariance[np.ix_(selected, selected)]
    selected_errors = np.sqrt(np.diag(selected_covariance))
    initial = np.asarray(
        [selected_fit["amplitude"], selected_fit["mass"]], dtype=np.float64
    )

    def transform(mean: FloatArray) -> FloatArray:
        fit = _uncorrelated_k0_fit_with_errors(
            separations[selected].astype(np.float64),
            mean[selected],
            selected_errors,
            ns,
            initial,
        )
        mass = float(fit["mass"])
        return np.asarray([1.0 / mass, mass], dtype=np.float64)

    lengths = [
        int(length)
        for length in primary_plateau["candidate_block_lengths_sweeps"]
    ]
    estimates = np.empty((len(lengths), 2), dtype=np.float64)
    errors = np.empty_like(estimates)
    error_uncertainties = np.empty_like(estimates)
    effective_degrees_of_freedom = np.empty_like(estimates)
    blocks_per_chain: list[list[int]] = []
    discarded_sweeps_per_chain: list[list[int]] = []
    for length_index, length in enumerate(lengths):
        blocks = [block_values(chain.correlator, length) for chain in chains]
        estimate, error, degrees_of_freedom = fused_replica_jackknife(
            blocks, transform
        )
        estimates[length_index] = estimate
        errors[length_index] = error
        effective_degrees_of_freedom[length_index] = degrees_of_freedom
        error_uncertainties[length_index] = error / np.sqrt(
            2.0 * degrees_of_freedom
        )
        blocks_per_chain.append([len(chain_blocks) for chain_blocks in blocks])
        discarded_sweeps_per_chain.append(
            [len(chain.correlator) % length for chain in chains]
        )

    groups_by_observable, common_groups = stable_three_length_groups(
        lengths,
        errors,
        error_uncertainties,
        int(primary_plateau["minimum_tau_block_length_sweeps"]),
    )
    selected_block_length = chains[0].block_length_sweeps
    selected_group = next(
        (group for group in common_groups if selected_block_length in group),
        None,
    )
    labels = ("xi_exp", "mass")
    observables = {
        label: {
            "estimate": _strict_float_list(estimates[:, index]),
            "error": _strict_float_list(errors[:, index]),
            "error_uncertainty": _strict_float_list(
                error_uncertainties[:, index]
            ),
            "effective_degrees_of_freedom": _strict_float_list(
                effective_degrees_of_freedom[:, index]
            ),
            "plateau_group_sweeps": (
                groups_by_observable[index][0]
                if groups_by_observable[index]
                else None
            ),
            "all_plateau_groups_sweeps": groups_by_observable[index],
        }
        for index, label in enumerate(labels)
    }
    return {
        "method": (
            "whole post-cut correlator histories; separate complete blocks "
            "per replica; fixed selected-fit marginal errors and range; "
            "replica-stratified delete-one-block refits"
        ),
        "fit_range": {
            "r_min": int(selected_fit["r_min"]),
            "r_max": int(selected_fit["r_max"]),
        },
        "candidate_block_lengths_sweeps": lengths,
        "minimum_tau_block_length_sweeps": int(
            primary_plateau["minimum_tau_block_length_sweeps"]
        ),
        "complete_blocks_per_replica": blocks_per_chain,
        "discarded_sweeps_per_replica": discarded_sweeps_per_chain,
        "observables": observables,
        "all_common_plateau_groups_sweeps": common_groups,
        "selected_common_plateau_group_sweeps": selected_group,
        "selected_block_length_sweeps": selected_block_length,
        "selected_block_on_xi_exp_and_mass_plateau": selected_group is not None,
    }


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
    campaign_type: str | None = None,
    target_c: float | None = None,
) -> dict[str, object]:
    """Produce the full result for one four-chain ensemble."""
    if bootstrap_samples not in ANALYSIS_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "analysis bootstrap samples must be exactly 500"
        )
    plateau = diagnostics["blocking_error_plateau"]
    if not bool(plateau["all_primary_and_secondary_plateaus_found"]):
        raise ValueError(
            "full analysis requires blocking-error plateaus for every "
            "primary and secondary observable"
        )
    separations = chains[0].raw.correlator.keys[:, 0].astype(np.int64)
    if not np.array_equal(separations, np.arange(ns // 2 + 1)):
        raise ValueError("correlator grid must be R=0,...,floor(Ns/2)")

    primary_size = chains[0].primary_blocks.shape[1]
    profile_size = chains[0].profile_blocks.shape[1]
    boundary_size = (
        chains[0].boundary_blocks.shape[1]
        if chains[0].boundary_blocks is not None
        else 0
    )
    joint_blocks = []
    for chain in chains:
        parts = [chain.primary_blocks, chain.profile_blocks]
        if chain.boundary_blocks is not None:
            parts.append(chain.boundary_blocks)
        if len({len(part) for part in parts}) != 1:
            raise ValueError(f"{chain.raw.directory}: joint block grids differ")
        joint_blocks.append(np.concatenate(parts, axis=1))
    joint_mean, joint_samples, _ = chainwise_bootstrap(
        joint_blocks,
        lambda mean: mean,
        bootstrap_samples,
        np.random.default_rng(bootstrap_seed),
    )
    mean_vector = joint_mean[:primary_size]
    mean_samples = joint_samples[:, :primary_size]
    profile_start = primary_size
    profile_stop = profile_start + profile_size
    profile_mean = joint_mean[profile_start:profile_stop]
    profile_samples = joint_samples[:, profile_start:profile_stop]
    profile_covariance = np.atleast_2d(
        np.cov(profile_samples, rowvar=False, ddof=1)
    )
    boundary_mean = (
        joint_mean[profile_stop : profile_stop + boundary_size]
        if boundary_size > 0
        else None
    )
    boundary_samples = (
        joint_samples[:, profile_stop : profile_stop + boundary_size]
        if boundary_size > 0
        else None
    )
    boundary_covariance = (
        np.atleast_2d(np.cov(boundary_samples, rowvar=False, ddof=1))
        if boundary_samples is not None
        else None
    )
    correlator = mean_vector[4:]
    bootstrap_correlators = mean_samples[:, 4:]
    correlator_covariance = np.atleast_2d(
        np.cov(bootstrap_correlators, rowvar=False, ddof=1)
    )
    correlator_error = np.sqrt(np.maximum(np.diag(correlator_covariance), 0.0))
    if correlator[0] != 1.0 or correlator_error[0] > 1.0e-14:
        raise ValueError("bootstrap did not preserve exact C_G(0)=1")

    xi_squared_samples = second_moment_squared_samples(mean_samples, ns)
    xi_samples = second_moment_samples(mean_samples, ns)
    xi_valid = np.isfinite(xi_samples)
    xi_valid_fraction = float(np.mean(xi_valid))
    xi = math.sqrt(mean_vector[2] / mean_vector[3] - 1.0) / (
        2.0 * math.sin(math.pi / ns)
    )
    xi_squared_interval = np.quantile(xi_squared_samples, [0.16, 0.84])
    if xi_squared_interval[0] <= 0.0:
        raise ValueError("xi_2nd is unresolved at 68 percent confidence")
    xi_interval = np.sqrt(xi_squared_interval)
    ratio_samples = ns / xi_samples
    ratio = ns / xi
    ratio_interval = np.asarray(
        [ns / xi_interval[1], ns / xi_interval[0]], dtype=np.float64
    )
    xi_error = (
        float(np.std(xi_samples, ddof=1))
        if np.all(xi_valid)
        else None
    )
    ratio_error = (
        float(np.std(ratio_samples, ddof=1))
        if np.all(xi_valid)
        else None
    )

    fit_result = scan_k0_fits(
        separations,
        correlator,
        correlator_covariance,
        bootstrap_correlators,
        ns,
        scan_r_max_to_ns_third=(campaign_type == "nonperturbative_mg_nt"),
    )
    xi_exp = fit_result.xi_exp
    xi_exp_error = fit_result.xi_exp_error
    mass = fit_result.mass
    mass_error = fit_result.mass_error
    xi_exp_bootstrap = fit_result.xi_exp_bootstrap
    mass_bootstrap = fit_result.mass_bootstrap
    fit_scan = fit_result.entries
    fit_blocking_plateau: dict[str, object] | None = None
    if xi_exp is not None:
        selected_fit = next(
            entry for entry in fit_scan if bool(entry.get("selected", False))
        )
        try:
            fit_blocking_plateau = k0_fit_blocking_error_plateau(
                chains,
                ns,
                correlator_covariance,
                selected_fit,
                plateau,
            )
        except (ValueError, RuntimeError, linalg.LinAlgError) as error:
            fit_blocking_plateau = {
                "error": str(error),
                "selected_block_on_xi_exp_and_mass_plateau": False,
            }
        blocking_pass = bool(
            fit_blocking_plateau[
                "selected_block_on_xi_exp_and_mass_plateau"
            ]
        )
        selected_fit["blocking_error_plateau_pass"] = blocking_pass
        if not blocking_pass:
            xi_exp = None
            xi_exp_error = None
            mass = None
            mass_error = None
            xi_exp_bootstrap = np.empty(0, dtype=np.float64)
            mass_bootstrap = np.empty(0, dtype=np.float64)

    covariance_arrays = {
        "R": separations,
        "C_G": correlator,
        "C_G_covariance": correlator_covariance,
        "C_G_bootstrap": bootstrap_correlators,
        "profile_kind_t": chains[0].raw.profile.keys,
        "profile_mean": profile_mean,
        "profile_covariance": profile_covariance,
        "xi_2nd_bootstrap": xi_samples,
        "xi_2nd_squared_bootstrap": xi_squared_samples,
        "Ns_over_xi_2nd_bootstrap": ratio_samples,
        "xi_exp_bootstrap": xi_exp_bootstrap,
        "mass_bootstrap": mass_bootstrap,
    }
    boundary_result: dict[str, object] | None = None
    if (
        boundary_mean is not None
        and boundary_samples is not None
        and boundary_covariance is not None
        and chains[0].raw.boundary is not None
    ):
        boundary_error = np.sqrt(
            np.maximum(np.diag(boundary_covariance), 0.0)
        )
        boundary_result = {
            "R_H": chains[0].raw.boundary.keys.tolist(),
            "mean": _strict_float_list(boundary_mean),
            "error": _strict_float_list(boundary_error),
            "measurement_interval_sweeps": regular_measurement_cadence(
                chains[0].raw.boundary
            ),
        }
        covariance_arrays.update(
            {
                "boundary_wilson_R_H": chains[0].raw.boundary.keys,
                "boundary_wilson_mean": boundary_mean,
                "boundary_wilson_covariance": boundary_covariance,
                "boundary_wilson_bootstrap": boundary_samples,
            }
        )
    covariance_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(covariance_output, **covariance_arrays)
    result = {
        "campaign_type": campaign_type,
        "target_c": target_c,
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
        "boundary_wilson_measurement_interval_sweeps": (
            regular_measurement_cadence(chains[0].raw.boundary)
            if chains[0].raw.boundary is not None
            else None
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resampling": (
            "fixed-chain complete-vector block bootstrap; equal chain weights; "
            "independent within-chain draws shared by every correlator "
            "separation, profile coordinate, and boundary-loop geometry"
        ),
        "thermalization_and_blocking": diagnostics,
        "R": separations.tolist(),
        "C_G": _strict_float_list(correlator),
        "C_G_error": _strict_float_list(correlator_error),
        "mean_Tr_G_real": float(mean_vector[0]),
        "mean_Tr_G_real_error": float(np.std(mean_samples[:, 0], ddof=1)),
        "Ctilde_0": float(mean_vector[2]),
        "Ctilde_pmin": float(mean_vector[3]),
        "xi_2nd": xi,
        "xi_2nd_error": xi_error,
        "xi_2nd_bootstrap_valid_fraction": xi_valid_fraction,
        "xi_2nd_squared_68_percent_interval": _strict_float_list(
            xi_squared_interval
        ),
        "xi_2nd_68_percent_interval": _strict_float_list(xi_interval),
        "Ns_over_xi_2nd": ratio,
        "Ns_over_xi_2nd_error": ratio_error,
        "Ns_over_xi_2nd_68_percent_interval": _strict_float_list(
            ratio_interval
        ),
        "xi_exp": xi_exp,
        "xi_exp_error": xi_exp_error,
        "mass": mass,
        "mass_error": mass_error,
        "K0_fit_method": {
            "model": "A * [K0(m R) + K0(m (Ns - R))]",
            "parameter_estimation": "diagonal covariance",
            "parameter_uncertainty": "joint fixed-chain block bootstrap fits",
            "goodness_of_fit": "complete covariance chi2 per dof",
            "goodness_of_fit_p_value": "recentered joint bootstrap",
            "r_max_scan": (
                "Ns/2, Ns/2-2, ..., last endpoint >= ceil(Ns/3)"
                if campaign_type == "nonperturbative_mg_nt"
                else "Ns/2, Ns/2-2, Ns/2-4"
            ),
        },
        "K0_fit_scan": fit_scan,
        "xi_exp_blocking_error_plateau": fit_blocking_plateau,
        "plaquette_profile": {
            "kind_t": chains[0].raw.profile.keys.tolist(),
            "mean": _strict_float_list(profile_mean),
            "error": _strict_float_list(
                np.sqrt(np.maximum(np.diag(profile_covariance), 0.0))
            ),
        },
        "boundary_wilson": boundary_result,
    }
    return result


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
    parser.add_argument("--campaign-type")
    parser.add_argument("--target-c", type=float)
    parser.add_argument("--minimum-cut", type=int, default=500)
    parser.add_argument("--maximum-sweep", type=int)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        choices=ANALYSIS_BOOTSTRAP_SAMPLES,
        default=ANALYSIS_BOOTSTRAP_SAMPLES[0],
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_721)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--thermalization-diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--covariance-output", type=Path)
    args = parser.parse_args()
    if args.nt < 1 or args.ns < 2 or args.beta <= 0.0 or args.minimum_cut < 0:
        raise ValueError("invalid ensemble parameters")
    if args.maximum_sweep is not None and args.maximum_sweep <= args.minimum_cut:
        raise ValueError("maximum sweep must exceed the minimum cut")

    raw_chains = load_ensemble(
        args.ensemble_dir, args.chain_directory, args.maximum_sweep
    )
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
            args.campaign_type,
            args.target_c,
        )
    if args.maximum_sweep is not None:
        result["analysis_status"] = "preliminary_common_sweep_snapshot"
        result["maximum_sweep_inclusive"] = args.maximum_sweep
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
