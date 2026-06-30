#!/usr/bin/env python3
"""Report partition-update throughput and autocorrelation-adjusted ESS/s.

The automatic window follows the Gamma-method prescription discussed in
U. Wolff, Comput. Phys. Commun. 156 (2004) 143, hep-lat/0306017.
Benchmark inputs must measure the plaquette every sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def integrated_autocorrelation_time(values: NDArray[np.float64], c: float = 5.0) -> float:
    """Estimate tau_int with an FFT autocovariance and automatic window."""
    centered = np.asarray(values, dtype=np.float64) - np.mean(values)
    n = len(centered)
    if n < 2:
        raise ValueError("at least two measurements are required")
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:n]
    autocovariance /= np.arange(n, 0, -1)
    if autocovariance[0] <= 0.0:
        raise ValueError("plaquette variance is zero")
    rho = autocovariance / autocovariance[0]
    tau = 0.5
    for lag in range(1, n):
        tau += rho[lag]
        if lag >= c * max(tau, 0.5):
            break
    return max(tau, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plaquette_file", type=Path)
    parser.add_argument("--volume", type=int, required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--hits", type=int, required=True)
    parser.add_argument("--thermalization", type=int, default=0,
                        help="number of initial measured sweeps to discard")
    args = parser.parse_args()

    data = np.loadtxt(args.plaquette_file, comments="#", ndmin=2)
    data = data[args.thermalization:]
    if len(data) < 2:
        raise ValueError("too few post-thermalization measurements")
    plaquette = data[:, 1]
    update_time = data[:, -1]
    tau_int = integrated_autocorrelation_time(plaquette)
    effective_samples = len(plaquette) / (2.0 * tau_int)
    proposals_per_sweep = args.volume * args.dimensions * args.hits
    proposals_per_second = proposals_per_sweep / np.mean(update_time)
    effective_samples_per_second = effective_samples / np.sum(update_time)

    print(f"measurements {len(plaquette)}")
    print(f"tau_int {tau_int:.8g}")
    print(f"proposals_per_second {proposals_per_second:.8g}")
    print(f"effective_samples_per_second {effective_samples_per_second:.8g}")


if __name__ == "__main__":
    main()
