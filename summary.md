# Numerical validation summary

## qbig environment

On 2026-07-17, Slurm reported `lnode13` through `lnode17` idle, each with eight Pascal GPUs. Direct inspection of `lnode13` identified eight idle Tesla P100-PCIE-16GB GPUs. This is an availability snapshot, not a permanent allocation.

The fresh qbig production build uses KLFT commit `2871b8abc9c28917c2049243d3483d6dada4276a`, CUDA 11.5.0, GCC 11.2.0, Kokkos CUDA/OpenMP, `Kokkos_ARCH_PASCAL60=ON`, `KLFT_NDIM=3`, and `KLFT_NC=2`. Its heatbath SHA-256 is `5e7ad09405829c2a7831b666de16bb0e195391c6b020b4df901134cd632cdaa4`.

## P100 smoke validation

Slurm job `259715` completed on `lnode13` with exit code 0. The deterministic suite passed. The small statistical suite reported hot/cold plaquette agreement, spatial and temporal reflection-profile differences of 0.67 and 2.04 standard deviations, the global-orientation group-averaging relation at 0.40 standard deviations, and long-slab central plaquette recovery at 0.20 standard deviations.

For eight hot-start `Ns=8`, physical `Nt=1` measurements at beta 12.15266, `C_G(0)` was exactly one configuration by configuration. The maximum absolute difference between `W_b(R,1)` and `C_G(R)` over the independent distances was exactly zero in the printed float64 output.

An `Ns=64`, physical `Nt=2` P100 timing run completed 100 heatbath sweeps, each followed by five overrelaxation sweeps and all selected measurements, in 89.42 seconds. Peak resident memory was 161912 KiB. This predicts about 1.99 hours for an 8000-sweep chain at the same thickness, before scheduler startup overhead.

The corresponding physical `Nt=12` timing run, measuring only the required boundary-loop heights `H=1` and `H=Nt`, took 106.01 seconds for 100 sweeps and 161840 KiB peak resident memory. This predicts about 2.36 hours for an 8000-sweep chain.

Twelve production chains at physical `Nt=2,6,12` completed with exit code 0. Each thickness has four independent hot starts and each chain has 8000 measurements. The `Nt=2`, `Nt=6`, and `Nt=12` task ranges were 2:06:30--2:08:29, 2:15:59--2:18:29, and 2:32:25--2:34:46, respectively. Every job's post-run checks confirmed exact `C_G(0)=1` and agreement of the full-height boundary loop with `C_G` within the printed-output tolerance.

MSER-5 thermalization diagnostics recommend retaining the preregistered 500-measurement cut for all three ensembles. Across chains, post-cut `tau_int[Tr G]` is 89--147 at `Nt=2`, 30--54 at `Nt=6`, and 8.6--10.7 at `Nt=12`; the invariant `C_G(1)` has `tau_int` below 0.76 throughout. All selected split-history z-scores are below 1.84 and the largest selected between-chain z-score is 3.18.

The full blocked analysis uses block lengths 294, 108, and 22 at Nt=2, 6, and 12, leaving 100, 276, and 1360 blocks. Its largest all-observable between-chain, split-history, and global-orientation-relation diagnostics are 3.68, 3.03, and 2.18 standard deviations, below the predeclared threshold of five. Plaquette-profile reflection discrepancies are at most 0.60, 1.02, and 1.16 standard deviations.

With the thesis normalization `(1/2)Tr G`, the hot-start traced holonomy is 0.0227(334), 0.0036(106), and 0.00406(153) at Nt=2, 6, and 12. Compared with thesis Table 4.4, the differences are -13.21, -2.86, and +1.18 combined standard deviations. The strong Nt=2 disagreement confirms the thesis's warning that its cold-start value was not thermalized on the 8000-sweep time scale.

The gauge-invariant boundary benchmark reproduces thesis Table 4.7: `W_b(64,1)=0.04138(27)` at Nt=6 and `0.03922(26)` at Nt=12, differing from the thesis by -0.65 and +0.98 combined standard deviations. The full-height boundary loop equals `C_G` configuration by configuration, while the height-one loop has the expected non-symmetric trend.

The second-moment lengths are 78.40(17), 33.07(13), and 10.27(11). At Nt=2 the length exceeds `Ns=64`, so this ensemble cannot yield a controlled mass gap. Every correlated periodic-K0 fit range tested is rejected; uncorrelated fits can look acceptable but are not used for inference because they discard strong cross-distance covariance.

Final analysis job `259734` and plotting jobs `259737` and `259738` completed with exit code 0. Compact JSON, covariance matrices, CSV, and PDFs are in `analysis/dirichlet_thesis/`; their SHA-256 hashes match the qbig copies. Raw histories remain only on qbig.
