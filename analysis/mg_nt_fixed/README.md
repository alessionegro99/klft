# Nonperturbative fixed-`m_G Nt` compact results

This directory contains only compact JSON, CSV, NPZ, metadata, and single-page
16:9 PDF products synchronized from qbig. It intentionally contains no raw
Monte Carlo histories. All 25 pilot analyses are synchronized under `pilot/`;
each has an accepted mass, accepted mass/`xi_exp` plateau, at least 30 blocks
per chain, and 1000 finite mass replicas.

The official initial tuning under `tuning/` passes seven of nine targets. The
two failed `c=0.5` targets triggered the separate `extensions/round1/` points.
Both round-1 extension analyses are synchronized and accepted. The round-2
point under `extensions/round2/` is also accepted, with
`m_G Nt=0.5051264 +/- 0.0020453` and fit `p=0.8891`. Round-2 tuning under
`tuning_round2/` passes all nine targets. Its `c=0.5,Nt=3` result is
`beta_star=4.29462 [4.29213,4.29828]` with fit `p=0.1474`.

The final checks under `final_checks/` replace beta 4.40 with four fresh
80000-sweep chains, replace the two low-volume `c=1,Nt=4` inputs by `Ns=96`
measurements, and repeat all three `c=2,Nt=4` analyses with 10000 replicas.
Final tuning under `tuning_final/` passes all nine groups. The independent
confirmation at `Nt=4,c=1,beta=4.61755576` gives
`m_G Nt=1.006200 +/- 0.006542` and passes its 3% agreement gate.

Required statistical increases are tracked in
the [canonical checklist](../../docs/mg_nt_fixed/summary.md#preliminary-statistics-checklist).

[`preliminary_plots/`](preliminary_plots/) contains the historical 84-plot
round-2 diagnostic set. [`final_plots/`](final_plots/) contains seven focused,
grid-free, single-page 16:9 PDFs from the checked final tuning:

- three final mass-bracket plots;
- tuned `beta_star/Nt` versus `1/Nt`;
- two direct periodic-correlator volume comparisons;
- one extracted-mass volume-stability comparison.

Reproduce the focused set from the repository root with:

```bash
MPLCONFIGDIR=/tmp/dirichlet-mpl \
UV_CACHE_DIR=/tmp/dirichlet-uv-cache uv run python \
  scripts/mg_nt_fixed_volume_plots.py \
  --pilot-results analysis/mg_nt_fixed/pilot \
  --replacement-results analysis/mg_nt_fixed/final_checks \
  --tuning-summary analysis/mg_nt_fixed/tuning_final/tuning_summary.json \
  --output analysis/mg_nt_fixed/final_plots
```

The focused plotting-program SHA-256 is
`cd12a7988b36b2400cc9d743b11bb7893b4e6056a95484dfb6b64fdf19889f8c`.

The historical [`preliminary_plots/`](preliminary_plots/) set contains 84
single-page 16:9 LaTeX PDFs made from all currently available compact pilot,
extension, and audit results:

- three tuning-bracket and two tuned-trajectory plots;
- 29 accepted periodic-`K0` mass-fit and 29 plaquette-profile plots;
- nine normalized-correlator bracket comparisons;
- three plots each for `xi_exp/xi_2nd`, `m_G Ns`, `Ns/xi_2nd`, and the real
  traced-holonomy symmetry diagnostic.

The two fixed-`beta/Nt` audit inputs are explicitly labelled as prior
information in every plot that contains or derives from them. Reproduce the
set from the repository root with:

```bash
MPLCONFIGDIR=/tmp/dirichlet-mpl \
UV_CACHE_DIR=/tmp/dirichlet-uv-cache uv run python \
  scripts/mg_nt_fixed_plot.py \
  --summary analysis/mg_nt_fixed/tuning_round2/tuning_summary.json \
  --analysis-root analysis/mg_nt_fixed \
  --output analysis/mg_nt_fixed/preliminary_plots \
  --preliminary
```

The plotting-program SHA-256 is
`f30d0d957e4d133ddfff35e6484cba648e9a4b19712f92be099f427bffc6e449`.
All 84 PDFs were checked to have one `1152 x 648` point page. Wilson-loop
plots do not exist for this pilot because Wilson measurements were disabled;
no bulk scalar-mass plot is made because no validated bulk scalar extraction
exists.

Prior fixed-`beta/Nt` points may appear only below `audit/`, with their source
campaign recorded explicitly. They are not campaign measurements and must not
be aggregated without that label.

See [`docs/mg_nt_fixed/summary.md`](../../docs/mg_nt_fixed/summary.md) for the
canonical status.
