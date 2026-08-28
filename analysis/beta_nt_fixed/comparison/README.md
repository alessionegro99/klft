# Fixed-beta/Nt comparison

This directory contains the current cross-line comparison. It is preliminary:
the `beta/Nt=2.1` line still has only two accepted continuum spacings, and
several accepted points need more independent Monte Carlo statistics or
volume checks.

Only the continuum ansatz

\[
m_GN_t=(m_GL)_{\rm cont}+c_1/N_t
\]

is retained. There are no `p=2` or free-exponent fits in the JSON, NPZ, or
plots. Every active ensemble analysis and every continuum bootstrap operation
uses exactly 1,000 replicas.

## Current mass result

| `beta/Nt` | accepted continuum `Nt` | `(m_G L)_cont`, `p=1` | fit p-value |
|---:|---|---:|---:|
| 1.8 | 2, 3, 4, 5, 6 | 0.118373(884) | 0.1328 |
| 2.0 | 2, 3, 4, 5 | 0.073017(764) | 0.8508 |
| 2.1 | 2, 3 | not fitted | -- |

The exact `Nt=1` PCM results at beta 2.0 and 2.1 are plotted as references but
are not included in the continuum fits.

The completed 48,300-sweep analysis of `(Nt,Ns,beta)=(5,688,10.5)` requested
1,000 replicas and produced 1,000 finite `xi_2nd` replicas. It found eight
individual mass windows passing the pointwise quality gates, but no stable
three-by-three asymptotic patch. Consequently `mass` and `xi_exp` remain null,
and this point is excluded. The `Nt=4` point on the same line remains excluded
because it has no common three-length blocking plateau.

## Compact products and reproduction

- [`selection.tsv`](selection.tsv): exact line-centric selection.
- [`continuum_summary.json`](continuum_summary.json): points, exclusions, and
  only the `p=1` fits.
- [`continuum_bootstrap.npz`](continuum_bootstrap.npz): exactly 1,000 fitted
  parameter replicas for each available line and observable.
- [`plots/`](plots/): 64 grid-free, single-page 16:9 PDFs using the current
  accepted data. All supported correlator, mass-fit, profile, boundary-loop,
  and comparison plots are retained; every continuum curve is `p=1`.
- [`checklist.md`](checklist.md): limitations that prevent a final result.

From the repository root:

```bash
uv run python scripts/beta_nt_fixed/continuum_summary.py \
  --selection analysis/beta_nt_fixed/comparison/selection.tsv \
  --analysis-root analysis/beta_nt_fixed \
  --output analysis/beta_nt_fixed/comparison/continuum_summary.json \
  --covariance-output analysis/beta_nt_fixed/comparison/continuum_bootstrap.npz \
  --allow-incomplete

uv run python scripts/beta_nt_fixed/plot_continuum.py \
  --summary analysis/beta_nt_fixed/comparison/continuum_summary.json \
  --output analysis/beta_nt_fixed/comparison/plots \
  --preliminary
```

The summary and plotting script SHA-256 values are respectively
`249f06aeff239d3769257d8504e9de10dae977757bb966d84c6646a1b439561e`
and `6ef42816494a4482fae112b9da2ae5260b939eddc20e0e3087a7b660df3a9b2c`.

## First-stage diagnostics

Every one of the 13 slab ensembles and both `Nt=1` PCM references has:

- `first_stage/thermalization.pdf`: all four independent chains, shown as
  100-sweep means of `Ctilde_0`, with each chain's selected cut;
- `first_stage/blocking_error.pdf`: the chain-aware blocking error of
  `Ctilde_0`, with the autocorrelation bound and accepted common plateau when
  available;
- `first_stage/metadata.json`: raw qbig paths, cutoff, method, and plot list.

These are 30 grid-free, one-page 16:9 PDFs. The two older PCM references
predate the full whole-history plateau schema, so their blocking curves use
the same replica-stratified delete-one-block jackknife implementation but do
not claim a new all-observable plateau. No bootstrap operation is used to make
these diagnostic plots.

qbig job `282513` generated the diagnostics from remote raw histories and
completed with exit code zero. The plotting script SHA-256 is
`816875b798661b55111a5d87850225f615a309c9e00423413e8bec3866605fc6`.
