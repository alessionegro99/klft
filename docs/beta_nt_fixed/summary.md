# Fixed-beta/Nt campaign summary

## Definition and conventions

This campaign takes the continuum limit at fixed

\[
g_3^2L, \qquad L=aN_t.
\]

For three-dimensional SU(2), with \(\beta=4/(ag_3^2)\),

\[
g_3^2L=\frac{4N_t}{\beta}.
\]

Thus each continuum trajectory has fixed \(\beta/N_t\), while
\(N_t,\beta\to\infty\). Here `Nt` is the physical number of temporal links
between the Dirichlet surfaces; KLFT stores `L2=Nt+1`. These conventions follow
the Wilson-action matching and geometry in Eqs. (A.10) and (2.1) of the
[Nebbia thesis](../marconebbia_tesi.pdf).

The measured holonomy correlator is
\(C_G(R)=\langle\operatorname{Tr}[G(x)G^\dagger(x+R)]\rangle\). A periodic
\(K_0\) fit gives \(am_G=1/(\xi_{\exp}/a)\), and the dimensionless continuum
observable is

\[
m_GL=(am_G)N_t.
\]

The mass relation and periodic fit are Eqs. (2.12) and (4.14) of the same
thesis.

## Current simulation and analysis status

The production grid contains the exact lines

\[
\frac{\beta}{N_t}=1.8,\ 2.0,\ 2.1,
\qquad
g_3^2L=\frac{20}{9},\ 2,\ \frac{40}{21}.
\]

Holonomy observables were measured every sweep, plaquette profiles every five
sweeps, and all-height boundary Wilson loops every 25 sweeps. The P100 build
used KLFT commit `f3928aa`, executable SHA-256
`a22087d10bdc78a9fae501cc0e4e0e91116577ce2bd684345331c4113e038ef8`.
Raw histories remain on qbig.

Production array `282274_[0-39]` completed 36 of 40 chains. The four
`(Nt,Ns,beta)=(5,688,10.5)` tasks timed out after 48,324--48,778 sweeps. No GPU
simulation is still running.

Every active ensemble analysis and every continuum bootstrap operation uses
exactly 1,000 replicas. Independent chains are cut, blocked, and resampled
separately; complete vectors of separations are resampled jointly. The older
10,000-replica confirmation products are excluded from the cleaned selection.

Accepted slab masses are:

| `beta/Nt` | `Nt` | `Ns` | `m_G Nt` | mass-fit p-value | minimum blocks/chain |
|---:|---:|---:|---:|---:|---:|
| 1.8 | 2 | 128 | 0.207356(773) | 0.5065 | 158 |
| 1.8 | 3 | 176 | 0.179114(571) | 0.0639 | 121 |
| 1.8 | 4 | 256 | 0.162359(916) | 0.0569 | 60 |
| 1.8 | 5 | 336 | 0.154523(647) | 0.0599 | 59 |
| 1.8 | 6 | 416 | 0.147475(814) | 0.0569 | 79 |
| 2.0 | 2 | 192 | 0.116564(258) | 0.1169 | 78 |
| 2.0 | 3 | 304 | 0.102186(401) | 0.0909 | 79 |
| 2.0 | 4 | 416 | 0.094690(423) | 0.2388 | 53 |
| 2.0 | 5 | 544 | 0.09266(567) | 0.8042 | 38 |
| 2.1 | 2 | 240 | 0.087457(216) | 0.0599 | 164 |
| 2.1 | 3 | 384 | 0.077537(404) | 0.0699 | 60 |

The `beta/Nt=2.1,Nt=4` ensemble has no common three-length plateau across all
924 primary and secondary observables and is rejected. Its maximum
between-chain discrepancy is 4.619 standard deviations.

For `beta/Nt=2.1,Nt=5`, the corrected common 48,300-sweep diagnostic accepts
the `[1000,1200,1400]`-sweep plateau and leaves 32--33 blocks per chain. qbig
job `282512` completed the requested 1,000-replica analysis, but none of the
eight pointwise acceptable mass windows belongs to a stable three-by-three
asymptotic patch. Consequently `mass` and `xi_exp` are null and the point is
not fitted.

## Continuum mass

Only the requested linear correction is retained:

\[
m_GN_t=(m_GL)_{\rm cont}+\frac{c_1}{N_t}.
\]

| `beta/Nt` | accepted continuum `Nt` | `(m_G L)_cont` | fit p-value |
|---:|---|---:|---:|
| 1.8 | 2, 3, 4, 5, 6 | 0.118373 +/- 0.000884 | 0.1328 |
| 2.0 | 2, 3, 4, 5 | 0.073017 +/- 0.000764 | 0.8508 |
| 2.1 | 2, 3 | not fitted | -- |

These are preliminary results. In particular, `beta/Nt=2.0,Nt=5` has only
38--40 blocks per chain and a 6.1% relative mass error. Direct volume checks
are also required: the accepted `(beta/Nt,Nt)=(1.8,6)` and `(2.1,3)` points
have `Ns/xi_2nd=9.787` and `9.918`, below the scale near 10 that previously
failed a full-correlator volume test; `(2.0,5)` is only `10.007`.

The exact `Nt=1` PCM results, `m_G=0.126859(380)` at beta 2.0 and
`0.096229(318)` at beta 2.1, are reference points and are not included in the
continuum fits.

## Plots and first-stage diagnostics

[`analysis/beta_nt_fixed/comparison/`](../../analysis/beta_nt_fixed/comparison/)
contains the exact selection, the 1,000-replica NPZ, the `p=1`-only JSON, and
64 grid-free one-page 16:9 PDFs. The plots cover all currently supported mass,
correlator, profile, boundary-loop, residual, and cross-line observables; no
background grids are used.

Each of the 13 slab ensembles and both PCM references also has two first-stage
PDFs inside its line/ensemble directory:

- `thermalization.pdf`: four independent 100-sweep-binned `Ctilde_0`
  histories and their selected cuts;
- `blocking_error.pdf`: the chain-aware `Ctilde_0` blocking error,
  autocorrelation bound, and common plateau where one exists.

qbig job `282513` generated all 30 PDFs from the remote histories and completed
with exit code zero. The older PCM schema lacks the all-observable plateau
record; its displayed curve is a chain-stratified jackknife diagnostic and is
not promoted to a new acceptance gate. The detailed remaining-statistics list
is [`comparison/checklist.md`](../../analysis/beta_nt_fixed/comparison/checklist.md).

## Clean repository layout and provenance

- `data/beta_nt_fixed/beta_over_nt_1p8/`, `beta_over_nt_2p0/`, and
  `beta_over_nt_2p1/`: active immutable inputs grouped by line;
- `analysis/beta_nt_fixed/beta_over_nt_1p8/`, `beta_over_nt_2p0/`, and
  `beta_over_nt_2p1/`: compact per-ensemble results and first-stage plots;
- `analysis/beta_nt_fixed/comparison/`: only `p=1` continuum products and all
  comparison plots;
- `data/beta_nt_fixed/comparison/provenance/` and
  `analysis/beta_nt_fixed/comparison/provenance/`: historical workflows and
  explicitly superseded products;
- `scripts/beta_nt_fixed/`: analysis, plotting, and tests.

Historical qbig paths and input header comments are unchanged so their hashes
remain valid. The fixed-`m_GN_t` campaign is separate and is never aggregated
here.
