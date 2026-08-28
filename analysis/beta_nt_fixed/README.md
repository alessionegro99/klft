# Fixed-beta/Nt analysis

Active compact results are grouped by the exact value of `beta/Nt`:

- [`beta_over_nt_1p8/`](beta_over_nt_1p8/): five accepted continuum masses,
  `Nt=2--6`, plus line-specific volume validation;
- [`beta_over_nt_2p0/`](beta_over_nt_2p0/): four accepted continuum masses,
  `Nt=2--5`, and the exact `Nt=1` PCM reference;
- [`beta_over_nt_2p1/`](beta_over_nt_2p1/): accepted `Nt=2,3`, rejected `Nt=4`,
  the massless preliminary `Nt=5` snapshot, and the `Nt=1` PCM reference;
- [`comparison/`](comparison/): the exact selection, only `p=1` continuum
  fits, all cross-line plots, and the preliminary-statistics checklist.

Every active bootstrap analysis uses exactly 1,000 replicas. The 10,000-
replica confirmation products are kept only under
`comparison/provenance/superseded_10000_mass_reanalysis/` and are not selected,
fitted, or plotted.

Each ensemble directory contains its compact `analysis.json` and
`covariance.npz` when a full analysis exists. It also contains
`first_stage/thermalization.pdf` and `first_stage/blocking_error.pdf`; rejected
or incomplete ensembles retain their diagnostics without being promoted to a
mass result. Raw histories remain on qbig.
