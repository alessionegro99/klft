# Fixed-beta/Nt inputs

Active immutable inputs are grouped by the exact continuum line:

- [`beta_over_nt_1p8/inputs/`](beta_over_nt_1p8/inputs/): selected
  `Nt=2--6` ensembles;
- [`beta_over_nt_2p0/inputs/`](beta_over_nt_2p0/inputs/): `Nt=2--5`;
- [`beta_over_nt_2p1/inputs/`](beta_over_nt_2p1/inputs/): `Nt=2--5`;
- [`comparison/`](comparison/): the shared first-stage manifest, historical
  immutable manifests, qbig wrappers, and superseded campaign provenance.

The line-1.8 finite-volume workflows are under
`beta_over_nt_1p8/validation/`. The exact `Nt=1` PCM reference inputs belong to
the older `data/dirichlet_thesis/qbig/pcm_beta_pilot/` campaign and are not
duplicated here.

Moving the local inputs did not change their bytes. Historical input headers,
qbig raw-data paths, and manifest `run` fields retain their original names so
their recorded SHA-256 values remain valid. Raw histories remain on qbig.
