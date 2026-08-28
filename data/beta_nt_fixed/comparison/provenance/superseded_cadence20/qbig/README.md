# qbig production stages

The stages are intentionally submitted and inspected one at a time:

1. `build_p100.slurm` builds the clean pinned KLFT revision and runs both
   Dirichlet CTests.
2. `submit_smoke.slurm` checks dimensions, boundary conditions, all output
   counts, and the exact identity `W_b(R,Nt) = C_G(R)`.
3. `submit_timing.slurm` times the largest ensemble including its boundary
   loops and refuses production if the deliberately conservative linear
   projection exceeds 93 hours.
4. `submit_production.slurm` runs the 40 independent chains only after both
   gate files contain `PASS`.

Each production task requests one typed Pascal GPU. Raw histories remain below
`/qbigwork2/negro/dirichlet/data/beta_nt_fixed/production/`.

The first immutable revision used boundary cadence 20 and failed its timing
gate at a 104.77-hour projection. The `cadence25/` revision preserves that
failure record, uses fresh seeds, repeats both gates, and is the only revision
eligible for production.
