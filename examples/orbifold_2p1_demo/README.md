# Finite-mass SU(3), 2+1D demonstration

The first unsmeared `32^3` orbifold-HMC demonstration is complete. It uses
the **full unfixed orbifold action and HMC**, not compact Wilson heatbath.
Measured-stage acceptance is **75.5%, 75.6%, 75.93%** in three independent
chains. Target 70--80%; sustained acceptance outside 70--90% requires retuning
during warmup, then a fixed integrator for retained measurements.

## Physics and measured result

All directions are periodic; 0,1 are spatial and 2 is Euclidean time.
Parameters are `a_s=a_t=0.2`, `g=1`, `m=m_U1=40`, `gamma=0`:
both bare `ma=8`, with constrained-limit Wilson coupling 15.
Spatial matrices remain noncompact; normalized loops use their full U(3)
polar factors and the explicit dynamical SU(3) temporal links. No smearing
or temporal gauge fixing is used.

Each chain has 1,550 unmeasured trajectories across restart stages, followed
by 3,000 measured-stage trajectories, measuring all `R,T=1,...,16` every ten.
The final analysis uses 288 of the 300 loop vectors per chain, in nine
32-vector blocks (320 trajectories each). Independent chains are kept
separate, with 10,000 vector hierarchical-bootstrap samples, seed 26091107.

Full-covariance constant fits to
`a_t V_eff(R,T+1/2)=log[W(R,T)/W(R,T+1)]` at the common late window
`T=7,8,9` give:

| R | a_t V | Bootstrap SE |
|---|---:|---:|
| 1 | 0.17535 | 0.00018 |
| 2 | 0.28573 | 0.00071 |
| 3 | 0.37034 | 0.00130 |
| 4 | 0.44768 | 0.00365 |
| 5 | 0.51724 | 0.00465 |
| 6 | 0.58087 | 0.00939 |
| 7 | 0.64627 | 0.01618 |

This is a modest-statistics implementation/physics demonstration, not a
precision or continuum result. Neighboring time windows and extra
thermalization cuts shift values by up to about two quoted main-fit errors;
these shifts are diagnostics, not separately estimated systematic errors.
Doubling blocks from 16 to 32 vectors changes values by at most 0.16 main-fit
SE. Do not claim a plateau at R>=8: conservative bootstrap positivity fails.
No mass, lattice-spacing or volume extrapolation, string-tension fit, or
Sommer scale is claimed.

## Validated workflow

The immutable campaign stages are:

1. `../qbig_orbifold_2p1_smoke.slurm`: clean pinned CUDA/P100 build,
   deterministic tests, CLI smoke/restart and input guards. Accepted source
   `1104e1af3fa39838d1ae6fdb9c2620f7b423b134`; job 289558.
2. `warmup_initial.slurm`: three independent 50-trajectory hot-start
   preconditioning runs. Its tiny step gave 98--100% acceptance; it is
   **not a sustained or production setting**.
3. `postwarm_tune.slurm`: warmed-field diagnostic scan, selecting
   `tau=0.1`, 57 leapfrog steps (`h=0.1/57`, YAML request `0.00175`).
   The selected candidate accepted 78%. Calibration clones are not
   independent production chains.
4. `equilibrate.slurm` and `equilibrate_chain*.yaml`: 1,000 more
   unmeasured trajectories per original independent chain, acceptance
   75.7/75.9/77.1%; job array 289570.
5. `pilot.slurm` and `pilot_chain*.yaml`: another 500 unmeasured
   trajectories, then the 3,000-trajectory measured pilot; array 289573.
   Momentum seeds are 26095101/26095201/26095301; fixed integrator as above.
6. `analyze_pilot.slurm`: checksum-guarded, locked-uv analysis of three
   extra thermalization cuts; job 289576.
7. `extract_pilot.slurm`: longer-block and neighboring-window checks,
   then the seven late-time plateaus; job 289577.

The earlier `initial_tune.slurm` hot-start attempts had zero acceptance;
they remain diagnostic history, not valid starting states.

## Run and reproduce

These inputs are pinned to the existing qbig campaign:
`/qbigwork2/negro/orbifold/campaigns/orbifold_2p1_demo_20260910/`.
Raw histories/checkpoints remain under the corresponding remote `data/`
directory. Final tables, covariance, and provenance are under
`analysis/orbifold_2p1_demo_20260910/measured_pilot/potential/` on qbig.

For a new chain, use a new working/output directory and independent seed.
A [measured input](pilot_chain01.yaml) requires a compatible **orbifold**
checkpoint, not a compact-Wilson checkpoint. Update its restart path.
The scripts refuse existing outputs and verify source, executable, input
and restart hashes. Do not edit already-run campaign files in place.
Inspect Slurm/GPU occupancy and request one typed Pascal GPU before running
on qbig; never run production on the login node.

Plot the compact result tables with the shared Bonn plotting dependency
and a uv environment containing numpy/matplotlib:

```bash
uv run --frozen --project /path/to/plot-environment python \
  analysis/plot_orbifold_2p1_demo.py /path/to/static_potential.tsv \
  --style-dir /path/to/statanalysis-Bonn --effective-r 4
```

This makes one potential PDF and one R=4 plateau PDF, each a single plot
on an uncropped 16:9 canvas with LaTeX text. The accompanying
`static_potential.effective.tsv` must be beside the main table.
