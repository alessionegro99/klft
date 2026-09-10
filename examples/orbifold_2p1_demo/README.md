# Finite-mass SU(3), 2+1D demonstration campaign

This directory starts the requested unsmeared `32^3` orbifold-HMC campaign.
It is not yet a calibrated production recipe or a delivered potential.
All directions are periodic; directions 0,1 are spatial and 2 is time.
The full unfixed action uses `a_s=a_t=0.2`, `g=1`, `m=m_U1=40`, so both
bare masses satisfy `ma=8` and the constrained-limit Wilson coupling is 15.
These are finite-mass orbifold runs, not compact Wilson heatbath runs.

First run `../qbig_orbifold_2p1_smoke.slurm` with its clean pinned source
checkout. After its GPU deterministic and smoke/restart checks pass, stage
the three `initial_*.yaml` files and `initial_tune.slurm` under
`/qbigwork2/negro/orbifold/campaigns/orbifold_2p1_demo_20260910/`.
Inspect Slurm and GPU occupancy, then submit the array on one available
P100 node, for example `sbatch --nodelist=lnode14 initial_tune.slurm`.

The pilot tests `tau=0.02` at steps `0.0005`, `0.001`, and `0.002`, using
distinct hot-start seeds `26091101`, `26091201`, and `26091301` (the HMC
momentum seed is the input seed plus one). Each runs 60 unmeasured
trajectories, writes action and W(1,1) diagnostics every trajectory, and
saves a final checkpoint. The script refuses existing run directories,
checks input/executable hashes and the GPU smoke completion marker, and
keeps all histories on qbig.

Choose a stable initial integrator from these results, warm up independent
chains, then retune toward 70% acceptance on the warmed fields. Only after
checking thermalization should Wilson-loop production begin. The intended
measurement range is `R,T<=16`; a modest-statistics curve at several resolved
separations suffices, with autocorrelation-aware uncertainties and stable
time plateaus. Do not accept these initial tuning diagnostics as that curve.
