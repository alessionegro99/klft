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

The initial pilot completed with zero accepted proposals at all three steps;
none of its checkpoints is an equilibrated state. The next, explicitly
small-step attempt is `warmup_initial.slurm` with its three
`warmup_initial_chain*.yaml` inputs. It uses fresh independent hot seeds
`26092101`, `26092201`, `26092301`, `tau=0.1`, step `0.0001`, and 50
unmeasured trajectories per chain. It retains checkpoints every ten
trajectories, with diagnostics every trajectory. Inspect acceptance and
action evolution before extending or increasing the step; this initial
stage is not a tuned production input either.

That 50-trajectory startup completed with 98%, 98%, and 100% acceptance;
the fields still show thermalization drift. Do not reuse its tiny step for
sustained runs. The user's target is 70--80% acceptance, with 90% the upper
limit. Before extending, `postwarm_tune.slurm` compares five step sizes from
the same checksum-verified chain-1 endpoint at fixed `tau=0.1`. Each candidate
runs 100 diagnostic-only trajectories with its own seed. The effective step
counts are 100, 80, 67, 57, and 50. These calibration trajectories are not
independent production chains. Select a setting in the requested range from
the measured acceptance, then monitor and retune during further warmup if it
exceeds 90%. Freeze the final integrator before retained measurements.

The warmed-field scan completed: acceptance for 100/80/67/57/50 steps was
85/89/85/78/61 percent. Select 57 steps (`h=0.1/57`, requested YAML step
`0.00175`): both 50-trajectory halves accepted 78%. The next stage is
`equilibrate.slurm` with `equilibrate_chain*.yaml`, continuing the three
original independent 50-trajectory checkpoints, not the calibration clones.
It runs 1,000 unmeasured trajectories per chain with fresh momentum seeds
26094101/26094201/26094301, diagnostics every ten and checkpoints every 100.
The script checks each restart hash and requires 70--90% overall acceptance.
Inspect rolling acceptance during warmup; sustained values outside that range
require retuning. Check action and loop stationarity separately before
production: passing the script's acceptance gate does not prove equilibrium.

That 1,000-trajectory warmup completed with acceptance 75.7/75.9/77.1%.
Late W11 is near 0.8085, with much less drift than startup. The first
measured pilot is `pilot.slurm` with `pilot_chain*.yaml`: restart these
independent endpoints, discard another 500 trajectories, then measure all
256 unsmeared loops `R,T<=16` every ten trajectories for 3,000 trajectories
(300 loop vectors per chain). Seeds are 26095101/26095201/26095301. The
integrator is fixed at tau=0.1 with 57 steps; checkpoint spacing is 500.
Monitor acceptance during running. The completion marker verifies execution,
finite complete histories and 70--90% measured-stage acceptance, not physics.
Accept a potential only after chainwise thermalization, autocorrelation-aware
blocking and correlated plateau/window-stability checks on the measured loops.
