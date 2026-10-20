# Nt=1, Ns=128 matched-beta/Nt diagnostics

Four independent hot-start chains of 8,000 composite sweeps at each coupling:

| beta = beta/Nt | Inputs |
|---:|---|
| 1.45 | [beta_1p45](beta_1p45/) |
| 1.50 | [beta_1p5](beta_1p5/) |
| 4.6/3 | [beta_1p5333333333333333](beta_1p5333333333333333/) |
| 1.55 | [beta_1p55](beta_1p55/) |

Each beta directory also contains one separate 100-sweep timing input. The
10-sweep smoke test uses Ns=4. Timing and smoke histories are excluded from
the production analysis. The immutable manifest records each run's purpose.

Nt counts temporal links, with L2=2 stored slices and open temporal storage.
The c_0p5 ancestor identifies the source reference family: target_c is null
and these runs are excluded from the fixed-c continuum fits. Holonomy is
measured every sweep, profiles every five; Wilson loops and flow are disabled.
Raw histories remain on qbig. Future analysis uses 500 bootstrap replicas.

See [manifest](../../../nt1_scaling_manifest.json) and the project
[execution register](../../../../../running_simulations.md) for provenance,
job IDs and completion status.
