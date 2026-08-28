# Nonperturbative fixed-`m_G Nt` campaign

The immutable 25-point pilot is in `pilot/`. It contains 100 new hot-start
chains, a 100-sweep largest-grid timing input, two explicitly labelled prior
points from `beta_nt_fixed`, and dependency/input hashes. The generator uses
master seed `20260724` and refuses all overwrites.

The trajectory is defined by `target_c = m_G Nt`, not by `beta/Nt`. Raw qbig
histories are written below `/qbigwork2/negro/dirichlet/data/mg_nt_fixed/` and
must not be mixed with the legacy raw directories used by `beta_nt_fixed`.
The canonical campaign status and gates are in
[`docs/mg_nt_fixed/summary.md`](../../docs/mg_nt_fixed/summary.md).
