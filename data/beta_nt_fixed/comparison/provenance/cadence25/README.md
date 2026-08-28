# Fixed-`g_3^2L` continuum production, cadence-25 revision

Regenerate these immutable inputs with

```bash
UV_CACHE_DIR=/tmp/dirichlet-uv-cache uv run python \
  data/beta_nt_fixed/production/generate_inputs.py --revision cadence25
```

The ten ensembles lie on `beta/Nt = 1.8, 2.0, 2.1`, hence
`g_3^2L = 20/9, 2, 40/21`. Each has four independent hot-start chains and
50000 composite sweeps. Holonomy observables use cadence one, plaquette
profiles cadence five, and all-height boundary Wilson loops cadence 25.

This revision supersedes, but does not overwrite, the failed cadence-20 timing
revision. Smoke job `282270` and largest-case timing job `282271` passed; the
latter projects 91.42711556 hours per largest chain. Production array `282274`
uses all 40 P100s. See `running_simulations.md` for live state and full
provenance.
