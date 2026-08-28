# Fixed-beta/Nt mixed pilot

`manifest.tsv` is the immutable 32-chain expansion of the eight-point pilot
grid. `array_index` is the production Slurm index; `physical_Nt` counts links
between Dirichlet surfaces and `stored_L2=physical_Nt+1`. Input hashes cover
the exact UTF-8 files staged into each remote run directory.

The generator refuses to overwrite any input, manifest, or timing target. To
test regeneration without touching the campaign, use a temporary output:

```bash
uv run --frozen python data/beta_nt_fixed/pilot/generate_inputs.py \
  --output /tmp/beta-nt-fixed-pilot-regeneration
```

Each composite sweep is one SU(2) heatbath sweep followed by five
overrelaxation sweeps. `measurement_interval: 1` controls holonomy output;
`dirichlet_suite_measurement_interval: 5` controls the spatial and temporal
plaquette profiles. No Wilson-loop or horizontal-Polyakov output is enabled.

The analysis references and numerical conventions are documented in the
module docstrings of `scripts/thermalization_diagnostics.py`,
`scripts/pcm_beta_pilot_analysis.py`, `scripts/dirichlet_analysis.py`, and
`scripts/beta_nt_fixed/analysis.py`. In particular, independent chains are
never concatenated before thermalization, blocking, or resampling.
