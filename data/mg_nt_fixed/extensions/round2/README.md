# Second fixed-`m_G Nt` bracket extension

Round-1 tuning job `282508` left only `(c,Nt)=(0.5,3)` rejected because its
local log-linear interpolation had `p=0.01350`. This immutable extension adds
four independent 8000-sweep hot-start chains at
`(c,Nt,Ns,beta)=(0.5,3,80,4.290)`. The coupling is a placement estimate from
the target-bracketing beta-4.27 and beta-4.40 points, not a tuned result.

Production seeds are draws 113--116 from `np.random.default_rng(20260724)`;
draws 117 and 118 are reserved for the 25-sweep smoke and 100-sweep timing
checks. Generate once with:

```bash
uv run --frozen python data/mg_nt_fixed/extensions/round2/generate_inputs.py \
  --tuning-summary analysis/mg_nt_fixed/tuning_round1/tuning_summary.json
```

The generator verifies the sole remaining tuning failure, records every hash
and seed, and refuses every overwrite. Raw histories remain on qbig below
`data/mg_nt_fixed/extensions/round2/`; compact outputs use the matching path
below `analysis/mg_nt_fixed/`.
