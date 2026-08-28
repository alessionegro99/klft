# First fixed-`m_G Nt` bracket extension

Official tuning job `282495` accepted seven of nine groups. It required a new
point for `(c,Nt)=(0.5,3)` because the original three-point log-linear fit had
`p=0.009674`, and a new point for `(c,Nt)=(0.5,4)` because the target was not
bracketed. This immutable extension contains four independent 8000-sweep
hot-start chains at each of:

- `(c,Nt,Ns,beta)=(0.5,3,80,4.27)`, centered on the target-bracketing
  two-point interpolation;
- `(c,Nt,Ns,beta)=(0.5,4,96,5.50)`, below the original beta-5.60 endpoint to
  form a safe bracket.

Seeds are draws 105--112 from `np.random.default_rng(20260724)` and are
recorded in `seed_provenance.tsv`. Generate once with:

```bash
uv run python data/mg_nt_fixed/extensions/round1/generate_inputs.py \
  --tuning-summary analysis/mg_nt_fixed/tuning/tuning_summary.json
```

The generator verifies the two tuning failures and refuses every overwrite.
Raw histories remain on qbig below
`data/mg_nt_fixed/extensions/round1/`; compact outputs use the matching path
below `analysis/mg_nt_fixed/`.
