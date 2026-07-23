# Step-3 `Ns=160` finite-volume extension

This campaign tests whether the Step-3 `Nt=2`, beta 3.6 correlator is stable
between `Ns=128` and 160. It is a separate immutable campaign because the
completed Step-3 manifest must not be changed.

- Geometry: physical `Nt=2`, stored `L2=3`, `Ns=160`
- Coupling: beta 3.6, `lambda_tree^2=20/9`
- Four independent hot starts, 10000 composite sweeps each
- Holonomy cadence 1; plaquette-profile cadence 5
- Wilson-loop, horizontal-Polyakov, PCM-matching, and effective-action output
  disabled
- Existing verified P100 executable reused by exact SHA-256; no second build

Regenerate into an empty temporary directory with:

```bash
uv run --frozen python data/step3_fv160/generate_inputs.py \
  --output /tmp/step3-fv160-regeneration
```

On qbig, submit `qbig/queue_stage.sh smoke`, validate its output, and then
submit `qbig/queue_stage.sh production`. Both stages are dependency-free.
Raw histories remain below `/qbigwork2/negro/dirichlet/data/`.
