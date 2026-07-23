# Step-3 corrected constant-coupling volumes

This campaign repeats the two longest-correlation-length constant-coupling
points at the production spatial sizes inferred from the Step-3 pilot:

| `Nt` | beta | pilot `xi_2nd` | corrected `Ns` |
|---:|---:|---:|---:|
| 3 | 5.4 | 16.581(631) | 176 |
| 4 | 7.2 | 25.177(1.044) | 256 |

The sizes use `16 ceil(10 xi_2nd/16)`. Each ensemble has four independent hot
starts and 20000 composite sweeps, where one composite sweep is one heatbath
sweep followed by five overrelaxation sweeps. Holonomy observables use cadence
one and plaquette profiles cadence five. Wilson loops, horizontal Polyakov
loops, PCM matching, and effective-action output remain disabled.

Regenerate into an empty temporary directory with:

```bash
uv run --frozen python data/step3_corrected_volumes/generate_inputs.py \
  --output /tmp/step3-corrected-volumes-regeneration
```

On qbig, submit and inspect the largest-grid 100-sweep timing job before
submitting production:

```bash
qbig/queue_stage.sh timing
qbig/queue_stage.sh production
```

Both stages are dependency-free. The production array has eight tasks with no
concurrency cap and requests one typed Pascal GPU per task. The timing gate
stops before production if the measured projection exceeds two hours for one
chain or 12 total P100 GPU-hours. Raw histories remain on qbig.
