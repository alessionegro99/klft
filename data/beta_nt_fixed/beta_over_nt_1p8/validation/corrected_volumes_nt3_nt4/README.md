# Fixed-beta/Nt corrected `Nt=3,4` volumes

This campaign repeats the two longest-correlation-length fixed-beta/Nt
points at the production spatial sizes inferred from the mixed pilot:

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
uv run --frozen python \
  data/beta_nt_fixed/beta_over_nt_1p8/validation/corrected_volumes_nt3_nt4/generate_inputs.py \
  --output /tmp/beta-nt-fixed-corrected-volumes-regeneration
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

After production has completed, run the statistical stages explicitly and
inspect each one before starting the next:

```bash
qbig/queue_stage.sh diagnostics
qbig/queue_stage.sh analysis
qbig/queue_stage.sh summary
qbig/queue_stage.sh plots
```

These stages also have no scheduler dependencies. Diagnostics and full
analysis operate on the raw histories on qbig. They use separate-chain MSER-5
cuts, Sokal-window autocorrelations in sweep units, whole-history blocking
plateaus for every primary and secondary observable, and a 20,000-sample
hierarchical bootstrap. Synchronize only compact products, first with a dry
run:

```bash
qbig/sync_compact.sh --dry-run
qbig/sync_compact.sh
```
