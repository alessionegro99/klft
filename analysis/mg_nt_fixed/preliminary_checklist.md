# Preliminary-statistics checklist

The fixed-`m_G Nt` results currently determine pilot tuning decisions. They are
not yet final production measurements.

- [x] All 25 pilot points and both round-1 extension points use complete,
  immutable histories and four independent chains.
- [x] Every reported mass passes the asymptotic-window, bootstrap-refit, and
  mass/`xi_exp` blocking-plateau gates with 1000 finite mass replicas.
- [x] Round-1 tuning uses only a local three-point target bracket; no failed
  group is extrapolated.
- [x] Generate all currently supported pilot plots without background grids:
  84 one-page 16:9 PDFs covering tuning, accepted mass fits, correlators,
  profiles, length estimators, volumes, and symmetry diagnostics. Label both
  fixed-`beta/Nt` audit inputs as prior information.
- [ ] Increase independent statistics at `Nt=3,Ns=80,beta=4.40`. Its 3.14%
  relative mass error is much larger than the approximately 0.4% errors at
  beta 4.20 and 4.27.
- [ ] Add another target-bracketing `c=0.5,Nt=3` point. The current local
  log-linear interpolation still fails with `p=0.01350`. A candidate near
  beta 4.290 is a placement estimate from the bracketing pair, not a tuned
  result.
- [ ] Increase statistics for `Nt=4,Ns=96,beta=6.20` if it enters a final
  result: 46 blocks per chain pass the hard floor of 30 but miss the target of
  50.
- [ ] Resolve the finite-volume warning exposed by the preliminary plots.
  `Ns/xi_2nd` is `7.313` at `(c,Nt,beta)=(1,4,5.30)`, `8.043` at
  `(0.5,4,6.20)`, and `9.476` at `(1,4,4.95)`. In particular, the accepted
  `c=1,Nt=4` three-point tuning fit uses the first and third points. Repeat at
  larger `Ns` or demonstrate volume stability before final tuning.
- [ ] Repeat borderline final goodness-of-fit calculations with 10000
  bootstrap replicas, especially `c=2,Nt=4`, whose interpolation has
  `p=0.05084`.
- [ ] Run the independent `Nt=4,c=1` confirmation only after all nine tuning
  groups pass.

More bootstrap replicas improve the numerical precision of resampling
estimates; they do not replace longer independent histories or additional
independent chains.
