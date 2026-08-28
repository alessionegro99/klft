# Preliminary-statistics checklist

- [x] Keep independent chains separate through thermalization, blocking, and
  the hierarchical joint bootstrap.
- [x] Use exactly 1,000 replicas in every active bootstrap operation. Each
  selected mass covariance has 1,000 finite mass replicas; each available
  continuum fit stores exactly 1,000 parameter replicas.
- [x] Retain only `p=1` continuum fits and remove `p=2` and free-exponent
  results from the compact comparison and plots.
- [x] Exclude points without an accepted asymptotic mass or without the full
  blocking plateau. No missing point is substituted.
- [x] Analyze the common 48,300-sweep `(5,688,10.5)` snapshot. Job `282512`
  completed, but no stable three-by-three mass-fit patch exists, so the point
  remains excluded.
- [x] Provide one thermalization and one blocking-error PDF for each of the 15
  simulated/reference ensembles. All 30 PDFs are grid-free and record the
  independent-chain cuts.
- [x] Regenerate every currently supported comparison plot: 64 grid-free,
  one-page 16:9 PDFs.
- [ ] Increase independent statistics for `(5,544,10.0)`. Its selected blocks
  leave only 38--40 blocks per chain, below the target of 50, and
  `m_G Nt=0.09266 +/- 0.00567` has a 6.1% relative error.
- [ ] Resolve `(4,528,8.4)` with longer histories or new independent chains.
  The present data have no common plateau across all 924 observables and a
  4.619-sigma maximum between-chain discrepancy.
- [ ] Replace or extend `(5,688,10.5)`. The four histories timed out at
  48,324--48,778 sweeps; the common 48,300 cutoff leaves only 32--33 selected
  blocks per chain and still yields no accepted mass.
- [ ] Perform direct spatial-volume checks. Accepted points
  `(beta/Nt,Nt)=(1.8,6)` and `(2.1,3)` have `Ns/xi_2nd=9.787` and `9.918`,
  below the scale near 10 that previously failed a correlator volume test;
  `(2.0,5)` is only `10.007`.
- [ ] Repeat the 1,000-replica analysis and `p=1` plots after the physical
  statistics and volume checks are increased.

Increasing bootstrap replicas is not an allowed substitute for increasing
physical statistics. Smaller physics errors require longer independent
histories or additional independent chains.
