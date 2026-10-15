# Shareable tables

One row is one named statistical unit. Missing numerical results are empty
CSV fields, with a flag or failure reason. All masses are in lattice units
(`mass = a mG`); all cuts, block lengths and `tau_int_sweeps` are composite
sweeps. One sweep is one heatbath plus five overrelaxation updates.

Global tables include `campaign_type`, `target_c`, `reference_c`,
`physical_Nt`, `Ns`, `beta` and `fixed_c_continuum_input`. For the new
`nt1_matched_beta` rows, `target_c` is empty and `reference_c` identifies
the source Nt=2 tuning. Their folder c value has this reference meaning.
They are excluded from fixed-c tuning and continuum fits regardless of
the manual numerical-inclusion flag. `comparison/nt1_ensembles.csv` is
the compact three-row Nt=1 register.

- `ensembles.csv`: one physical `(c,Nt,Ns,beta)`; chain/sweep totals, used
  measurements, mass and bootstrap standard error, selected R window, flags.
  `mass_Nt` and `mass_Ns` use the measured mass at that coupling;
  `Ns_over_xi_2nd` is the ensemble structure-factor volume diagnostic.
- `chains.csv`: one independent seed; original qbig path, source campaign,
  total/cut/retained/used sweeps, discarded final partial block, block size,
  block count, fixed pooled weight, representative and maximum tau, flags.
- `autocorrelations.csv`: one chain and observable, including every R;
  measurement cadence and tau in sweeps. Profiles have cadence five.
- `chain_consistency.csv` (each ensemble): the largest standardized
  discrepancy and its observable for each independent-chain pair.
- `correlators.csv`: one ensemble and R; C_G and 500-replica bootstrap SE.
- `fit_windows.csv`: every mass window, including undefined fits; mass and
  amplitude errors, full covariance chi-square, nominal dof, covariance
  condition number, number of successful refits, selected window and flags.
  For the large Nt=1 boxes, `scan_stage` distinguishes coarse endpoints
  from dense adjacent checks; every integer R inside each window is used.
  `adaptive_endpoint_scan_review` means that the recommendation is earliest
  among tested patches. It does not certify an exhaustive endpoint search.
- `blocking_scan.csv` (each ensemble): every B and monitored observable;
  bootstrap SE, approximate SE uncertainty, minimum/maximum chain block
  counts and deterministic bootstrap seed. Mass/xi-exp scans hold the
  suggested mass window fixed. Approximate SE uncertainties select diagnostic
  plateaus; observable uncertainties themselves are bootstrap estimates.
- `interpolation_windows.csv` (each Nt comparison): all contiguous triples;
  bracket status, beta-star median and 16–84% interval, separate largest
  two-point shift, full-covariance chi-square, refit counts and flags.
- `tuning.csv`: the chosen interpolation per `(c,Nt)`, its source ensembles,
  uncertainty interval, interpolation systematic and propagated flags.
- `continuum.csv`: p=1 only; `Q0` extrapolates `c beta_star/(4 Nt)`;
  `all_Nt` and `drop_coarsest` are sensitivity fits. A two-point line has zero
  residual dof. Statistical bands exclude fit-window and volume systematics.
  `Q0_interpolation_shift_bound` is a separate conservative linear-response
  bound from independently signed two-point interpolation shifts at each Nt;
  it is not combined in quadrature with the statistical error.
  `drop_coarsest_shift` and its paired-replica bootstrap error compare the
  two resolution ranges while retaining their shared-data covariance.

The full covariance describes separations jointly. Central and replica mass
fits minimize the diagonal-error objective. `chi2_full` is evaluated after
fitting and is descriptive; there is no analytic p-value or chi-square veto.
A covariance from 500 replicas has rank at most 499. Wider windows have
empty full-covariance chi-square fields and `singular_full_covariance`;
the diagonal-fit parameters and their bootstrap errors remain available.

`length_outlier` identifies exactly the four 80,000-sweep chains. It is a
duration label. `prior_10000_sweeps` labels the eight prior-input chains.
`between_chain_review` and `split_history_review` mark a maximum standardized
discrepancy above three over many correlated observables; they are not
calibrated significance tests. Other flags name their quantitative condition.
`late_window_drift` marks a later window at the same Rmax differing from the
chosen mass by more than two combined SEs. This is a heuristic across
correlated windows. `finite_volume_safety_review` means measured `mass_Ns < 12`,
the nominal campaign volume target; it is a warning, not an exclusion.
`continuum_resolution_range_sensitivity` flags a drop-coarsest shift larger
than twice its paired-replica SE. All requested fits remain visible.

## Manual review

Edit `analysis/mg_nt_fixed/review.csv`. Keep `scope` and `key` fixed. For an
ensemble, set both `r_min` and `r_max` to choose a saved numerical window.
`decision` is `auto`, `include` or `exclude`. Auto includes defined flagged
results in preliminary downstream work. For a tuning row, `betas` can select
an available bracketing triple as semicolon-separated numeric values. Excluding
a tuning row removes that resolution from the continuum fit. Add a short note.
Rerun `scripts/mg_nt_fixed_campaign.py`, then the logbook builder, on qbig.

`bootstrap.npz` contains exactly 500 aligned replicas per operation, full
correlator covariance, all-window parameter draws, blocking scans and chain
estimates. Failed refits remain NaN in their original slots; no replacement
draws are made. The sample SD is conditional on successful refits, with their
count recorded. Operation seeds and input/code hashes are in `analysis.json`.

Import RNG seeds and hashes as text when using spreadsheet software; 64-bit
bootstrap seeds can exceed a spreadsheet's exact numeric-integer precision.

The short/long/pooled diagnostic uses the same per-chain replicas for all
three estimators, preserving their joint covariance. Its pooled uncertainty
can differ slightly from the main estimate's SE because the main bootstrap
uses its own recorded 500-draw stream. Central estimates use identical data
and weights. A manual mass-window override flags its blocking diagnostic for
review; the first-stage mass-error curve is explicitly for the suggested window.
