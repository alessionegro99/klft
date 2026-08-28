# Corrected preliminary 48300-sweep snapshot

The four `(Nt,Ns,beta)=(5,688,10.5)` production tasks timed out after
48324--48778 holonomy sweeps instead of completing 50000 sweeps. The largest
cutoff shared by holonomy, profile, and boundary-loop outputs is sweep 48300,
so every raw observable was truncated there before validation. Raw histories
remain on qbig.

The original high-block grid could not test three adjacent lengths above the
measured autocorrelation bound. The corrected analysis adds only the viable
1000- and 1400-sweep candidates; all candidates remain multiples of the
25-sweep common cadence and retain at least 30 blocks per chain. Focused job
`282511` completed in 5:43 with exit code zero. It finds the single common
plateau `[1000,1200,1400]` sweeps and selects 1400-sweep blocks, leaving
`32,32,33,32` complete blocks in the four chains. All chains pass the hard
30-block floor but miss the target of 50 blocks. The independent MSER-5 cuts
are `2125,2250,1750,3250` sweeps, and the maximum between-chain discrepancy is
3.102 standard deviations in `Tr_G_real`.

The byte-matched local/qbig diagnostic SHA-256 is
`a4e6a2911fb4ca6ebb7e9db5ab368d314ce0f209114ba6ba1a82db220360c579`.
The corrected analysis-program SHA-256 is
`8bd3ae8041623ba579a75affca3b333d0fb823314c384c29c60a8eaa34a3deea`;
the exact wrapper SHA-256 is
`b876d51af207425b83befd0d3d397c2270b5331c06614220ca896c6e151a8045`.

Seeded 1,000-replica mass-analysis job `282512` completed in 3:56:46 with exit
code zero. It produced 1,000 finite `xi_2nd` replicas, but no accepted mass:
eight individual windows pass the pointwise quality gates, while none belongs
to a stable three-by-three asymptotic patch. Therefore `mass` and `xi_exp` are
null and this point enters no continuum fit or mass plot.

The synchronized `analysis.json` and `covariance.npz` SHA-256 values are
`85833f39b691473786ff5e17964308adf7baa49437026041d3d43f60e413347a`
and `c9410d6201d7d8ef1ca10a2ba9580235579dc893ad950befbbc8cb63dcfba9cc`.
The `first_stage/` directory shows the four-chain 48,300-sweep history and its
accepted primary blocking plateau. These diagnostics do not override the
failed mass-fit stability gate or the missed 50-block target.
