# beta/Nt = 2.0

The continuum selection uses accepted 1,000-replica masses at `Nt=2,3,4,5`.
`Nt=1,beta=2.0` is the exact PCM reference and is not fitted. The `Nt=5`
point passes the hard gates but has only 38--40 blocks per chain and needs
more physical statistics.

Every directory has one thermalization and one blocking-error PDF under
`first_stage/`. The current `p=1` fit is in [`../comparison/`](../comparison/).
