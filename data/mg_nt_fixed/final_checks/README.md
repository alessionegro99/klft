# Fixed-mass final checks

This immutable campaign contains three four-chain ensembles:

- `Nt=3,Ns=80,beta=4.40`, 80000 sweeps per chain, replacing the noisy pilot
  measurement with fresh independent statistics;
- `Nt=4,Ns=96,beta=4.95`, 8000 sweeps per chain;
- `Nt=4,Ns=96,beta=5.30`, 8000 sweeps per chain.

The latter two directly check and replace the low-volume `Ns=48` inputs. The
filtered `reanalysis_manifest.tsv` records the existing three `c=2,Nt=4`
ensembles that are reanalyzed with 10000 bootstrap replicas. Raw histories
remain on qbig. The campaign was triggered only after round-2 tuning passed
all nine groups and retained beta 4.40 in the selected `c=0.5,Nt=3` triple.
