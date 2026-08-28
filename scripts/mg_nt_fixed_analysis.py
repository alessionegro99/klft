#!/usr/bin/env python3
"""CLI for chain-aware nonperturbative fixed-``m_G N_t`` analysis.

The statistical implementation is shared with ``beta_nt_fixed.analysis`` so
the audit and the new campaign use exactly the same thermalization, blocking,
bootstrap, covariance, and periodic-K0 acceptance code.
"""

from beta_nt_fixed.analysis import main


if __name__ == "__main__":
    main()
