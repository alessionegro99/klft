#!/usr/bin/env python3
"""CLI for chain-aware nonperturbative fixed-``m_G N_t`` analysis.

The canonical workflow retains flagged fits and uses pooled measurement
weights while preserving independent chain boundaries.
"""

from mg_nt_fixed_ensemble import main


if __name__ == "__main__":
    main()
