#!/usr/bin/env python3
"""Fast normalization checks for the thesis comparison."""

from __future__ import annotations

import unittest

from compare_thesis import normalized_trace


class CompareThesisTest(unittest.TestCase):
    def test_normalized_trace(self) -> None:
        self.assertEqual(
            normalized_trace({"mean_TrG": 1.2, "mean_TrG_error": 0.08}),
            (0.6, 0.04),
        )
        self.assertEqual(
            normalized_trace(
                {
                    "mean_TrG": 1.2,
                    "mean_TrG_error": 0.08,
                    "mean_half_TrG": 0.59,
                    "mean_half_TrG_error": 0.03,
                }
            ),
            (0.59, 0.03),
        )


if __name__ == "__main__":
    unittest.main()
