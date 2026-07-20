#!/usr/bin/env python3
"""Tests that enforce the project plotting rules."""

from pathlib import Path
import unittest
from unittest import mock

import matplotlib.pyplot as plt

from plot_thesis_results import FIGURE_SIZE, save_figure


class PlotRulesTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_rejects_non_pdf_output(self) -> None:
        figure, _ = plt.subplots(figsize=FIGURE_SIZE)
        with self.assertRaisesRegex(ValueError, "must be PDF"):
            save_figure(figure, Path("figure.png"))

    def test_rejects_non_16_by_9_figure(self) -> None:
        figure, _ = plt.subplots(figsize=(8.0, 6.0))
        with self.assertRaisesRegex(ValueError, "figsize"):
            save_figure(figure, Path("figure.pdf"))

    def test_rejects_multiple_plots(self) -> None:
        figure, _ = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        with self.assertRaisesRegex(ValueError, "exactly one plot"):
            save_figure(figure, Path("figure.pdf"))

    def test_accepts_one_16_by_9_pdf_plot(self) -> None:
        figure, _ = plt.subplots(figsize=FIGURE_SIZE)
        with mock.patch.object(figure, "savefig") as savefig:
            save_figure(figure, Path("figure.pdf"))
        savefig.assert_called_once_with(Path("figure.pdf"))


if __name__ == "__main__":
    unittest.main()
