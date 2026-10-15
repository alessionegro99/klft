"""Check scientific provenance and bootstrap policy in the logbook workflow."""

from pathlib import Path
import errno
import tempfile
import unittest
from unittest.mock import patch

from beta_nt_fixed.analysis import analyse_ensemble
from mg_nt_fixed_plot_layout import arrange_plots, plot_destination
from mg_nt_fixed_logbook import check_freshness


class LogbookTest(unittest.TestCase):
    def test_plot_routing_across_filesystems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scratch"
            source.mkdir()
            (source / "mass_brackets_c_1p0.pdf").write_bytes(b"test figure")
            with patch("shutil.os.rename", side_effect=OSError(errno.EXDEV, "cross device")):
                moves = arrange_plots(source, [], root / "comparison")
            self.assertEqual(moves[0][1].read_bytes(), b"test figure")
            self.assertFalse(moves[0][0].exists())

    def test_simulation_routing_matches_numeric_beta(self) -> None:
        source = Path("pilot/Nt_3_Ns_80_b_4p40/analysis.json")
        result = {"physical_Nt": 3, "Ns": 80, "beta": 4.4}
        name = "C_G_mass_fit_c_0p5_Nt_3_Ns_80_b_4p4.pdf"
        self.assertEqual(plot_destination(name, [(source, result)], Path("comparison")),
                         source.parent / "plots" / name)

    def test_duplicate_simulations_require_source_disambiguation(self) -> None:
        result = {"physical_Nt": 3, "Ns": 80, "beta": 4.4}
        with self.assertRaisesRegex(ValueError, "expected one source simulation"):
            plot_destination("C_G_mass_fit_c_0p5_Nt_3_Ns_80_b_4p4.pdf",
                             [(Path("pilot/analysis.json"), result),
                              (Path("replacement/analysis.json"), result)], Path("comparison"))

    def test_multi_simulation_routing(self) -> None:
        self.assertEqual(plot_destination("mass_brackets_c_1p0.pdf", [], Path("comparison")),
                         Path("comparison/c_1/mass_brackets_c_1p0.pdf"))
        self.assertEqual(plot_destination("periodic_correlator_volume_beta_4p95.pdf", [], Path("comparison")),
                         Path("comparison/volume/Nt_4_b_4p95/periodic_correlator_volume_beta_4p95.pdf"))

    def test_analysis_rejects_non500_before_reading_histories(self) -> None:
        for n in (100, 499, 501, 1000, 10000):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "exactly 500"):
                analyse_ensemble([], {}, 4, 48, 4.6, "test", n, 1, Path("unused.npz"))

    def test_new_result_invalidates_logbook(self) -> None:
        saved = {"source_sha256": {"old.json": "a"}, "output_sha256": {}}
        with patch("mg_nt_fixed_logbook.read_json", return_value=saved), patch(
            "mg_nt_fixed_logbook.source_hashes", return_value={"old.json": "a", "new.json": "b"}
        ), self.assertRaisesRegex(ValueError, "new.json"):
            check_freshness()


if __name__ == "__main__":
    unittest.main()
