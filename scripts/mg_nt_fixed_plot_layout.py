"""Route fixed-mass figures to their simulation or comparison directory."""

from __future__ import annotations

from pathlib import Path
import re
import shutil


def plot_destination(name: str, results: list[tuple[Path, dict]], comparison: Path) -> Path:
    """Resolve a figure's physics parameters to one unambiguous result path."""
    match = re.search(r"_Nt_(\d+)_Ns_(\d+)_b_([0-9p]+)\.pdf$", name)
    if match:
        nt, ns, beta = int(match[1]), int(match[2]), float(match[3].replace("p", "."))
        matches = [p for p, r in results if (r["physical_Nt"], r["Ns"], r["beta"]) == (nt, ns, beta)]
        if len(matches) != 1:
            raise ValueError(f"expected one source simulation for {name}, found {len(matches)}")
        return matches[0].parent / "plots" / name
    target = re.search(r"_c_([0-9p]+)(?:_|\.pdf$)", name)
    if target:
        tag = f"{float(target[1].replace('p', '.')):g}".replace(".", "p")
        return comparison / f"c_{tag}" / name
    volume = re.search(r"periodic_correlator_volume_beta_([0-9p]+)\.pdf$", name)
    if volume:
        return comparison / "volume" / f"Nt_4_b_{volume[1]}" / name
    if name.startswith("mass_volume_stability"):
        return comparison / "volume" / name
    return comparison / name


def arrange_plots(source: Path, results: list[tuple[Path, dict]], comparison: Path) -> list[tuple[Path, Path]]:
    """Move complete PDF products into the requested tree without changing bytes."""
    moves = [(path, plot_destination(path.name, results, comparison))
             for path in sorted(source.glob("*.pdf"))]
    for path, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # qbig scratch (/tmp) and persistent analysis storage may be different
        # filesystems. move() falls back to copying before removing the scratch PDF.
        shutil.move(str(path), str(destination))
    return moves
