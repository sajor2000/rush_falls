"""JAMA Network Open figure styling, color palette, and save utilities."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ── JAMA-style matplotlib rcParams ──────────────────────────────────
JAMA_STYLE: dict[str, object] = {
    # Typography
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    # Resolution
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # Ensure deterministic light backgrounds & dark text (Marimo dark theme
    # would otherwise inherit white text, making saved PNGs unreadable)
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    # Axes
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Lines / markers
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    # Legend
    "legend.frameon": False,
}

# ── Color palette (colorblind-friendly) ─────────────────────────────
COLORS: dict[str, str] = {
    "epic": "#2166AC",
    "morse": "#B2182B",
    "treat_all": "#777777",
    "treat_none": "#333333",
    "ci_fill": "0.85",
}

# ── Standard figure sizes (inches) ──────────────────────────────────
FIG_SINGLE_COL = (3.5, 2.8)
FIG_DOUBLE_COL = (7.0, 4.5)
FIG_MULTI_PANEL = (7.0, 8.0)

# ── Output directory ────────────────────────────────────────────────
FIGURES_DIR = Path("outputs/figures")


def apply_jama_style() -> None:
    """Apply JAMA rcParams globally."""
    mpl.rcParams.update(JAMA_STYLE)


def save_figure(
    fig: Figure,
    name: str,
    formats: tuple[str, ...] = ("pdf", "png"),
    output_dir: Path = FIGURES_DIR,
    *,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.1,
) -> None:
    """Save figure in multiple formats for JAMA submission.

    Creates the output directory if it doesn't exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(
            output_dir / f"{name}.{fmt}",
            format=fmt,
            dpi=300,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
    plt.close(fig)
