# viz/stride_viz.py

import numpy as np
import matplotlib.pyplot as plt
from src.viz.base import StrideViz

def plot_rank_summary(StrideViz, rho_main, rho_total, tau_main, tau_total,
                        overlap_main: float | None = None, overlap_total: float | None = None,
                        title="Rank correlation"):
    """Creates a table summarizing rank correlation metrics.

    Compares STRIDE's feature rankings (main and total) against a baseline like SHAP.

    Args:
        StrideViz: An instance of the StrideViz class.
        rho_main (float): Spearman correlation for main effects.
        rho_total (float): Spearman correlation for total effects.
        tau_main (float): Kendall's Tau for main effects.
        tau_total (float): Kendall's Tau for total effects.
        overlap_main (float | None): Top-k overlap for main effects.
        overlap_total (float | None): Top-k overlap for total effects.
        title (str): The title of the plot.
    """
    rows = [
        ("Spearman", rho_main, rho_total),
        ("Kendall",  tau_main, tau_total),
    ]
    if overlap_main is not None and overlap_total is not None:
        rows.append((f"Top-k overlap", overlap_main, overlap_total))

    fig, ax = plt.subplots(figsize=(6, 1.2*len(rows)+1))
    ax.axis("off")
    table = ax.table(
        cellText=[[r[0], f"{r[1]:.3f}" if r[1] is not None else "NA", f"{r[2]:.3f}" if r[2] is not None else "NA"] for r in rows],
        colLabels=["metric", "vs main", "vs total"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.4)
    StrideViz._finalize(fig, title, "rank_corr.png")