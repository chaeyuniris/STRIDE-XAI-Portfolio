# viz/whatif.py

import numpy as np
import matplotlib.pyplot as plt
from src.viz.base import StrideViz

def plot_whatif(StrideViz, feat_names, phi_before_total, phi_after_total, target_feat, delta_mean, title=None):
    """Visualizes the results of a what-if simulation.

    Shows the change in feature importance ("phi_total") before and after
    a hypothetical change to a target feature.

    Args:
        StrideViz: An instance of the StrideViz class.
        feat_names (list): List of all feature names.
        phi_before_total (np.ndarray): Array of total importance values before the change.
        phi_after_total (np.ndarray): Array of total importance values after the change.
        target_feat (str): The name of the feature that was changed.
        delta_mean (float): The change in the model's mean prediction.
        title (str, optional): The title of the plot.
    """
    phi_b = np.asarray(phi_before_total); phi_a = np.asarray(phi_after_total)
    delta = phi_a - phi_b
    order = np.argsort(-np.abs(delta))[:min(10, len(delta))]

    names = [feat_names[i] for i in order]
    before = phi_b[order]; after = phi_a[order]; dlt = delta[order]

    ttl = title or f"What-if on '{target_feat}': Δmean_pred={delta_mean:+.4f}"
    fig = plt.figure(figsize=(max(7, 0.7*len(names)), 4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(np.arange(len(names))-0.2, before, width=0.4, label="before")
    ax1.bar(np.arange(len(names))+0.2, after,  width=0.4, label="after")
    ax1.set_xticks(np.arange(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("φ_total")
    ax1.legend()

    ax2 = fig.add_subplot(1,2,2)
    x = np.arange(len(names))
    ax2.bar(x, dlt)
    ax2.axhline(0, lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Δφ_total")
    StrideViz._finalize(fig, ttl, "whatif.png")