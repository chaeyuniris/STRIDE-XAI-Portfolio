# viz/interaction.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.viz.base import StrideViz

def plot_pair_heatmap(StrideViz, feat_names, pair_contribs, top_m: int = 30, title="Interaction (pair) importance heatmap"):
    """Plots a heatmap of the strongest pairwise feature interactions.

    Args:
        StrideViz: An instance of the StrideViz class.
        feat_names (list): List of all feature names.
        pair_contribs (list): A list of tuples (feat_idx_i, feat_idx_j, importance).
        top_m (int): The maximum number of interactions to display.
        title (str): The title of the plot.
    """
    if not pair_contribs:
        fig, ax = plt.subplots(figsize=(5,2))
        ax.text(0.5, 0.5, "No pair_contribs available", ha="center", va="center")
        ax.axis("off")
        StrideViz._finalize(fig, title, "pair_heatmap.png")
        return

    pairs = sorted(pair_contribs, key=lambda x: -x[2])[:min(top_m, len(pair_contribs))]
    # 유니크 피처만 축으로
    used = sorted(set([i for i,_,_ in pairs] + [j for _,j,_ in pairs]))
    idx_map = {p:i for i,p in enumerate(used)}
    M = np.zeros((len(used), len(used)))
    for i, j, v in pairs:
        a, b = idx_map[i], idx_map[j]
        M[a,b] = M[b,a] = v

    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(used)), max(5, 0.5*len(used))))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(used))); ax.set_yticks(np.arange(len(used)))
    ax.set_xticklabels([feat_names[u] for u in used], rotation=45, ha="right")
    ax.set_yticklabels([feat_names[u] for u in used])
    fig.colorbar(im, ax=ax, shrink=0.8, label="pair importance")
    StrideViz._finalize(fig, title, "pair_heatmap.png")

def plot_synergy_heatmap_signed(StrideViz, synergy_df: pd.DataFrame, title="Signed synergy heatmap (top |synergy|)",
                                top_m: int = 30):
    """Visualizes interactions, distinguishing between synergy (+) and redundancy (-).

    This plot is a key unique capability of STRIDE.

    Args:
        StrideViz: An instance of the StrideViz class.
        synergy_df (pd.DataFrame): DataFrame containing 'feat_i', 'feat_j', and 'synergy_signed'.
        title (str): The title of the plot.
        top_m (int): The maximum number of interactions to display based on absolute synergy.
    """
    if synergy_df.empty:
        fig, ax = plt.subplots(figsize=(5,2))
        ax.text(0.5,0.5,"No synergy data", ha="center", va="center"); ax.axis("off")
        return StrideViz._finalize(fig, title, "ins_synergy_heatmap.png")
    top = synergy_df.head(min(top_m, len(synergy_df)))
    used = sorted(set(top["feat_i"]).union(set(top["feat_j"])))
    idx = {name:i for i,name in enumerate(used)}
    M = np.zeros((len(used), len(used)), dtype=float)
    for _, row in top.iterrows():
        a, b, s = idx[row["feat_i"]], idx[row["feat_j"]], float(row["synergy_signed"])
        M[a,b] = M[b,a] = s
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(used)), max(5, 0.5*len(used))))
    im = ax.imshow(M, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(used))); ax.set_yticks(np.arange(len(used)))
    ax.set_xticklabels(used, rotation=45, ha="right"); ax.set_yticklabels(used)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("signed synergy  (+: synergy, −: redundancy)")
    return StrideViz._finalize(fig, title, "ins_synergy_heatmap.png")

