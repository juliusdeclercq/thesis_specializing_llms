# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:58:38 2025

@author: Julius de Clercq
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load gradient norm CSV
df_train = pd.read_csv("IT0 gradnorm.csv")
df_train.columns = ["iteration", "gradient norm"]

# Convert iteration to epoch
ITER_PER_EPOCH = 415
df_train["epoch"] = df_train["iteration"] / ITER_PER_EPOCH

sns.set_theme(style="whitegrid")
sns.set_context("talk")

fig, ax = plt.subplots(figsize=(8, 5))

# Gradient norm line (log scale on y-axis)
sns.lineplot(
    data=df_train,
    x="epoch",
    y="gradient norm",
    color="tab:blue",
    linewidth=2,
    ax=ax
)

ax.set_yscale("log")

# Force x-axis to start at 0
ax.set_xlim(left=0)

# Only horizontal grid lines
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.grid(True, axis="x", linestyle="--", alpha=0.5)
# ax.grid(False, axis="x")

sns.despine(ax=ax)

ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient norm")

# Remove legend if automatically added
if ax.get_legend():
    ax.get_legend().remove()

plt.tight_layout()
plt.savefig(
    "IT0 gradnorm.pdf",
    bbox_inches="tight",
    dpi=300
)
plt.show()


