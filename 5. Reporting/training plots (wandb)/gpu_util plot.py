# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 19:12:24 2025

@author: Julius de Clercq
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("IT0 gpu util.csv")
df.columns = ["Runtime (hours)", "GPU utilization (%)"]

# Apply rolling average to smooth fluctuations
window_size = 10  # adjust depending on how smooth you want it
df["Rolling average"] = df["GPU utilization (%)"].rolling(window=window_size, center=True).mean()

# Set a clean style
sns.set_theme(style="whitegrid")
sns.set_context("talk")

fig, ax = plt.subplots(figsize=(10, 5))

# Raw line (light & thin)
sns.lineplot(
    data=df,
    x="Runtime (hours)",
    y="GPU utilization (%)",
    color="tab:blue",
    alpha=0.2,
    linewidth=0.8,
    ax=ax,
    label="Raw data"
)

# Smoothed line (thicker & prominent)
sns.lineplot(
    data=df,
    x="Runtime (hours)",
    y="Rolling average",
    color="tab:blue",
    linewidth=2,
    ax=ax,
    label=f"Rolling average"
)

# X-axis settings
ax.set_xlim(left=0)
ax.set_xlabel("Runtime (hours)")
ax.set_ylabel("GPU utilization (%)")

# Grid and borders
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.grid(False, axis='x')
sns.despine(ax=ax)

# Legend bottom right with opaque white background
legend = ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    # edgecolor="black",
    framealpha=0.95
)

plt.tight_layout()
plt.savefig("IT0 gpu util.pdf", bbox_inches="tight", dpi=300)
plt.show()

# Percentage of observations with low utilization
percentage_below_50 = (df["GPU utilization (%)"] < 50).mean() * 100
print(f"Percentage of observations with GPU utilization below 50%: {percentage_below_50:.2f}%")

# Percentage of observations with high utilization
percentage_high = (df["GPU utilization (%)"] >  95).mean() * 100
print(f"Percentage of observations with GPU utilization above 95%: {percentage_high:.2f}%")

