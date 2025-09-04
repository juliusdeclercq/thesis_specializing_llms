# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 13:20:27 2025

@author: Julius de Clercq
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load training loss CSV
df_train = pd.read_csv("IT0 training loss.csv")
df_train.columns = ["Iteration", "Loss"]

# Convert training iterations to epochs
iters_per_epoch = 415
df_train["Epoch"] = df_train["Iteration"] / iters_per_epoch

# Validation loss data from new eval_loss logs
val_data = [
    (1, 1.1153154373168945),
    (2, 1.103398084640503),
    (3, 1.2747893333435059),
    (4, 1.4346009492874146),
    (5, 1.4689078330993652),
    (6, 1.6047909259796143),
    (7, 1.701279640197754),
    (8, 1.8249861001968384),
    (9, 1.992558479309082),
    (10, 2.1864447593688965),
]

# Convert to DataFrame
val_df = pd.DataFrame(val_data, columns=["Epoch", "Loss"])

# Set style
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))

# Training loss line
sns.lineplot(
    data=df_train,
    x="Epoch",
    y="Loss",
    color="tab:blue",
    linewidth=2,
    label="Training loss",
    ax=ax
)

# Validation loss line (dashed with markers)
sns.lineplot(
    data=val_df,
    x="Epoch",
    y="Loss",
    color="tab:orange",
    linewidth=2,
    linestyle="--",
    marker="o",
    markersize=6,
    label="Validation loss",
    ax=ax
)

# Force x-axis to start at 0
ax.set_xlim(left=0)

# Only horizontal grid lines
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.grid(False, axis='x')

# Remove top/right borders
sns.despine(ax=ax)

# Labels
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

# Legend with solid background in upper right
legend = ax.legend(loc="upper right", frameon=True)
legend.get_frame().set_facecolor('white')

ax.set_title("Vanilla")
plt.tight_layout()
plt.savefig(
    "IT0 loss plot.jpg",
    bbox_inches="tight",
    dpi=300
)
plt.show()
