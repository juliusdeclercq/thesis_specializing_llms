# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 17:11:12 2025

@author: Julius de Clercq
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Settings
num_positions = 1000
embedding_dim = 1000

# Generate sinusoidal embeddings
def sinusoidal_embeddings(positions, d):
    emb = np.zeros((len(positions), d))
    for i in range(d // 2):
        theta = 10000 ** (-2 * i / d)
        emb[:, 2 * i] = np.sin(positions * theta)
        emb[:, 2 * i + 1] = np.cos(positions * theta)
    return emb

positions = np.arange(num_positions)
sin_emb = sinusoidal_embeddings(positions, embedding_dim)

# Compute the difference between consecutive positions (position p - position p-1)
# Shape: (num_positions-1, embedding_dim)
diff = np.diff(sin_emb, axis=0)

# Create a heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(
    diff.T,  # Transpose so that embeddings are vertical (columns are positions)
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Change in Embedding Value"},
    # Show every 100th position and dimension label
    xticklabels=int(num_positions/10),  
    yticklabels=int(embedding_dim/10)
)
# plt.title("Change in Sinusoidal Embedding Across Positions")
plt.xlabel("Position Index")
plt.ylabel("Embedding Dimension")
plt.tight_layout()
plt.savefig("pos_embs_sinusoidal.pdf", bbox_inches="tight")
plt.show()

