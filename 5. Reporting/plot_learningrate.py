# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 00:24:22 2025

@author: Julius de Clercq
"""

from transformers import get_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Training config
total_training_steps = 1000
warmup_ratio = 0.05
warmup_steps = int(warmup_ratio * total_training_steps)
learning_rate = 2e-4
adam_betas = (0.9, 0.95)
weight_decay = 0.01

# Dummy model params and optimizer config
dummy_params = [torch.nn.Parameter(torch.randn(2, 2))]
optimizer = torch.optim.AdamW(
    dummy_params,
    lr=learning_rate,
    betas=adam_betas,
    weight_decay=weight_decay
)

# Cosine LR scheduler
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)

# Collect LR values for each step
lrs = []
for step in range(total_training_steps):
    lr_scheduler.step()
    lrs.append(lr_scheduler.get_last_lr()[0])

sns.set_theme(style="whitegrid")
sns.set_context("talk")

fig, ax = plt.subplots(figsize=(8, 5))

sns.lineplot(
    x=range(total_training_steps),
    y=lrs,
    color="tab:blue",
    linewidth=2,
    ax=ax
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Learning rate")

ax.set_xticks([0, total_training_steps])
ax.set_xticklabels(["0", "T"])

ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.grid(False, axis="x")

sns.despine(ax=ax)

# ax.set_title("Cosine learning rate schedule", fontsize=16, pad=15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("lr_schedule.jpg", bbox_inches="tight", dpi=300)
plt.show()
