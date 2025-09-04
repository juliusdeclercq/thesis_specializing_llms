# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 17:32:58 2025

@author: Julius de Clercq
"""

import numpy as np
import matplotlib.pyplot as plt

# Define activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Input range
x_full = np.linspace(-12, 12, 1000)

# Compute values
y_sigmoid = sigmoid(x_full)
y_swish = swish(x_full)
y_relu = relu(x_full)
y_tanh = tanh(x_full)

# List determines subplot order
plot_positions = ["tanh", "sigmoid", "swish", "ReLU"]

# Create figure with 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Common axis limits
xlim = (-4, 4)
ylim = (-2, 3)

# Loop over axes and plot accordingly
for ax, name in zip(axes, plot_positions):
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if name == "sigmoid":
        line, = ax.plot(x_full, y_sigmoid, color="seagreen")
        ax.set_title("Sigmoid")
        ax.legend([line], [r"$\sigma(x) = \frac{1}{1 + e^{-x}}$"], loc="upper left")
    elif name == "swish":
        line, = ax.plot(x_full, y_swish, color="dodgerblue")
        ax.set_title(r"$Swish_{\beta=1}$ (SiLU)")
        ax.legend([line], [r"$\mathrm{Swish_\beta}(x) = \frac{x}{1 + e^{-x\beta}}$"], loc="upper left")
    elif name == "ReLU":
        line, = ax.plot(x_full, y_relu, color="orangered")
        ax.set_title("ReLU")
        ax.legend([line], [r"$\mathrm{ReLU}(x) = \max(0, x)$"], loc="upper left")
    elif name == "tanh":
        line, = ax.plot(x_full, y_tanh, color="purple")
        ax.set_title("Tanh")
        ax.legend([line], [r"$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$"], loc="upper left")

plt.tight_layout()
plt.savefig("activations.pdf", format="pdf", bbox_inches="tight")
plt.show()

