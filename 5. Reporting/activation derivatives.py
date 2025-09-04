# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 21:48:24 2025

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

# Define derivatives
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def swish_derivative(x):
    s = sigmoid(x)
    return swish(x) + s * (1 - swish(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

# Input range
x_full = np.linspace(-12, 12, 1000)

# Compute derivative values
y_sigmoid_deriv = sigmoid_derivative(x_full)
y_swish_deriv = swish_derivative(x_full)
y_relu_deriv = relu_derivative(x_full)
y_tanh_deriv = tanh_derivative(x_full)

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
        line, = ax.plot(x_full, y_sigmoid_deriv, color="seagreen")
        ax.set_title("Sigmoid'")
        ax.legend([line], [r"$\sigma^{\prime}(x) = \sigma(x)(1 - \sigma(x))$"], loc="upper left")
    elif name == "swish":
        line, = ax.plot(x_full, y_swish_deriv, color="dodgerblue")
        ax.set_title(r"$Swish_{\beta=1}$'")
        ax.legend([line], [r"$\mathrm{Swish}^{\prime}(x) = \mathrm{Swish}(x) + \sigma(x)(1 - \mathrm{Swish}(x))$"], loc="upper left")
    elif name == "ReLU":
        line, = ax.plot(x_full, y_relu_deriv, color="orangered")
        ax.set_title("ReLU'")
        ax.legend([line], [r"$\mathrm{ReLU}^{\prime}(x) = 1$ if $x > 0$, else $0$"], loc="upper left")
    elif name == "tanh":
        line, = ax.plot(x_full, y_tanh_deriv, color="purple")
        ax.set_title("Tanh'")
        ax.legend([line], [r"$\tanh^{\prime}(x) = 1 - \tanh^2(x)$"], loc="upper left")

plt.tight_layout()
plt.savefig("activations_derivatives.pdf", format="pdf", bbox_inches="tight")
plt.show()