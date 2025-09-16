# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 21:06:27 2025

@author: Julius de Clercq
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for frames
output_dir = 'gradient_descent_frames'
os.makedirs(output_dir, exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate data
np.random.seed(42)
X = np.random.uniform(-3, 3, 40)
X = np.column_stack([np.ones(len(X)), X])
true_theta = np.array([1, 2])
z_true = X @ true_theta
true_probs = sigmoid(z_true)
y = np.random.binomial(1, true_probs)

# Function to compute loss for any theta values
def compute_loss(theta0, theta1):
    z = X @ np.array([theta0, theta1])
    probs = sigmoid(z)
    eps = 1e-15  # small constant to avoid log(0)
    return -np.mean(y * np.log(probs + eps) + (1-y) * np.log(1 - probs + eps))

# Create grid of theta values
theta0_range = np.linspace(-2, 4, 100)
theta1_range = np.linspace(-2, 4, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)
loss_grid = np.zeros_like(theta0_grid)

# Compute loss for each point in the grid
for i in range(len(theta0_range)):
    for j in range(len(theta1_range)):
        loss_grid[i,j] = compute_loss(theta0_grid[i,j], theta1_grid[i,j])

# Compute gradient descent path
theta_start = np.array([-1.5, -1.5])  # New starting point
learning_rate = 0.05
n_steps = 200
theta_path = np.zeros((n_steps, 2))
theta_path[0] = theta_start

for i in range(1, n_steps):
    z = X @ theta_path[i-1]
    probs = sigmoid(z)
    gradient = X.T @ (probs - y)
    theta_path[i] = theta_path[i-1] - learning_rate * gradient

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set limits and labels
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Add labels (now in black for white background)
ax.set_xlabel(r'$\theta_1$', color='black', fontsize=14)
ax.set_ylabel(r'$\theta_2$', color='black', fontsize=14)

# Create contour plot with darker blue colors and black labels
contour = ax.contour(theta0_grid, theta1_grid, loss_grid, levels=20, colors='#0066CC', alpha=0.6, linewidths=1.2)
# Add contour labels with black color for visibility
plt.clabel(contour, inline=True, fmt=r'$\mathcal{L}(\theta)=%.1f$', colors='black', fontsize=8)

# Initialize plots with colors suitable for white background
point = ax.scatter([], [], color='#FF4500', s=120, zorder=3, edgecolors='black', linewidth=1.5)  # Orange-red point with black edge
path_line, = ax.plot([], [], color='#2E8B57', alpha=0.8, zorder=2, linewidth=2.5)  # Sea green path
gradient_arrow = ax.quiver([], [], [], [], color='#DC143C', scale=20, width=0.008, zorder=3)  # Crimson arrow

# Add loss function text (now in black for white background)
loss_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='black',
                   verticalalignment='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Function to update frame
def update_frame(frame):
    fps = 30
    
    # Phase 1: Show contour plot (5 seconds)
    if frame < 5 * fps:
        point.set_offsets(np.array([theta_start]))
        loss_val = compute_loss(theta_start[0], theta_start[1])
        loss_text.set_text(r'$\mathcal{L}(\theta) = $' + f'{loss_val:.2f}')
        gradient_arrow.set_UVC(0, 0)
    
    # Phase 2: Gradient descent animation (15 seconds)
    elif frame < 20 * fps:
        # Calculate which step we're on and interpolation progress
        total_descent_frames = 15 * fps
        frames_per_step = 6  # 6 frames per step = 0.2 seconds per step at 30fps
        progress_frame = frame - 5 * fps
        
        step_idx = int(progress_frame / frames_per_step)
        step_progress = (progress_frame % frames_per_step) / frames_per_step
        
        if step_idx >= n_steps - 1:
            current_theta = theta_path[-1]
        else:
            # Interpolate between steps
            current_theta = theta_path[step_idx] * (1 - step_progress) + theta_path[step_idx + 1] * step_progress
        
        # Update point position
        point.set_offsets(np.array([current_theta]))
        
        # Update path line
        path_line.set_data(theta_path[:step_idx+1, 0], theta_path[:step_idx+1, 1])
        if step_idx < n_steps - 1:
            # Add interpolated point to path
            path_x = np.append(theta_path[:step_idx+1, 0], current_theta[0])
            path_y = np.append(theta_path[:step_idx+1, 1], current_theta[1])
            path_line.set_data(path_x, path_y)
        
        # Update loss text
        loss_val = compute_loss(current_theta[0], current_theta[1])
        loss_text.set_text(r'$\mathcal{L}(\theta) = $' + f'{loss_val:.2f}')
        
        # Show gradient arrow
        if step_idx < n_steps - 1:
            z = X @ current_theta
            probs = sigmoid(z)
            gradient = X.T @ (probs - y)
            gradient_arrow.set_UVC(-gradient[0], -gradient[1])
            gradient_arrow.set_offsets(np.array([current_theta]))
        else:
            gradient_arrow.set_UVC(0, 0)
    
    # Phase 3: Static display (25 seconds)
    else:
        final_theta = theta_path[-1]
        point.set_offsets(np.array([final_theta]))
        path_line.set_data(theta_path[:, 0], theta_path[:, 1])
        loss_val = compute_loss(final_theta[0], final_theta[1])
        loss_text.set_text(r'$\mathcal{L}(\theta) = $' + f'{loss_val:.2f}')
        gradient_arrow.set_UVC(0, 0)

# Generate and save frames
total_frames = 45 * 30  # 45 seconds total at 30 fps

print(f"Generating {total_frames} frames...")
for frame in range(total_frames):
    # Update the frame
    update_frame(frame)
    
    # Save the frame with white background
    filename = os.path.join(output_dir, f'frame_{frame:05d}.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
    
    # Print progress every 100 frames
    if frame % 100 == 0:
        print(f"Saved frame {frame}/{total_frames}")

print(f"All frames saved in '{output_dir}' directory!")
plt.close()

# Optional: Create the GIF from saved frames using PIL
# You can uncomment this if you still want the GIF as well
"""
from PIL import Image
import glob

# Get all frame files
frame_files = sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))

# Load images
frames = [Image.open(f) for f in frame_files]

# Save as GIF
if frames:
    frames[0].save('anim3_gradient_descent.gif', 
                   save_all=True, 
                   append_images=frames[1:], 
                   duration=33.33, 
                   loop=0)
    print("GIF also created!")
"""