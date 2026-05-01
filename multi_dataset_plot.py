import matplotlib.pyplot as plt
import numpy as np 

# Advanced plotting with multiple datasets and custom styling
x = np.array([1,2,3,48,5,6,7])
y = np.array([12,37,112,3,4,5,6])
y1 = np.array([20, 25, 30, 45, 50, 55, 60])  # steadily increasing values

# Main plot configuration
plt.figure(figsize=(12, 6))

# Title with custom formatting
plt.title("Data Visualization Example", fontsize=16,
          family="arial",
          fontweight="bold",
          color="#256ac5")

# First plot with explicit styling
plt.plot(x, y, marker=".",
         markersize=10,
         markerfacecolor="#51aa5d",
         markeredgecolor="#000000",
         linestyle="solid",
         linewidth=4,
         color="#841FB3",
         label="Dataset 1")

# Create a dictionary for line style to reuse across plots
line_style = dict(marker=".",
                  markersize=10,
                  markerfacecolor="#51aa5d",
                  markeredgecolor="#8B3939",
                  linestyle="solid",
                  linewidth=4,
                  color="#D89090")

# Plot second dataset with style dictionary
plt.plot(x, y, **line_style, label="Dataset 2")

# Plot third dataset with same style
plt.plot(x, y1, **line_style, label="Dataset 3")

# Add legend, grid, and labels
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlabel("X Axis", fontsize=12)
plt.ylabel("Y Axis", fontsize=12)

plt.tight_layout()
plt.show()