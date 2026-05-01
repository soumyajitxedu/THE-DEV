import numpy as np 
import matplotlib.pyplot as plt 

# Generate sample data with normal distribution
np.random.seed(42)  # For reproducibility
scores = np.random.normal(loc=80, scale=10, size=1000)
scores = np.clip(scores, 0, 100)  # Clip to [0, 100] range

# Create histogram with custom styling
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(scores, color="green", alpha=0.8, bins=100, edgecolor="black", linewidth=1.5)

# Customize axes labels and title
ax.set_xlabel("exam scores", size=15, color="red", fontweight="bold")
ax.set_ylabel("number of students", size=15, color="red", fontweight="bold")
ax.set_title("Score Distribution Analysis", size=22, color="blue", fontweight="bold")

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')

# Add statistics text box
mean_score = np.mean(scores)
std_score = np.std(scores)
min_score = np.min(scores)
max_score = np.max(scores)

stats_text = f'Mean: {mean_score:.2f}\nStd Dev: {std_score:.2f}\nMin: {min_score:.2f}\nMax: {max_score:.2f}\nTotal: {len(scores)}'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()