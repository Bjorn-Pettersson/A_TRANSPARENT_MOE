import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Dummy data: 4 experts × 6 subjects (routing percentages)
# Each row = expert id, each column = subject
# ---------------------------------------------------------
data = np.array([
    [0.25, 0.50, 0.20, 0.45, 0.50, 0.48],   # Expert 0
    [0.50, 0.45, 0.55, 0.48, 0.52, 0.47],   # Expert 1
    [0.22, 0.00, 0.33, 0.00, 0.00, 0.00],   # Expert 2
    [0.00, 0.05, 0.00, 0.00, 0.00, 0.00],   # Expert 3
])

experts = np.arange(4)

subjects = [
    "global_facts",
    "abstract_algebra",
    "medical_genetics",
    "management",
    "college_biology",
    "college_chemistry",
]

colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#c7c7c7",  # light grey
    "#7f7f7f",  # dark grey
    "#aec7e8",  # light blue
    "#bc621c",  # brownish-orange
]

# width of each small bar
bar_width = 0.12

fig, ax = plt.subplots(figsize=(8, 2.6))

# ---------------------------------------------------------
# Plot grouped bars: one subject per color
# ---------------------------------------------------------
for idx, subject in enumerate(subjects):
    offset = (idx - len(subjects)/2) * bar_width + bar_width/2
    ax.bar(
        experts + offset,
        data[:, idx],
        width=bar_width,
        color=colors[idx],
        label=subject
    )

ax.set_xticks(experts)
ax.set_xlabel("Experts")
ax.set_ylabel("Routing share")
ax.set_title("Layer id: 11")

# Legend below the plot, same as the paper style
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35),
          ncol=3, frameon=False)

plt.tight_layout()
plt.show()
