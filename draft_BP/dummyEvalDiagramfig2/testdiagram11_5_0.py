import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Dummy data for multiple layers
# Shape: layers × experts × subjects
# Here we simulate 3 layers (0, 5, 11), 4 experts, 6 subjects
# -------------------------------------------------------------------
data_by_layer = {
    0: np.array([
        [0.45, 0.48, 0.10, 0.42, 0.50, 0.47],   # Expert 0
        [0.30, 0.25, 0.02, 0.05, 0.18, 0.15],   # Expert 1
        [0.05, 0.05, 0.00, 0.48, 0.10, 0.05],   # Expert 2
        [0.20, 0.22, 0.03, 0.03, 0.22, 0.18],   # Expert 3
    ]),
    5: np.array([
        [0.48, 0.50, 0.45, 0.15, 0.50, 0.42],
        [0.03, 0.08, 0.10, 0.18, 0.20, 0.08],
        [0.02, 0.03, 0.05, 0.12, 0.25, 0.20],
        [0.47, 0.39, 0.40, 0.55, 0.05, 0.30],
    ]),
    11: np.array([
        [0.26, 0.50, 0.15, 0.10, 0.50, 0.48],
        [0.50, 0.45, 0.48, 0.45, 0.47, 0.46],
        [0.22, 0.00, 0.33, 0.00, 0.00, 0.00],
        [0.02, 0.05, 0.00, 0.00, 0.00, 0.00],
    ])
}

# -------------------------------------------------------------------
# Subjects and colors (same ordering as your example)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Choose layers to plot
# -------------------------------------------------------------------
layers_to_plot = [0, 5, 11]

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
num_layers = len(layers_to_plot)
fig, axes = plt.subplots(num_layers, 1, figsize=(8, 6), sharex=True)

if num_layers == 1:
    axes = [axes]  # ensure iterable

bar_width = 0.12
experts = np.arange(4)

for ax, layer in zip(axes, layers_to_plot):
    data = data_by_layer[layer]
    for s_idx, subject in enumerate(subjects):
        offset = (s_idx - len(subjects)/2) * bar_width + bar_width/2

        ax.bar(
            experts + offset,
            data[:, s_idx],
            width=bar_width,
            color=colors[s_idx],
            label=subject if layer == layers_to_plot[0] else None,  # legend once
        )

    ax.set_ylabel(f"Layer id: {layer}")
    ax.set_ylim(0, 0.6)

axes[-1].set_xlabel("Experts")
axes[-1].set_xticks(experts)

# Legend under the whole figure (only one copy)
fig.legend(
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()
