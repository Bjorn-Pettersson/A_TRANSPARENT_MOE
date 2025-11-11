# file: moe_activation_plot.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


def simulate_training_and_average(num_experts: int = 4,
                                  total_iterations: int = 6000,
                                  nr_avg_pts: int = 20,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Simulate a training loop where at each iteration each expert produces one "point".
    Collect `nr_avg_pts` points per window and compute per-expert averages for each window.

    Returns:
        raw_activations: numpy array shape (num_experts, total_iterations)
        avg_matrix: numpy array shape (num_experts, num_windows) with averaged values
        avg_iters: numpy array shape (num_windows,) with iteration midpoints for each window
        averages_dict: dictionary mapping window_index -> averaged vector (length num_experts)
    """
    np.random.seed(seed)
    iterations = np.arange(total_iterations)

    # Create per-expert deterministic trends (so simulation is repeatable)
    trends = []
    for i in range(num_experts):
        trends.append(0.4 * np.exp(-iterations / (2500 + 400 * i)) + 0.05 * i)
    trends = np.array(trends)

    # Raw per-iteration activations (num_experts x total_iterations)
    raw_activations = np.zeros((num_experts, total_iterations))

    # Buffers are implicit via slices, but we maintain a dictionary for the averaged windows
    averages_dict: Dict[int, np.ndarray] = {}

    num_windows = total_iterations // nr_avg_pts
    avg_matrix = np.zeros((num_experts, num_windows))
    avg_iters = np.zeros(num_windows)

    # Simulate iteration-by-iteration and fill raw_activations
    for it in range(total_iterations):
        noise = np.random.normal(0, 0.03, size=num_experts)
        point = np.clip(trends[:, it] + noise, 0.0, 0.5)
        raw_activations[:, it] = point

        # When we complete a window, compute averages for that window
        if (it + 1) % nr_avg_pts == 0:
            w = (it + 1) // nr_avg_pts - 1
            start = w * nr_avg_pts
            end = start + nr_avg_pts
            window_avg = raw_activations[:, start:end].mean(axis=1)
            avg_matrix[:, w] = window_avg
            avg_iters[w] = (start + end - 1) / 2.0  # midpoint iteration index
            averages_dict[w] = window_avg.copy()

    return raw_activations, avg_matrix, avg_iters, averages_dict


def plot_averages(avg_matrix: np.ndarray, avg_iters: np.ndarray, layer_id: int = 9) -> None:
    num_experts = avg_matrix.shape[0]
    plt.figure(figsize=(5, 3))
    for i in range(num_experts):
        plt.scatter(avg_iters, avg_matrix[i], s=12, alpha=0.8, label=f'Expert {i}')

    plt.title(f"layer id {layer_id}")
    plt.xlabel("Iterations")
    plt.ylabel("Average % activation per expert")
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with nr_avg_pts=50 so at iteration 50 we get the first averaged vector
    num_experts = 4
    total_iterations = 6000
    nr_avg_pts = 500
    # --- New: simulate multiple layers and plot them in a grid ---
    def simulate_layers(num_layers: int = 12,
                        num_experts: int = 4,
                        total_iterations: int = 6000,
                        nr_avg_pts: int = 500,
                        base_seed: int = 42) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Simulate `num_layers` independent MoE layers. For variety each layer uses a different seed
        (base_seed + layer_idx). Returns a list of avg_matrices (one per layer) and the avg_iters
        (shared if nr_avg_pts and total_iterations are same for all layers).
        """
        avg_matrices = []
        avg_iters = None
        for layer_idx in range(num_layers):
            _, avg_mat, avg_it, _ = simulate_training_and_average(
                num_experts=num_experts,
                total_iterations=total_iterations,
                nr_avg_pts=nr_avg_pts,
                seed=base_seed + layer_idx,
            )
            avg_matrices.append(avg_mat)
            if avg_iters is None:
                avg_iters = avg_it
        return avg_matrices, avg_iters


    def plot_multi_layer_averages(avg_matrices: List[np.ndarray], avg_iters: np.ndarray, ncols: int = 4) -> None:
        """
        Plot a grid of subplots for each layer's averaged activations.
        Each subplot shows one layer (num_experts x num_windows scatter).
        """
        num_layers = len(avg_matrices)
        nrows = (num_layers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for idx, avg_mat in enumerate(avg_matrices):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            num_experts = avg_mat.shape[0]
            for e in range(num_experts):
                ax.scatter(avg_iters, avg_mat[e], s=10, alpha=0.8, label=f'E{e}' if idx == 0 else None)
            ax.set_title(f'Layer {idx}')
            ax.set_ylim(0, 0.5)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Avg activation')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=7)

        # turn off any unused axes
        for j in range(num_layers, nrows * ncols):
            r = j // ncols
            c = j % ncols
            axes[r][c].axis('off')

        plt.tight_layout()
        plt.show()


    # Parameters for multi-layer simulation
    num_layers = 12
    num_experts = 4
    total_iterations = 6000
    nr_avg_pts = 500

    avg_matrices, avg_iters = simulate_layers(
        num_layers=num_layers,
        num_experts=num_experts,
        total_iterations=total_iterations,
        nr_avg_pts=nr_avg_pts,
        base_seed=42,
    )

    print(f"Simulated {len(avg_matrices)} layers")
    print(f"Each avg_matrix shape (num_experts x num_windows): {avg_matrices[0].shape}")
    # Print first averaged vector for layer 0 window 0
    print(f"Layer 0, window 0 averaged vector: {avg_matrices[0][:,0]}")

    plot_multi_layer_averages(avg_matrices, avg_iters, ncols=4)
