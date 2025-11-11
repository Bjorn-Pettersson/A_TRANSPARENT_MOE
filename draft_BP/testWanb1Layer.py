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

    raw, avg_mat, avg_iters, avg_dict = simulate_training_and_average(
        num_experts=num_experts,
        total_iterations=total_iterations,
        nr_avg_pts=nr_avg_pts,
        seed=42,
    )

    # Print shapes and the first averaged vector (window 0 corresponds to iterations 0..49)
    print(f"raw shape: {raw.shape}")
    print(f"avg_matrix shape: {avg_mat.shape}")
    if 0 in avg_dict:
        print(f"First averaged vector (window 0, iters 0..{nr_avg_pts-1}): {avg_dict[0]}")

    # Plot averaged values
    plot_averages(avg_mat, avg_iters, layer_id=9)
