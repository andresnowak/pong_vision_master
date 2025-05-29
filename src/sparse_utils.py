import matplotlib.pyplot as plt
import os

import json
import os
import numpy as np

def save_results(dense_result, sparse_results, save_dir):
    """Save evaluation results to JSON, converting NumPy types to native Python."""
    os.makedirs(save_dir, exist_ok=True)

    def convert_numpy_types(obj):
        """Recursively convert NumPy types to native Python types."""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(x) for x in obj]
        return obj

    # Convert all NumPy types in results
    results = {
        "dense": convert_numpy_types(dense_result),
        "sparse": {
            f"{sparsity}": convert_numpy_types(result)
            for sparsity, result in zip([0.2, 0.4, 0.6, 0.8], sparse_results)
        },
    }

    # Save to JSON
    filename = os.path.join(save_dir, "evaluation_results.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to {filename}")


def plot_results_for_paper(
    dense_result,
    sparse_results,
    save_path="figures",
    file_format="pdf",
    dpi=600,
    show=False,
):
    """
    Publication-quality plot with white background.

    Args:
        dense_result: Results from dense model (dict with 'avg_reward', 'std_reward').
        sparse_results: List of sparse model results.
        save_path: Directory to save figures (default: "figures").
        file_format: File type ('pdf', 'png', 'svg', 'eps').
        dpi: Resolution for raster formats (e.g., 'png').
        show: Whether to display the plot (default: False; saves only).
    """
    # --- Data ---
    sparsity_levels = [0.0] + [0.2, 0.4, 0.6, 0.8]
    avg_rewards = [dense_result["avg_reward"]] + [
        r["avg_reward"] for r in sparse_results
    ]
    std_rewards = [dense_result["std_reward"]] + [
        r["std_reward"] for r in sparse_results
    ]

    # --- Styling ---
    plt.style.use("default")  # Reset to default (white background)
    plt.rcParams.update(
        {
            "font.family": "serif",  # LaTeX-like fonts
            "font.size": 10,  # Base font size
            "axes.facecolor": "white",  # White plot background
            "figure.facecolor": "white",  # White figure background
            "grid.color": "0.9",  # Light gray grid
            "grid.linestyle": "--",  # Dashed grid
            "grid.linewidth": 0.5,  # Thin grid lines
            "axes.grid": True,  # Enable grid
            "axes.axisbelow": True,  # Grid below plot elements
        }
    )

    # --- Figure Setup ---
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Single-column width
    fig.patch.set_facecolor("white")  # Ensure no transparent background

    # --- Plot ---
    ax.errorbar(
        sparsity_levels,
        avg_rewards,
        yerr=std_rewards,
        fmt="o-",  # Line + markers
        color="#1f77b4",  # Matplotlib blue
        markersize=5,  # Slightly larger markers
        capsize=3,  # Error bar cap width
        linewidth=1.2,  # Slightly thicker line
        markeredgecolor="black",  # Black marker outline
        markerfacecolor="white",  # White marker fill
        ecolor="0.4",  # Gray error bars (40% black)
    )

    # --- Labels & Title ---
    ax.set_xlabel("Sparsity Level", fontsize=10, labelpad=4)
    ax.set_ylabel("Average Reward", fontsize=10, labelpad=4)
    ax.set_title("Performance vs. Sparsity", fontsize=11, pad=12)

    # --- Grid & Layout ---
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)
    plt.tight_layout(pad=1.5)  # Slightly more padding

    # --- Save ---
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}/sparsity_performance.{file_format}"
    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),  # Force white background on save
    )
    print(f"✅ Saved to: {filename} (white background)")

    if show:
        plt.show()
    plt.close()






def plot_results(dense_result, sparse_results, save_path=None, file_format="png"):
    # Extract data
    sparsity_levels = [0.0] + [0.2, 0.4, 0.6, 0.8]  # 0.0 represents dense model
    avg_rewards = [dense_result["avg_reward"]] + [
        r["avg_reward"] for r in sparse_results
    ]
    std_rewards = [dense_result["std_reward"]] + [
        r["std_reward"] for r in sparse_results
    ]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot average rewards with error bars
    plt.errorbar(
        sparsity_levels,
        avg_rewards,
        yerr=std_rewards,
        fmt="-o",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
    )

    # Add labels and title
    plt.xlabel("Sparsity Level", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Agent Performance vs Model Sparsity", fontsize=14)

    # Add grid and adjust layout
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/performance_vs_sparsity.{file_format}"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filename}")
    else:
        print("No path error")
        exit()

    # Show plot
    plt.close()