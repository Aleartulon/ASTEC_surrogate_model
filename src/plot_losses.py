from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  CONFIGURE THIS
# ─────────────────────────────────────────────
LOSSES_DIR = "../../../../../scratch/ROM_datasets_ale/ASTEC/saved_logs/SBO/AE_NODE/Models/6_latent_1.0_AR_WORKING/losses"   # <── change this
MODEL_NAME  = "My Model"                     # <── used in plot titles

# Group losses by theme so each group gets its own figure
LOSS_GROUPS = {
    "Reconstruction": [
        "train_full_reconstruction_actual_per_shape",
        "train_full_reconstruction_per_shape",
        "train_full_reconstruction_scalar",
    ],
    "L1": [
        "train_l1_latent",
        "train_l1_per_shape",
        "train_l1",
        "valid_l1_latent",
        "valid_l1_per_shape",
        "valid_l1_unnorm_per_variable",
        "valid_l1_unnorm",
        "valid_l1",
    ],
    "L2 & L3": [
        "train_l2_AR_latent",
        "train_l2_TF",
        "train_l3",
        "valid_l2_AR_latent",
        "valid_l2_TF",
        "valid_l3",
    ],
    "Total & Regularization": [
        "train_loss_tot",
        "train_regularization",
        "valid_loss_tot",
        "valid_regularization",
    ],
    "Real": [
        "valid_real_per_variable",
        "valid_real",
    ],
}


def trim_trailing_plateau(arr):
    """Trim trailing flat-zero plateau from a 1D array."""
    diffs = np.diff(arr)
    nonflat = np.nonzero(diffs)[0]
    if len(nonflat) == 0:
        return None if np.all(arr == 0) else arr
    return arr[:nonflat[-1] + 2]


def load_loss(losses_dir, name):
    """Load a .npy loss file. Returns a dict with 'mean' and optionally 'std'
    for per-shape (2D) arrays, or just a 1D array for scalar losses."""
    path = os.path.join(losses_dir, f"{name}.npy")
    if not os.path.isfile(path):
        return None
    data = np.load(path)

    if data.ndim == 1:
        # Scalar loss per epoch
        trimmed = trim_trailing_plateau(data)
        if trimmed is None:
            return None
        return {"mean": trimmed, "std": None}

    elif data.ndim == 2:
        # Per-shape loss: shape (epochs, num_shapes) or (num_shapes, epochs)
        # Assume longer axis is epochs
        if data.shape[0] < data.shape[1]:
            data = data.T  # now (epochs, num_shapes)
        mean = data.mean(axis=1)
        std  = data.std(axis=1)
        mean = trim_trailing_plateau(mean)
        if mean is None:
            return None
        std = std[:len(mean)]
        return {"mean": mean, "std": std}

    else:
        return None


def plot_group(group_name: str, loss_names: list[str], losses_dir: str, model_name: str):
    # Load available losses
    available = {n: load_loss(losses_dir, n) for n in loss_names}
    available = {n: v for n, v in available.items() if v is not None}

    if not available:
        print(f"  [skip] No files found for group '{group_name}'")
        return

    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"{model_name}  —  {group_name}", fontsize=15, fontweight="bold", y=1.02)

    for idx, (name, entry) in enumerate(available.items()):
        ax = axes[idx // cols][idx % cols]
        mean = entry["mean"]
        std  = entry["std"]
        epochs = np.arange(1, len(mean) + 1)

        # Colour: blue for train, orange for valid
        color = "#2196F3" if name.startswith("train") else "#FF9800"
        lbl = ("train" if name.startswith("train") else "valid") + (" (mean±std)" if std is not None else "")

        ax.plot(epochs, mean, color=color, linewidth=1.5, label=lbl)
        if std is not None:
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Log scale if range is large
        if mean.max() > 0 and mean.max() / (mean.min() + 1e-12) > 100:
            ax.set_yscale("log")
            ax.set_title(name.replace("_", " ") + "  [log]", fontsize=9)

    # Hide unused axes
    for idx in range(len(available), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    safe_name = group_name.replace(" ", "_").replace("&", "and")
    out_path = os.path.join(losses_dir, f"plot_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()


def plot_train_valid_overlay(losses_dir: str, model_name: str):
    """For every loss that exists in both train_ and valid_ variants, overlay them."""
    all_files = {
        f.replace(".npy", "")
        for f in os.listdir(losses_dir)
        if f.endswith(".npy")
    }

    train_keys = {k.removeprefix("train_") for k in all_files if k.startswith("train_")}
    valid_keys = {k.removeprefix("valid_") for k in all_files if k.startswith("valid_")}
    shared = sorted(train_keys & valid_keys)

    if not shared:
        print("  [skip] No matching train/valid pairs found.")
        return

    cols = min(3, len(shared))
    rows = (len(shared) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"{model_name}  —  Train vs Valid (overlaid)", fontsize=15, fontweight="bold", y=1.02)

    for idx, key in enumerate(shared):
        ax = axes[idx // cols][idx % cols]
        t = load_loss(losses_dir, f"train_{key}")
        v = load_loss(losses_dir, f"valid_{key}")

        for entry, color, lbl in [(t, "#2196F3", "train"), (v, "#FF9800", "valid")]:
            if entry is None:
                continue
            mean, std = entry["mean"], entry["std"]
            epochs = np.arange(1, len(mean) + 1)
            label = lbl + (" (mean±std)" if std is not None else "")
            ax.plot(epochs, mean, color=color, linewidth=1.5, label=label)
            if std is not None:
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_title(key.replace("_", " "), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(len(shared), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(losses_dir, "plot_train_vs_valid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()


def main():
    print(f"Loading losses from: {LOSSES_DIR}\n")

    if not os.path.isdir(LOSSES_DIR):
        raise FileNotFoundError(f"Directory not found: {LOSSES_DIR}")

    for group_name, loss_names in LOSS_GROUPS.items():
        print(f"Plotting group: {group_name}")
        plot_group(group_name, loss_names, LOSSES_DIR, MODEL_NAME)

    print("\nPlotting train vs valid overlays...")
    plot_train_valid_overlay(LOSSES_DIR, MODEL_NAME)

    print("\nDone.")


if __name__ == "__main__":
    main()