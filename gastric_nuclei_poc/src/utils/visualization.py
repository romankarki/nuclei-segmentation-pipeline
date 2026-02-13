"""
Visualization utilities for nuclei segmentation pipeline.

Provides functions for side-by-side comparisons, overlay visualizations,
metric bar charts, and violin plots similar to those in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns


def show_image_and_mask(image, mask, title="", figsize=(12, 5)):
    """Display an image and its corresponding mask side by side."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image)
    axes[0].set_title("H&E Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def show_segmentation_overlay(image, gt_mask, pred_mask, title="", figsize=(15, 5)):
    """
    Show segmentation results as overlay (paper-style).

    Colors:
    - White: True Positives (matching areas)
    - Purple: False Negatives (missed nuclei)
    - Green: False Positives (falsely detected)
    """
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)

    tp = gt_binary & pred_binary
    fn = gt_binary & (~pred_binary.astype(bool)).astype(np.uint8)
    fp = (~gt_binary.astype(bool)).astype(np.uint8) & pred_binary

    # Create overlay image
    overlay = image.copy().astype(np.float32) / 255.0
    # Darken the base image
    overlay = overlay * 0.4

    # True Positives - White
    overlay[tp == 1] = [1.0, 1.0, 1.0]
    # False Negatives - Purple
    overlay[fn == 1] = [0.6, 0.0, 0.8]
    # False Positives - Green
    overlay[fp == 1] = [0.0, 0.8, 0.0]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt_binary, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Segmentation Overlay")
    axes[2].axis("off")

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="True Positive"),
        mpatches.Patch(facecolor=(0.6, 0, 0.8), label="False Negative"),
        mpatches.Patch(facecolor=(0, 0.8, 0), label="False Positive"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def compare_normalizations(original, normalized_dict, figsize=None):
    """
    Show original image alongside multiple normalized versions.

    Parameters
    ----------
    original : np.ndarray
        Original H&E image
    normalized_dict : dict
        {method_name: normalized_image}
    """
    n = len(normalized_dict) + 1
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for idx, (name, img) in enumerate(normalized_dict.items(), 1):
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=11)
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def compare_deconvolutions(original, deconv_dict, figsize=None):
    """
    Show original image alongside deconvolved stain channels.

    Parameters
    ----------
    original : np.ndarray
        Original H&E image
    deconv_dict : dict
        {method_name: {'hematoxylin': array, 'eosin': array}}
    """
    n_methods = len(deconv_dict)
    fig, axes = plt.subplots(n_methods + 1, 3, figsize=(12, 4 * (n_methods + 1)))

    if n_methods == 0:
        return fig

    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original H&E")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for idx, (name, channels) in enumerate(deconv_dict.items(), 1):
        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f"{name}\n(Original)")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(channels["hematoxylin"], cmap="gray")
        axes[idx, 1].set_title(f"{name}\n(Hematoxylin)")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(channels["eosin"], cmap="gray")
        axes[idx, 2].set_title(f"{name}\n(Eosin)")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    return fig


def compare_segmentation_methods(image, gt_mask, results_dict, figsize=None):
    """
    Compare multiple segmentation methods side-by-side with overlays.

    Parameters
    ----------
    image : np.ndarray
        Original image
    gt_mask : np.ndarray
        Ground truth mask
    results_dict : dict
        {method_name: predicted_mask}
    """
    n = len(results_dict)
    if figsize is None:
        figsize = (5 * (n + 1), 5)

    fig, axes = plt.subplots(1, n + 1, figsize=figsize)

    # Original + GT overlay
    gt_overlay = image.copy().astype(np.float32) / 255.0
    gt_binary = (gt_mask > 0)
    gt_overlay[gt_binary] = [1.0, 1.0, 1.0]
    axes[0].imshow(np.clip(gt_overlay, 0, 1))
    axes[0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for idx, (name, pred_mask) in enumerate(results_dict.items(), 1):
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)

        tp = gt_binary & pred_binary
        fn = gt_binary & (~pred_binary.astype(bool)).astype(np.uint8)
        fp = (~gt_binary.astype(bool)).astype(np.uint8) & pred_binary

        overlay = image.copy().astype(np.float32) / 255.0 * 0.4
        overlay[tp == 1] = [1.0, 1.0, 1.0]
        overlay[fn == 1] = [0.6, 0.0, 0.8]
        overlay[fp == 1] = [0.0, 0.8, 0.0]

        axes[idx].imshow(np.clip(overlay, 0, 1))
        axes[idx].set_title(name, fontsize=11)
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict, metric_names=None, figsize=(12, 6)):
    """
    Bar chart comparing metrics across methods.

    Parameters
    ----------
    metrics_dict : dict
        {method_name: {metric_name: value}}
    metric_names : list of str or None
        Which metrics to plot. If None, plot all.
    """
    if metric_names is None:
        metric_names = list(next(iter(metrics_dict.values())).keys())

    methods = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", n_metrics)

    for i, metric in enumerate(metric_names):
        values = [metrics_dict[m].get(metric, 0) for m in methods]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Segmentation Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_violin_metrics(scores_dict, metric_name="F1", figsize=(10, 6)):
    """
    Violin plot for a metric across methods (paper-style Fig. 6 & 8).

    Parameters
    ----------
    scores_dict : dict
        {method_name: list of per-image scores}
    metric_name : str
        Name of the metric for labeling
    """
    import pandas as pd

    data = []
    for method, scores in scores_dict.items():
        for s in scores:
            data.append({"Method": method, metric_name: s})
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x="Method", y=metric_name, ax=ax, inner="box", cut=0)
    ax.set_title(
        f"{metric_name} Distribution by Method", fontsize=14, fontweight="bold"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def show_blur_detection(image, blur_map, clean_image, figsize=(15, 5)):
    """Show blur detection results: original, blur map, and cleaned image."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(blur_map, cmap="hot")
    axes[1].set_title("Blur Detection Map")
    axes[1].axis("off")

    axes[2].imshow(clean_image)
    axes[2].set_title("After Blur Removal")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def show_pipeline_steps(steps_dict, figsize=None):
    """
    Show each step of the processing pipeline.

    Parameters
    ----------
    steps_dict : dict
        OrderedDict of {step_name: image_array}
    """
    n = len(steps_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (4 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, img) in enumerate(steps_dict.items()):
        r, c = divmod(idx, cols)
        if img.ndim == 2:
            axes[r, c].imshow(img, cmap="gray")
        else:
            axes[r, c].imshow(img)
        axes[r, c].set_title(name, fontsize=10)
        axes[r, c].axis("off")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    return fig
