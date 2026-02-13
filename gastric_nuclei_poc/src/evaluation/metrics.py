"""
Evaluation metrics for nuclei segmentation.

Implements:
1. F1-measure (precision & recall based)
2. Dice Coefficient (Eq. 11 in paper)
3. Aggregated Jaccard Index - AJI (Eq. 12 in paper)
4. Basic pixel-level accuracy
5. IoU (Intersection over Union / Jaccard Index)

Reference: Paper Section 3.8
"""

import numpy as np
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment


def pixel_accuracy(gt_mask, pred_mask):
    """
    Basic pixel-level accuracy.

    Parameters
    ----------
    gt_mask : np.ndarray
        Ground truth binary mask
    pred_mask : np.ndarray
        Predicted binary mask

    Returns
    -------
    accuracy : float
        Percentage of correctly classified pixels
    """
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    correct = np.sum(gt_binary == pred_binary)
    total = gt_binary.size
    return (correct / total) * 100


def precision_recall_f1(gt_mask, pred_mask):
    """
    Compute pixel-level precision, recall, and F1-measure.

    Parameters
    ----------
    gt_mask : np.ndarray
        Ground truth binary mask
    pred_mask : np.ndarray
        Predicted binary mask

    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """
    gt_binary = (gt_mask > 0).astype(bool)
    pred_binary = (pred_mask > 0).astype(bool)

    tp = np.sum(gt_binary & pred_binary)
    fp = np.sum(~gt_binary & pred_binary)
    fn = np.sum(gt_binary & ~pred_binary)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def dice_coefficient(gt_mask, pred_mask):
    """
    Dice Coefficient (Eq. 11 in paper).

    DC = 2 * |G ∩ S| / (|G| + |S|)

    Parameters
    ----------
    gt_mask : np.ndarray
        Ground truth binary mask
    pred_mask : np.ndarray
        Predicted binary mask

    Returns
    -------
    dice : float
    """
    gt_binary = (gt_mask > 0).astype(bool)
    pred_binary = (pred_mask > 0).astype(bool)

    intersection = np.sum(gt_binary & pred_binary)
    gt_sum = np.sum(gt_binary)
    pred_sum = np.sum(pred_binary)

    if (gt_sum + pred_sum) == 0:
        return 1.0  # Both empty = perfect match

    return 2.0 * intersection / (gt_sum + pred_sum)


def iou_score(gt_mask, pred_mask):
    """
    Intersection over Union (Jaccard Index).

    Parameters
    ----------
    gt_mask : np.ndarray
    pred_mask : np.ndarray

    Returns
    -------
    iou : float
    """
    gt_binary = (gt_mask > 0).astype(bool)
    pred_binary = (pred_mask > 0).astype(bool)

    intersection = np.sum(gt_binary & pred_binary)
    union = np.sum(gt_binary | pred_binary)

    if union == 0:
        return 1.0

    return intersection / union


def aggregated_jaccard_index(gt_mask, pred_mask):
    """
    Aggregated Jaccard Index (AJI) - Eq. 12 in paper.

    Object-level metric that penalizes both over- and under-segmentation.

    AJI = sum(|Gi ∩ Sk|) / (sum(|Gi ∪ Sk|) + sum(|Sk_rest|))

    Where Sk is the predicted nucleus maximizing Jaccard with GT nucleus Gi,
    and Sk_rest are falsely predicted nuclei with no GT match.

    Parameters
    ----------
    gt_mask : np.ndarray
        Ground truth mask (labeled or binary)
    pred_mask : np.ndarray
        Predicted mask (labeled or binary)

    Returns
    -------
    aji : float
    """
    # Ensure labeled
    gt_labeled = label(gt_mask > 0)
    pred_labeled = label(pred_mask > 0)

    gt_ids = np.unique(gt_labeled)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_labeled)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    # For each GT nucleus, find the best matching predicted nucleus
    used_pred = set()
    total_intersection = 0
    total_union = 0

    for gt_id in gt_ids:
        gt_region = (gt_labeled == gt_id)
        best_iou = 0
        best_pred_id = None
        best_intersection = 0
        best_union = 0

        for pred_id in pred_ids:
            pred_region = (pred_labeled == pred_id)
            intersection = np.sum(gt_region & pred_region)

            if intersection == 0:
                continue

            union = np.sum(gt_region | pred_region)
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id
                best_intersection = intersection
                best_union = union

        if best_pred_id is not None:
            total_intersection += best_intersection
            total_union += best_union
            used_pred.add(best_pred_id)
        else:
            # No matching prediction found -> union is just the GT area
            total_union += np.sum(gt_region)

    # Add falsely predicted nuclei (no GT match)
    for pred_id in pred_ids:
        if pred_id not in used_pred:
            total_union += np.sum(pred_labeled == pred_id)

    if total_union == 0:
        return 0.0

    return total_intersection / total_union


def compute_all_metrics(gt_mask, pred_mask):
    """
    Compute all segmentation metrics at once.

    Parameters
    ----------
    gt_mask : np.ndarray
        Ground truth mask
    pred_mask : np.ndarray
        Predicted mask

    Returns
    -------
    metrics : dict
        Dictionary with all metric scores
    """
    precision, recall, f1 = precision_recall_f1(gt_mask, pred_mask)
    dice = dice_coefficient(gt_mask, pred_mask)
    aji = aggregated_jaccard_index(gt_mask, pred_mask)
    accuracy = pixel_accuracy(gt_mask, pred_mask)
    iou = iou_score(gt_mask, pred_mask)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Dice": dice,
        "AJI": aji,
        "Accuracy": accuracy,
        "IoU": iou,
    }


def compute_dataset_metrics(gt_masks, pred_masks, names=None):
    """
    Compute metrics for an entire dataset.

    Parameters
    ----------
    gt_masks : list of np.ndarray
    pred_masks : list of np.ndarray
    names : list of str or None

    Returns
    -------
    per_image : list of dict
        Per-image metrics
    summary : dict
        Mean and std for each metric
    """
    per_image = []
    for i, (gt, pred) in enumerate(zip(gt_masks, pred_masks)):
        metrics = compute_all_metrics(gt, pred)
        if names is not None:
            metrics["name"] = names[i]
        per_image.append(metrics)

    # Compute summary statistics
    metric_names = ["Precision", "Recall", "F1", "Dice", "AJI", "Accuracy", "IoU"]
    summary = {}
    for metric in metric_names:
        values = [m[metric] for m in per_image]
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    return per_image, summary


def print_metrics_summary(summary, title="Metrics Summary"):
    """Pretty-print metrics summary."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<12} {'Mean':>8} {'± STD':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-' * 60}")
    for metric, stats in summary.items():
        fmt = ".3f" if metric != "Accuracy" else ".1f"
        print(
            f"{metric:<12} {stats['mean']:>8{fmt}} ± {stats['std']:>6{fmt}} "
            f"{stats['min']:>8{fmt}} {stats['max']:>8{fmt}}"
        )
    print(f"{'=' * 60}")
