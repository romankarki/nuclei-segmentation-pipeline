"""
Nuclei Segmentation POC - Main Execution Script

Reproduces the key results from:
"Optimized detection and segmentation of nuclei in gastric cancer images
using stain normalization and blurred artifact removal"
(Martos et al., Pathology - Research and Practice, 2023)

Usage:
    python run_poc.py                    # Run with defaults
    python run_poc.py --dataset monuseg  # Run on MoNuSeg only
    python run_poc.py --dataset tnbc     # Run on TNBC only
    python run_poc.py --max-images 5     # Limit images for quick test
    python run_poc.py --save-visuals     # Save visual outputs
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import load_dataset, load_all_datasets
from src.utils.visualization import (
    plot_metrics_comparison,
    plot_violin_metrics,
    show_segmentation_overlay,
    compare_normalizations,
)
from src.preprocessing.stain_normalization import (
    normalize_macenko,
    normalize_clahe,
    normalize_reinhard,
    normalize_vahadane_simple,
)
from src.preprocessing.stain_deconvolution import deconvolve_ruifrok, deconvolve_pca
from src.preprocessing.blur_detection import detect_blur, remove_blur_artifacts
from src.segmentation.otsu_watershed import (
    segment_otsu_watershed,
    segment_adaptive_watershed,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    print_metrics_summary,
)


# ============================================================================
# Pipeline Configurations
# ============================================================================
def run_pipeline(image, method_name):
    """Run a specific pipeline configuration."""

    if method_name == "Baseline (No Norm)":
        seg, _ = segment_otsu_watershed(image)
        return seg

    elif method_name == "CLAHE + Otsu":
        norm, _ = normalize_clahe(image)
        seg, _ = segment_otsu_watershed(norm)
        return seg

    elif method_name == "Macenko + Otsu":
        norm, h_ch, _ = normalize_macenko(image)
        seg, _ = segment_otsu_watershed(h_ch)
        return seg

    elif method_name == "Ruifrok + Otsu":
        result = deconvolve_ruifrok(image)
        seg, _ = segment_otsu_watershed(result["hematoxylin"])
        return seg

    elif method_name == "PCA Deconv + Otsu":
        result = deconvolve_pca(image)
        seg, _ = segment_otsu_watershed(result["hematoxylin"])
        return seg

    elif method_name == "Macenko + Adaptive WS":
        norm, h_ch, _ = normalize_macenko(image)
        seg, _ = segment_adaptive_watershed(norm, h_ch)
        return seg

    elif method_name == "Reinhard + Otsu":
        norm = normalize_reinhard(image)
        seg, _ = segment_otsu_watershed(norm)
        return seg

    else:
        raise ValueError(f"Unknown method: {method_name}")


METHOD_NAMES = [
    "Baseline (No Norm)",
    "CLAHE + Otsu",
    "Reinhard + Otsu",
    "Macenko + Otsu",
    "Ruifrok + Otsu",
    "PCA Deconv + Otsu",
    "Macenko + Adaptive WS",
]


# ============================================================================
# Main Execution
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Nuclei Segmentation POC - Run full evaluation pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["monuseg", "tnbc", "all"],
        help="Which dataset to evaluate on (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="Path to data directory (default: ../data)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images per dataset (for quick testing)",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save visual outputs to data/results/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Directory for output files (default: experiments)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Nuclei Segmentation POC")
    print("  Based on: Martos et al. (2023)")
    print("  'Optimized detection and segmentation of nuclei in gastric")
    print("   cancer images using stain normalization and blurred")
    print("   artifact removal'")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_visuals:
        os.makedirs("data/results", exist_ok=True)

    # Load datasets
    print(f"\nLoading data from: {args.data_root}")
    if args.dataset == "all":
        dataset_names = ["monuseg", "tnbc"]
    else:
        dataset_names = [args.dataset]

    datasets = {}
    for ds_name in dataset_names:
        try:
            images, masks, names = load_dataset(
                args.data_root, ds_name, args.max_images
            )
            datasets[ds_name] = {
                "images": images,
                "masks": masks,
                "names": names,
            }
            print(f"  {ds_name.upper()}: {len(images)} images loaded")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if not datasets:
        print("\nERROR: No datasets found. Check --data-root path.")
        sys.exit(1)

    # Run evaluation
    all_results = []
    total_start = time.time()

    for ds_name, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"  Evaluating on {ds_name.upper()} ({len(data['images'])} images)")
        print(f"{'=' * 60}")

        for method in METHOD_NAMES:
            print(f"\n  Method: {method}")
            method_start = time.time()
            method_scores = []

            for idx, (img, mask, name) in enumerate(
                zip(data["images"], data["masks"], data["names"])
            ):
                try:
                    seg = run_pipeline(img, method)
                    metrics = compute_all_metrics(mask, seg)
                    metrics["Dataset"] = ds_name.upper()
                    metrics["Method"] = method
                    metrics["Image"] = name
                    all_results.append(metrics)
                    method_scores.append(metrics["F1"])

                    print(
                        f"    [{idx+1}/{len(data['images'])}] {name}: "
                        f"F1={metrics['F1']:.3f}, Dice={metrics['Dice']:.3f}, "
                        f"AJI={metrics['AJI']:.3f}"
                    )

                    # Save visual if requested
                    if args.save_visuals:
                        fig = show_segmentation_overlay(img, mask, seg, title=f"{method}: {name}")
                        fig.savefig(
                            f"data/results/{ds_name}_{name}_{method.replace(' ', '_')}.png",
                            dpi=100,
                            bbox_inches="tight",
                        )
                        plt.close(fig)

                except Exception as e:
                    print(f"    [{idx+1}] {name}: FAILED - {e}")

            elapsed = time.time() - method_start
            if method_scores:
                print(
                    f"  -> Mean F1: {np.mean(method_scores):.3f} "
                    f"± {np.std(method_scores):.3f} ({elapsed:.1f}s)"
                )

    total_elapsed = time.time() - total_start

    # Results summary
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for ds_name in df["Dataset"].unique():
        ds_data = df[df["Dataset"] == ds_name]

        print(f"\n  {ds_name} Dataset:")
        print(f"  {'-' * 65}")
        print(f"  {'Method':<30s} {'F1':>12s} {'Dice':>12s} {'AJI':>12s}")
        print(f"  {'-' * 65}")

        summary = (
            ds_data.groupby("Method")[["F1", "Dice", "AJI"]]
            .agg(["mean", "std"])
            .sort_values(("F1", "mean"), ascending=False)
        )

        for method_name, row in summary.iterrows():
            f1_str = f"{row[('F1', 'mean')]:.3f}±{row[('F1', 'std')]:.3f}"
            dice_str = f"{row[('Dice', 'mean')]:.3f}±{row[('Dice', 'std')]:.3f}"
            aji_str = f"{row[('AJI', 'mean')]:.3f}±{row[('AJI', 'std')]:.3f}"
            print(f"  {method_name:<30s} {f1_str:>12s} {dice_str:>12s} {aji_str:>12s}")

    # Calculate improvement
    print(f"\n  Improvement Analysis:")
    for ds_name in df["Dataset"].unique():
        ds_data = df[df["Dataset"] == ds_name]
        baseline_f1 = ds_data[ds_data["Method"] == "Baseline (No Norm)"]["F1"].mean()
        best_method = ds_data.groupby("Method")["F1"].mean().idxmax()
        best_f1 = ds_data[ds_data["Method"] == best_method]["F1"].mean()
        improvement = best_f1 - baseline_f1

        print(f"  {ds_name}: Baseline F1={baseline_f1:.3f} -> "
              f"Best ({best_method}): F1={best_f1:.3f} "
              f"(+{improvement:.3f}, +{improvement/baseline_f1*100:.1f}%)")

    # Save results
    output_csv = os.path.join(args.output_dir, "results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n  Results saved to: {output_csv}")

    # Save summary plots
    for ds_name in df["Dataset"].unique():
        ds_data = df[df["Dataset"] == ds_name]

        # Bar chart
        avg = ds_data.groupby("Method")[["F1", "Dice", "AJI"]].mean()
        fig = plot_metrics_comparison(avg.to_dict("index"), ["F1", "Dice", "AJI"])
        plt.title(f"{ds_name} - Metrics Comparison")
        fig.savefig(
            os.path.join(args.output_dir, f"{ds_name.lower()}_metrics.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Violin plot for F1
        f1_scores = {}
        for method in METHOD_NAMES:
            mdata = ds_data[ds_data["Method"] == method]
            if len(mdata) > 0:
                f1_scores[method] = mdata["F1"].tolist()

        if any(len(v) > 1 for v in f1_scores.values()):
            fig = plot_violin_metrics(f1_scores, "F1")
            plt.title(f"{ds_name} - F1 Distribution")
            fig.savefig(
                os.path.join(args.output_dir, f"{ds_name.lower()}_f1_violin.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # Save normalization comparison for first image
    if datasets:
        first_ds = list(datasets.values())[0]
        img = first_ds["images"][0]
        try:
            clahe_n, _ = normalize_clahe(img)
            mac_n, _, _ = normalize_macenko(img)
            rein_n = normalize_reinhard(img)
            norm_dict = {"CLAHE": clahe_n, "Macenko": mac_n, "Reinhard": rein_n}
            fig = compare_normalizations(img, norm_dict)
            fig.savefig(
                os.path.join(args.output_dir, "normalization_comparison.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        except Exception:
            pass

    print(f"\n  Plots saved to: {args.output_dir}/")
    print(f"\n  Total execution time: {total_elapsed:.1f}s")
    print("\n" + "=" * 70)
    print("  POC Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
