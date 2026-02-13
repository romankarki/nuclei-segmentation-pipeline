# Nuclei Segmentation POC

A proof-of-concept implementation of image processing techniques from the paper:

> **"Optimized detection and segmentation of nuclei in gastric cancer images using stain normalization and blurred artifact removal"**
> Martos et al., Pathology - Research and Practice, 2023

This project demonstrates how traditional image processing techniques (stain normalization, blur removal, adaptive thresholding, watershed segmentation) can achieve competitive nuclei segmentation results **without deep learning**.

---

## Project Structure

```
gastric_nuclei_poc/
├── src/                              # Source code modules
│   ├── preprocessing/
│   │   ├── blur_detection.py         # Laplacian & DCT-based blur detection
│   │   ├── stain_normalization.py    # CLAHE, Macenko, Reinhard, Vahadane
│   │   └── stain_deconvolution.py    # Ruifrok & PCA-based deconvolution
│   ├── segmentation/
│   │   ├── otsu_watershed.py         # Otsu + Watershed, Adaptive + Watershed
│   │   └── morphological_ops.py      # Post-processing operations
│   ├── evaluation/
│   │   └── metrics.py                # F1, Dice, AJI, Accuracy, IoU
│   └── utils/
│       ├── data_loader.py            # Dataset loading (MoNuSeg, TNBC)
│       └── visualization.py          # Plotting and overlay utilities
│
├── notebooks/                        # Interactive Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Dataset stats and visualization
│   ├── 02_blur_detection.ipynb       # Blur detection demonstration
│   ├── 03_normalization_comparison.ipynb  # Compare normalization methods
│   ├── 04_segmentation_pipeline.ipynb     # Full pipeline walkthrough
│   └── 05_results_analysis.ipynb     # Complete evaluation and results
│
├── experiments/                      # Configs and output files
│   ├── config_baseline.yaml
│   └── config_macenko_adaptive.yaml
│
├── data/                             # Data directory (not committed)
│   ├── raw/                          # For processed datasets
│   └── results/                      # Segmentation outputs
│
├── run_poc.py                        # Main execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd nuclei-segmentation-pipeline
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
cd gastric_nuclei_poc
pip install -r requirements.txt
```

> **Note**: If `spams` fails to install (it can be tricky on Windows), the project will still work. The Vahadane method uses scikit-learn's NMF as a fallback. You can skip `spams` with:
> ```bash
> pip install numpy scipy scikit-image scikit-learn opencv-python matplotlib seaborn jupyter notebook pandas tqdm Pillow imageio ipywidgets
> ```

### 4. Prepare the data

The project expects datasets in the following location relative to the project root:

```
nuclei-segmentation-pipeline/
├── data/
│   ├── monuseg_images/    # MoNuSeg H&E images (PNG)
│   ├── monuseg_masks/     # MoNuSeg ground truth masks (PNG)
│   ├── tnbc_images/       # TNBC H&E images (PNG)
│   └── tnbc_masks/        # TNBC ground truth masks (PNG)
└── gastric_nuclei_poc/    # This project
```

Image and mask files must have matching filenames within their respective directories.

---

## How to Run

### Option A: Run the Main Script (Quick Results)

```bash
cd gastric_nuclei_poc

# Run full evaluation on all datasets
python run_poc.py

# Run on a specific dataset
python run_poc.py --dataset monuseg
python run_poc.py --dataset tnbc

# Quick test with limited images
python run_poc.py --max-images 3

# Save visual outputs
python run_poc.py --save-visuals

# Specify custom data path
python run_poc.py --data-root ../data
```

The script will:
1. Load the dataset(s)
2. Run 7 different pipeline configurations
3. Compute F1, Dice, AJI, and Accuracy for each
4. Print a results summary table
5. Save results to `experiments/results.csv`
6. Save comparison plots to `experiments/`

### Option B: Interactive Jupyter Notebooks (Recommended for Learning)

```bash
cd gastric_nuclei_poc/notebooks
jupyter notebook
```

Run the notebooks in order:

| Notebook | What You'll Learn |
|----------|-------------------|
| **01_data_exploration** | Dataset structure, image statistics, staining variation |
| **02_blur_detection** | Why blur matters, detection methods, impact on segmentation |
| **03_normalization_comparison** | Side-by-side normalization comparison, color analysis |
| **04_segmentation_pipeline** | Full pipeline step-by-step, method comparison |
| **05_results_analysis** | Complete evaluation, paper-style tables and violin plots |

---

## Methods Implemented

### Preprocessing

| Method | Type | Paper Section | Description |
|--------|------|---------------|-------------|
| Laplacian Variance | Blur Detection | 3.3 | Simple edge-based sharpness measure |
| DCT-based (HiFST) | Blur Detection | 3.3 | Multi-scale frequency domain analysis |
| CLAHE | Enhancement | 3.4.1 | Adaptive histogram equalization (baseline) |
| Macenko | Normalization | 3.4.3 | SVD-based stain vector estimation |
| Reinhard | Normalization | - | Lab color space transfer |
| Vahadane (NMF) | Normalization | - | Non-negative matrix factorization |
| Ruifrok | Deconvolution | 3.4.2 | Standard H&E stain vector deconvolution |
| PCA-based | Deconvolution | 3.4.2 | Adaptive PCA stain vector estimation |

### Segmentation

| Method | Paper Section | Description |
|--------|---------------|-------------|
| Otsu + Watershed | Baseline | Global threshold + marker-controlled watershed |
| Adaptive + Watershed | 3.5.1-3.5.2 | MANA-inspired adaptive thresholding + watershed |

### Evaluation Metrics

| Metric | Paper Eq. | Description |
|--------|-----------|-------------|
| F1-measure | - | Harmonic mean of precision and recall |
| Dice Coefficient | Eq. 11 | Pixel-level similarity (2*TP / (2*TP+FP+FN)) |
| Aggregated Jaccard Index (AJI) | Eq. 12 | Object-level metric penalizing over/under-segmentation |
| Accuracy | - | Pixel-level accuracy percentage |
| IoU | - | Intersection over Union |

---

## Paper's Key Results (Reference)

### Gastric Cancer Dataset (Table 2)

| Method | F1 | Dice | AJI |
|--------|----|----|-----|
| Basic (no normalization) | 0.807 | 0.643 | 0.344 |
| **Macenko** | **0.854** | **0.695** | **0.389** |
| Hoque (CD) | 0.847 | 0.674 | 0.376 |

### TCGA/MoNuSeg Dataset (Table 3)

| Method | F1 | Dice | AJI |
|--------|----|----|-----|
| Basic (no normalization) | 0.847 | 0.650 | 0.384 |
| Macenko | 0.896 | 0.727 | 0.434 |
| **Hoque (CD)** | **0.907** | **0.739** | **0.458** |

**Key finding**: Stain normalization improves F1 by up to **5.8%** and AJI by up to **7.4%**.

---

## Understanding the Visual Outputs

The segmentation overlay uses the paper's color scheme:
- **White**: True Positives (correctly detected nuclei)
- **Purple**: False Negatives (missed nuclei)
- **Green**: False Positives (falsely detected regions)

---

## Troubleshooting

**"Module not found" errors in notebooks:**
Make sure you're running notebooks from the `notebooks/` directory and the kernel has the virtual environment activated.

**Slow execution:**
Use `--max-images 3` for quick testing. The adaptive watershed method is slower than basic Otsu.

**spams installation fails:**
Skip it - the Vahadane method uses sklearn NMF as fallback. Install other deps manually:
```bash
pip install numpy scipy scikit-image scikit-learn opencv-python matplotlib seaborn jupyter pandas tqdm Pillow
```

**Images not loading:**
Verify data directory structure matches what's described in the Setup section. Image and mask filenames must match exactly.

---

## References

- Martos, O., et al. (2023). "Optimized detection and segmentation of nuclei in gastric cancer images using stain normalization and blurred artifact removal." *Pathology - Research and Practice*, 248, 154694.
- Macenko, M., et al. (2009). "A method for normalizing histology slides for quantitative analysis." *ISBI*.
- Ruifrok, A.C. & Johnston, D.A. (2001). "Quantification of histochemical staining by color deconvolution." *Analytical and Quantitative Cytology and Histology*.
- Kumar, N., et al. (2017). "A dataset and a technique for generalized nuclear segmentation for computational pathology." *IEEE TMI*.
