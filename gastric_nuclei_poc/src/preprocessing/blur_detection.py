"""
Blur detection and removal for histopathology images. (to improve accuracy with frequency measure)

Implements two approaches:
1. Laplacian Variance method (simple, fast baseline)
2. DCT-based method inspired by HiFST (Spatially Varying Blur Detection)

Reference: Paper Section 3.3 - "Artifact removal with spatially varying blur detection"
The HiFST method uses Discrete Cosine Transform on multi-scale patches to detect
out-of-focus areas that negatively impact segmentation accuracy.
"""

import numpy as np
import cv2
from scipy.fftpack import dct


def laplacian_variance_blur_map(image, kernel_size=15):
    """
    Compute a blur map using local Laplacian variance.

    This is a simple and effective baseline method. Blurred regions
    have lower variance of the Laplacian (fewer high-frequency details).

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale)
    kernel_size : int
        Size of the local window for variance computation

    Returns
    -------
    blur_map : np.ndarray
        Normalized blur map [0, 1], higher = sharper
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute local variance using box filter
    laplacian_sq = laplacian ** 2
    local_mean = cv2.blur(laplacian, (kernel_size, kernel_size))
    local_mean_sq = cv2.blur(laplacian_sq, (kernel_size, kernel_size))
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.maximum(local_var, 0)

    # Normalize to [0, 1]
    if local_var.max() > 0:
        blur_map = local_var / local_var.max()
    else:
        blur_map = np.zeros_like(local_var)

    return blur_map.astype(np.float32)


def dct_blur_map(image, scales=(4, 8, 16), top_k=5):
    """
    Compute a blur map using multi-scale DCT coefficients (HiFST-inspired).

    Based on the paper's description of Spatially Varying Blur Detection:
    - For each pixel, extract patches at multiple scales
    - Compute high-frequency DCT coefficients
    - Sort and combine across scales
    - Generate blur detection map via max pooling and entropy weighting

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale)
    scales : tuple of int
        Patch sizes at different scales
    top_k : int
        Number of top DCT coefficients to use

    Returns
    -------
    blur_map : np.ndarray
        Normalized blur map [0, 1], higher = sharper
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    h, w = gray.shape

    # Step 1: Gaussian filtering to remove high-frequency noise
    filtered = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Step 2: Compute gradient magnitude (Eq. 2 in paper)
    gx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Step 3: Multi-scale DCT coefficient extraction
    layers = []
    for scale in scales:
        half = scale // 2
        layer = np.zeros((h, w), dtype=np.float64)

        # Pad the gradient magnitude image
        padded = cv2.copyMakeBorder(
            gradient_mag, half, half, half, half, cv2.BORDER_REFLECT
        )

        # Process with stride for efficiency (every 4th pixel)
        stride = max(1, scale // 4)
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                patch = padded[i : i + scale, j : j + scale]
                if patch.shape[0] != scale or patch.shape[1] != scale:
                    continue

                # 2D DCT
                dct_coeffs = dct(dct(patch, axis=0, norm="ortho"), axis=1, norm="ortho")

                # Extract high-frequency coefficients (exclude DC and low-freq)
                hf_mask = np.zeros_like(dct_coeffs, dtype=bool)
                n = scale // 2
                hf_mask[n:, :] = True
                hf_mask[:, n:] = True
                hf_coeffs = np.abs(dct_coeffs[hf_mask])

                # Sort and take top-k
                hf_sorted = np.sort(hf_coeffs)[::-1]
                value = np.mean(hf_sorted[:top_k]) if len(hf_sorted) >= top_k else np.mean(hf_sorted)

                # Fill the stride block
                i_end = min(i + stride, h)
                j_end = min(j + stride, w)
                layer[i:i_end, j:j_end] = value

        # Normalize layer to [0, 1]
        if layer.max() > 0:
            layer = layer / layer.max()
        layers.append(layer)

    # Step 4: Max pooling across scales (T matrix in paper)
    T = np.maximum.reduce(layers)

    # Step 5: Compute entropy weighting (omega in paper)
    # Local entropy as weighting factor
    kernel = 15
    local_entropy = np.zeros_like(T)
    padded_T = cv2.copyMakeBorder(T, kernel // 2, kernel // 2, kernel // 2, kernel // 2, cv2.BORDER_REFLECT)

    for i in range(h):
        for j in range(w):
            patch = padded_T[i : i + kernel, j : j + kernel]
            # Compute entropy of the patch
            hist, _ = np.histogram(patch, bins=32, range=(0, 1))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            local_entropy[i, j] = -np.sum(hist * np.log2(hist))

    if local_entropy.max() > 0:
        omega = local_entropy / local_entropy.max()
    else:
        omega = np.zeros_like(local_entropy)

    # Step 6: Hadamard product (Eq. 3 in paper)
    blur_map = T * omega

    # Normalize final result
    if blur_map.max() > 0:
        blur_map = blur_map / blur_map.max()

    # Smooth with edge-preserving filter
    blur_map_uint8 = (blur_map * 255).astype(np.uint8)
    blur_map_smooth = cv2.bilateralFilter(blur_map_uint8, 9, 75, 75)
    blur_map = blur_map_smooth.astype(np.float32) / 255.0

    return blur_map


def detect_blur(image, method="laplacian", threshold=0.15, **kwargs):
    """
    Detect blurred regions in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB)
    method : str
        'laplacian' for Laplacian variance, 'dct' for DCT-based
    threshold : float
        Threshold for blur/sharp classification

    Returns
    -------
    blur_map : np.ndarray
        Continuous blur map [0, 1]
    blur_mask : np.ndarray
        Binary mask (1 = sharp/keep, 0 = blurred/remove)
    """
    if method == "laplacian":
        blur_map = laplacian_variance_blur_map(image, **kwargs)
    elif method == "dct":
        blur_map = dct_blur_map(image, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Binary mask: 1 = sharp (keep), 0 = blurred (remove)
    blur_mask = (blur_map > threshold).astype(np.uint8)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_CLOSE, kernel)
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_OPEN, kernel)

    return blur_map, blur_mask


def remove_blur_artifacts(image, blur_mask):
    """
    Remove blurred artifacts from image using the blur mask.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB)
    blur_mask : np.ndarray
        Binary mask (1 = keep, 0 = remove)

    Returns
    -------
    clean_image : np.ndarray
        Image with blurred regions set to white (background)
    blur_fraction : float
        Fraction of image that was blurred
    """
    clean_image = image.copy()
    # Set blurred regions to white (background)
    clean_image[blur_mask == 0] = [255, 255, 255]

    total_pixels = blur_mask.size
    blurred_pixels = np.sum(blur_mask == 0)
    blur_fraction = blurred_pixels / total_pixels

    return clean_image, blur_fraction


def compute_blur_score(image):
    """
    Compute a single blur score for the entire image.

    Uses the variance of Laplacian: lower = more blurred.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    score : float
        Blur score (higher = sharper)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()
