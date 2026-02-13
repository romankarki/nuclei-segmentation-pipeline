"""
Nuclei segmentation using thresholding and marker-controlled watershed.

Implements:
1. Simple Otsu + Watershed (baseline comparison)
2. Adaptive thresholding + Watershed (paper's proposed method)
3. Full pipeline with morphological operations and area-based correction

Reference: Paper Sections 3.5 and 3.6
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk,
    opening,
    closing,
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops


# ============================================================================
# Baseline: Simple Otsu + Watershed
# ============================================================================
def segment_otsu_watershed(image, min_nucleus_area=150):
    """
    Basic nuclei segmentation: global Otsu threshold + watershed.

    This is the baseline method WITHOUT any stain normalization.
    Operates on the grayscale or hematoxylin channel.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale)
    min_nucleus_area : int
        Minimum nucleus area in pixels

    Returns
    -------
    segmented : np.ndarray
        Binary segmentation mask
    labeled : np.ndarray
        Labeled nuclei (each nucleus has unique integer ID)
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Global Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary.astype(bool)

    # Morphological cleaning
    binary = remove_small_holes(binary, area_threshold=500)
    binary = remove_small_objects(binary, min_size=min_nucleus_area)

    # Opening to smooth borders
    selem = disk(3)
    binary = opening(binary, selem)

    # Distance transform for watershed markers
    distance = ndimage.distance_transform_edt(binary)
    distance_smooth = cv2.GaussianBlur(distance, (7, 7), 2)

    # Find markers (local maxima of distance transform)
    coords = peak_local_max(
        distance_smooth, min_distance=8, labels=binary.astype(int)
    )
    mask = np.zeros(distance_smooth.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)

    # Watershed segmentation
    labeled = watershed(-distance_smooth, markers, mask=binary)

    # Remove small objects (area-based correction)
    labeled = _area_based_correction(labeled, min_nucleus_area)

    # Binary mask
    segmented = (labeled > 0).astype(np.uint8)

    return segmented, labeled


# ============================================================================
# Paper Method: Adaptive Threshold + Watershed
# ============================================================================
def _compute_threshold_candidates(gray_image):
    """
    Compute candidate threshold proposals using PWMCURVE method (Eq. 10).

    Inspired by MANA algorithm: computes progressive weighted mean from
    the histogram, fits polynomial, finds inflection points.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image normalized to [0, 1]

    Returns
    -------
    candidates : list of float
        Candidate threshold values in [0, 1]
    """
    # Compute histogram
    hist, bin_edges = np.histogram(gray_image.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Progressive weighted mean curve (PWMCURVE, Eq. 10)
    pwm_curve = np.zeros(256)
    cum_weights = 0
    cum_weighted_sum = 0
    for i in range(256):
        cum_weights += hist[i]
        cum_weighted_sum += hist[i] * bin_centers[i]
        if cum_weights > 0:
            pwm_curve[i] = cum_weighted_sum / cum_weights

    # Fit polynomial (15th order as in paper)
    try:
        x = np.arange(256)
        coeffs = np.polyfit(x, pwm_curve, 15)
        poly = np.poly1d(coeffs)

        # Second derivative for inflection points
        poly_2nd = poly.deriv(2)
        roots = np.roots(poly_2nd)

        # Filter real, valid roots and convert to [0, 1] range
        candidates = []
        for root in roots:
            if np.isreal(root):
                val = float(np.real(root))
                if 0 < val < 256:
                    normalized = val / 255.0
                    if normalized > 0.5:  # Paper: choose candidates above 0.5
                        candidates.append(normalized)
    except Exception:
        candidates = []

    # Fallback: if no candidates found, use median-based estimation
    if not candidates:
        median_val = np.median(gray_image[gray_image > 0.1])
        candidates = [max(0.5, median_val)]

    # Add median intensity as minimum sensitivity level
    median_intensity = np.median(gray_image)
    if median_intensity > 0.5:
        candidates.append(median_intensity)

    return sorted(set(candidates))


def segment_adaptive_watershed(
    image,
    h_channel=None,
    min_nucleus_area=150,
    correction_ratio=0.23,
):
    """
    Paper's proposed segmentation: adaptive thresholding + watershed.

    Steps (from paper Sections 3.5.1 - 3.6):
    1. Compute candidate threshold proposals (MANA-inspired)
    2. For each candidate, perform segmentation
    3. Select threshold with highest median object area
    4. Apply morphological operations
    5. Marker-controlled watershed
    6. Area-based correction

    Parameters
    ----------
    image : np.ndarray (H, W, 3) or (H, W)
        Input image (RGB or grayscale)
    h_channel : np.ndarray or None
        Hematoxylin channel (grayscale). If None, derived from image.
    min_nucleus_area : int
        Minimum nucleus area in pixels (chi_min in paper)
    correction_ratio : float
        Area-based correction ratio (23% in paper)

    Returns
    -------
    segmented : np.ndarray
        Binary segmentation mask
    labeled : np.ndarray
        Labeled nuclei
    """
    # Prepare the channel for segmentation
    if h_channel is not None:
        gray = h_channel.copy()
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Normalize to [0, 1]
    if gray.max() > 1:
        gray_norm = gray.astype(np.float64) / 255.0
    else:
        gray_norm = gray.astype(np.float64)

    # Gaussian smoothing (sigma=1 as in paper)
    gray_smooth = cv2.GaussianBlur(gray_norm, (5, 5), 1.0)

    # Compute candidate thresholds
    candidates = _compute_threshold_candidates(gray_smooth)

    # Evaluate each candidate
    best_median_area = 0
    best_threshold = candidates[0] if candidates else 0.5
    best_binary = None

    for thresh in candidates:
        # Local adaptive thresholding with candidate as sensitivity
        # The sensitivity parameter controls how much the threshold adapts
        block_size = 51  # Must be odd
        C = (1 - thresh) * 255  # Convert sensitivity to constant offset

        gray_uint8 = (gray_smooth * 255).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            gray_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        binary = binary.astype(bool)

        # Quick morphological cleanup
        binary = remove_small_objects(binary, min_size=min_nucleus_area // 2)

        if np.sum(binary) == 0:
            continue

        # Compute median area of objects
        labeled_temp = label(binary)
        props = regionprops(labeled_temp)
        if len(props) > 0:
            areas = [p.area for p in props]
            median_area = np.median(areas)

            if median_area > best_median_area:
                best_median_area = median_area
                best_threshold = thresh
                best_binary = binary

    if best_binary is None:
        # Fallback to Otsu
        return segment_otsu_watershed(image, min_nucleus_area)

    binary = best_binary

    # ---- Morphological operations (Section 3.5.2) ----
    # Fill holes
    binary = ndimage.binary_fill_holes(binary)

    # Opening with disk(3)
    selem = disk(3)
    binary = opening(binary, selem)

    # Remove small objects below chi_min
    binary = remove_small_objects(binary, min_size=min_nucleus_area)

    # ---- Marker-controlled Watershed ----
    distance = ndimage.distance_transform_edt(binary)
    distance_smooth = cv2.GaussianBlur(distance.astype(np.float32), (7, 7), 2)

    # Extended-minima transform (H-minima + regional minima)
    h_value = 2  # H-minima suppression height
    suppressed = distance_smooth.copy()
    suppressed = np.maximum(suppressed - h_value, 0)
    coords = peak_local_max(suppressed, min_distance=5, labels=binary.astype(int))

    if len(coords) == 0:
        segmented = binary.astype(np.uint8)
        labeled = label(segmented)
        return segmented, labeled

    mask = np.zeros(distance_smooth.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)

    # Watershed
    labeled = watershed(-distance_smooth, markers, mask=binary)

    # Remove small objects post-watershed
    labeled = remove_small_objects(labeled.astype(bool), min_size=min_nucleus_area)
    labeled = label(labeled)

    # ---- Area-based correction (Section 3.6) ----
    labeled = _area_based_correction(labeled, min_nucleus_area, correction_ratio)

    segmented = (labeled > 0).astype(np.uint8)

    return segmented, labeled


def _area_based_correction(labeled, min_area=150, correction_ratio=0.23):
    """
    Area-based correction: remove objects smaller than 23% of mean area.

    Reference: Paper Section 3.6
    """
    props = regionprops(labeled)
    if len(props) == 0:
        return labeled

    areas = [p.area for p in props]
    mean_area = np.mean(areas)
    min_corrected_area = max(min_area, int(correction_ratio * mean_area))

    corrected = labeled.copy()
    for prop in props:
        if prop.area < min_corrected_area:
            corrected[labeled == prop.label] = 0

    # Relabel
    corrected = label(corrected > 0)
    return corrected


# ============================================================================
# Full pipeline
# ============================================================================
def segment_nuclei(
    image,
    h_channel=None,
    method="adaptive",
    min_nucleus_area=150,
    **kwargs,
):
    """
    Main segmentation function.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image
    h_channel : np.ndarray or None
        Hematoxylin channel from stain normalization/deconvolution
    method : str
        'otsu' for baseline, 'adaptive' for paper's method

    Returns
    -------
    segmented : np.ndarray
        Binary segmentation mask
    labeled : np.ndarray
        Labeled nuclei
    """
    if method == "otsu":
        if h_channel is not None:
            return segment_otsu_watershed(h_channel, min_nucleus_area)
        return segment_otsu_watershed(image, min_nucleus_area)
    elif method == "adaptive":
        return segment_adaptive_watershed(
            image, h_channel, min_nucleus_area, **kwargs
        )
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
