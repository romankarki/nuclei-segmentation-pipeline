"""
Nuclei segmentation using thresholding and marker-controlled watershed.

Implements:
1. Li/Otsu threshold + Watershed (baseline and normalization methods)
2. Adaptive thresholding + Watershed (paper's proposed method)
3. Full pipeline with morphological operations and area-based correction

Reference: Paper Sections 3.5 and 3.6

NOTE: Li's minimum cross-entropy threshold is used by default instead of
Otsu because it handles the unequal foreground/background distribution of
nuclei segmentation much better (nuclei are typically <50% of the image).
Otsu assumes roughly equal class variances, which fails on hard images.

NOTE: The paper uses 1000x1000 images. All morphological parameters are
automatically scaled based on actual image resolution so the pipeline
works correctly on smaller images (e.g., 256x256 TNBC patches).
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
from skimage.filters import threshold_li


# ============================================================================
# Resolution-adaptive parameter helper
# ============================================================================
def _scale_params(image):
    """
    Compute resolution-scaled morphological parameters.

    The paper's parameters are calibrated for ~1000x1000 images.
    This function scales them proportionally for any image size.

    Returns dict with: min_area, disk_r, hole_area, min_dist, block_size
    """
    h, w = image.shape[:2] if image.ndim == 3 else image.shape
    # Scale factor: 1.0 for 1000x1000, ~0.065 for 256x256
    scale = (h * w) / 1_000_000
    # Linear dimension scale: 1.0 for 1000, 0.256 for 256
    lin_scale = min(h, w) / 1000

    return {
        "min_area": max(10, int(150 * scale)),
        "disk_r": max(1, round(3 * lin_scale)),
        "hole_area": max(30, int(500 * scale)),
        "min_dist": max(3, round(8 * lin_scale)),
        "block_size": max(11, int(51 * lin_scale)) | 1,  # must be odd
        "gauss_ksize": max(3, int(7 * lin_scale)) | 1,   # must be odd
    }


# ============================================================================
# Global Threshold + Watershed (supports both Li and Otsu)
# ============================================================================
def segment_otsu_watershed(image, min_nucleus_area=None, threshold_method='li'):
    """
    Nuclei segmentation: global threshold + watershed.

    Supports multiple threshold methods. Li's minimum cross-entropy
    threshold is the default because it handles the unequal class
    distribution of nuclei segmentation better than Otsu.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale)
    min_nucleus_area : int or None
        Minimum nucleus area in pixels. If None, auto-scaled from resolution.
    threshold_method : str
        'li' (default, recommended) or 'otsu'

    Returns
    -------
    segmented : np.ndarray
        Binary segmentation mask
    labeled : np.ndarray
        Labeled nuclei (each nucleus has unique integer ID)
    """
    params = _scale_params(image)
    if min_nucleus_area is None:
        min_nucleus_area = params["min_area"]

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Global thresholding
    if threshold_method == 'li':
        # Li's minimum cross-entropy threshold — better for nuclei
        thresh = threshold_li(blurred)
        binary = (blurred < thresh)  # nuclei are dark
    else:
        # Otsu's threshold (legacy)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = binary.astype(bool)

    # Auto-detect polarity: nuclei should be the minority class.
    fg_frac = np.sum(binary) / binary.size
    if fg_frac > 0.50:
        binary = ~binary

    # Morphological cleaning (resolution-scaled)
    binary = remove_small_holes(binary, area_threshold=params["hole_area"])
    binary = remove_small_objects(binary, min_size=min_nucleus_area)

    # Opening to smooth borders (resolution-scaled disk)
    selem = disk(params["disk_r"])
    binary = opening(binary, selem)

    # Distance transform for watershed markers
    distance = ndimage.distance_transform_edt(binary)
    gk = params["gauss_ksize"]
    distance_smooth = cv2.GaussianBlur(distance, (gk, gk), 2)

    # Find markers (local maxima of distance transform)
    coords = peak_local_max(
        distance_smooth, min_distance=params["min_dist"],
        labels=binary.astype(int)
    )

    if len(coords) == 0:
        segmented = binary.astype(np.uint8)
        labeled_out = label(segmented)
        return segmented, labeled_out

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
    min_nucleus_area=None,
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
    min_nucleus_area : int or None
        Minimum nucleus area in pixels (chi_min in paper).
        If None, auto-scaled from resolution.
    correction_ratio : float
        Area-based correction ratio (23% in paper)

    Returns
    -------
    segmented : np.ndarray
        Binary segmentation mask
    labeled : np.ndarray
        Labeled nuclei
    """
    params = _scale_params(image)
    if min_nucleus_area is None:
        min_nucleus_area = params["min_area"]

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
        block_size = params["block_size"]
        C = (1 - thresh) * 255  # Convert sensitivity to constant offset

        gray_uint8 = (gray_smooth * 255).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            gray_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        binary = binary.astype(bool)

        # Auto-detect polarity: if foreground > 50%, invert
        if np.sum(binary) / binary.size > 0.50:
            binary = ~binary

        # Quick morphological cleanup
        binary = remove_small_objects(binary, min_size=max(5, min_nucleus_area // 2))

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

    # Opening (resolution-scaled disk)
    selem = disk(params["disk_r"])
    binary = opening(binary, selem)

    # Remove small objects below chi_min
    binary = remove_small_objects(binary, min_size=min_nucleus_area)

    # ---- Marker-controlled Watershed ----
    distance = ndimage.distance_transform_edt(binary)
    gk = params["gauss_ksize"]
    distance_smooth = cv2.GaussianBlur(distance.astype(np.float32), (gk, gk), 2)

    # Extended-minima transform (H-minima + regional minima)
    h_value = 2  # H-minima suppression height
    suppressed = distance_smooth.copy()
    suppressed = np.maximum(suppressed - h_value, 0)
    coords = peak_local_max(
        suppressed, min_distance=params["min_dist"],
        labels=binary.astype(int)
    )

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


def _area_based_correction(labeled, min_area=10, correction_ratio=0.23):
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
    method="li",
    min_nucleus_area=None,
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
        'li' for Li threshold + watershed (recommended),
        'otsu' for Otsu threshold + watershed,
        'adaptive' for paper's adaptive threshold method
    min_nucleus_area : int or None
        If None, auto-scaled from image resolution.

    Returns
    -------
    segmented : np.ndarray
        Binary segmentation mask
    labeled : np.ndarray
        Labeled nuclei
    """
    if method in ("li", "otsu"):
        if h_channel is not None:
            return segment_otsu_watershed(h_channel, min_nucleus_area,
                                          threshold_method=method)
        return segment_otsu_watershed(image, min_nucleus_area,
                                      threshold_method=method)
    elif method == "adaptive":
        return segment_adaptive_watershed(
            image, h_channel, min_nucleus_area, **kwargs
        )
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
