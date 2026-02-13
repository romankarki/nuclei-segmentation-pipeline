"""
Morphological operations for nuclei post-processing.

Reference: Paper Section 3.5.2
Describes the morphological operations applied after thresholding:
filling, opening, area filtering, and watershed refinement.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import (
    disk,
    opening as morph_opening,
    closing as morph_closing,
    erosion,
    dilation,
    remove_small_objects,
    remove_small_holes,
)
from skimage.measure import label, regionprops


def fill_holes(binary_mask):
    """Fill holes in binary objects."""
    return ndimage.binary_fill_holes(binary_mask).astype(np.uint8)


def smooth_borders(binary_mask, radius=3):
    """Smooth object borders using opening with disk structuring element."""
    selem = disk(radius)
    return morph_opening(binary_mask.astype(bool), selem).astype(np.uint8)


def remove_small_nuclei(binary_mask, min_area=150):
    """Remove objects smaller than minimum nucleus area."""
    labeled = label(binary_mask)
    cleaned = remove_small_objects(labeled, min_size=min_area)
    return (cleaned > 0).astype(np.uint8)


def remove_large_artifacts(binary_mask, max_area_factor=10):
    """
    Remove excessively large objects (likely artifacts).

    Objects larger than max_area_factor * mean_area are removed.
    Note: Paper mentions this can be risky as large nuclei are valid.
    """
    labeled = label(binary_mask)
    props = regionprops(labeled)

    if len(props) == 0:
        return binary_mask

    areas = [p.area for p in props]
    mean_area = np.mean(areas)
    max_area = max_area_factor * mean_area

    cleaned = labeled.copy()
    for prop in props:
        if prop.area > max_area:
            cleaned[labeled == prop.label] = 0

    return (cleaned > 0).astype(np.uint8)


def separate_touching_nuclei(binary_mask, method="watershed"):
    """
    Separate touching/overlapping nuclei.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask with potentially touching nuclei
    method : str
        'watershed' or 'erosion'

    Returns
    -------
    separated : np.ndarray
        Labeled mask with separated nuclei
    """
    if method == "watershed":
        # Distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        distance_smooth = cv2.GaussianBlur(
            distance.astype(np.float32), (7, 7), 2
        )

        # Find local maxima as markers
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        coords = peak_local_max(
            distance_smooth, min_distance=5, labels=binary_mask.astype(int)
        )
        mask = np.zeros(distance_smooth.shape, dtype=bool)
        if len(coords) > 0:
            mask[tuple(coords.T)] = True
        markers = label(mask)

        separated = watershed(-distance_smooth, markers, mask=binary_mask.astype(bool))

    elif method == "erosion":
        # Simple erosion to separate touching objects
        selem = disk(2)
        eroded = erosion(binary_mask.astype(bool), selem)
        separated = label(eroded)
        # Dilate back within original mask
        for region_id in range(1, separated.max() + 1):
            region = (separated == region_id)
            dilated = dilation(region, disk(2))
            # Only expand within original mask
            dilated = dilated & binary_mask.astype(bool)
            separated[dilated & (separated == 0)] = region_id

    else:
        raise ValueError(f"Unknown method: {method}")

    return separated


def full_morphological_pipeline(binary_mask, min_area=150, correction_ratio=0.23):
    """
    Complete morphological post-processing pipeline.

    Steps (from paper):
    1. Fill holes
    2. Opening with disk(3)
    3. Remove small objects (< min_area)
    4. Separate touching nuclei with watershed
    5. Area-based correction (remove < 23% of mean area)

    Parameters
    ----------
    binary_mask : np.ndarray
        Input binary mask from thresholding
    min_area : int
        Minimum nucleus area
    correction_ratio : float
        Ratio for area-based correction

    Returns
    -------
    result : np.ndarray
        Cleaned and separated labeled nuclei
    """
    # Step 1: Fill holes
    mask = fill_holes(binary_mask)

    # Step 2: Opening
    mask = smooth_borders(mask, radius=3)

    # Step 3: Remove small objects
    mask = remove_small_nuclei(mask, min_area)

    # Step 4: Watershed separation
    labeled = separate_touching_nuclei(mask, method="watershed")

    # Step 5: Area-based correction
    props = regionprops(labeled)
    if len(props) > 0:
        areas = [p.area for p in props]
        mean_area = np.mean(areas)
        min_corrected = max(min_area, int(correction_ratio * mean_area))

        for prop in props:
            if prop.area < min_corrected:
                labeled[labeled == prop.label] = 0

    # Relabel
    labeled = label(labeled > 0)

    return labeled
