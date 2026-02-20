"""
Stain normalization methods for H&E histopathology images.

Implements:
1. CLAHE (baseline) - Contrast-Limited Adaptive Histogram Equalization
2. Macenko normalization (SVD-based, paper Method 3)
3. Reinhard normalization (color transfer in Lab space)
4. Vahadane normalization (sparse NMF-based)

Reference: Paper Sections 3.4.1 - 3.4.6
The paper shows that stain normalization is critical for improving
nuclei detection accuracy (up to 5.8% F1 improvement).
"""

import numpy as np
import cv2


# ============================================================================
# Method 1: CLAHE Baseline (Paper Section 3.4.1)
# ============================================================================
def normalize_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE on the V channel of HSV space.

    This is the simplest baseline from the paper. Does not produce
    a separate hematoxylin channel - uses enhanced value channel.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image
    clip_limit : float
        Contrast limit for CLAHE
    tile_grid_size : tuple
        Size of grid for histogram equalization

    Returns
    -------
    normalized : np.ndarray
        Enhanced RGB image
    h_channel : np.ndarray
        Hue channel (used as proxy for hematoxylin)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the V channel
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Hue channel as proxy for hematoxylin (paper uses this for segmentation)
    h_channel = hsv[:, :, 0]

    return normalized, h_channel


# ============================================================================
# Method 2: Macenko Normalization (Paper Section 3.4.3)
# ============================================================================
def _get_optical_density(image, I0=255):
    """Convert RGB image to Optical Density (OD) space (Eq. 4 in paper)."""
    image = image.astype(np.float64) + 1  # Avoid log(0)
    od = -np.log10(image / I0)
    return od


def _normalize_rows(matrix):
    """Normalize rows of a matrix to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def normalize_macenko(image, target_image=None, alpha=1, beta=0.15,
                      max_conc_target=None):
    """
    Macenko stain normalization (SVD-geodesic method).

    Reference: Macenko et al. "A method for normalizing histology slides
    for quantitative analysis" (ISBI 2009)

    Steps:
    1. Convert to Optical Density space
    2. Remove transparent pixels (below beta threshold)
    3. SVD to find stain directions
    4. Project OD values onto stain vectors
    5. Normalize using percentiles
    6. Reconstruct normalized image

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
        Source RGB image
    target_image : np.ndarray or None
        Target image for normalization. If None, uses default reference.
    alpha : float
        Percentile parameter (99th percentile when alpha=1)
    beta : float
        Threshold for transparent pixel removal in OD space
    max_conc_target : array-like of length 2, or None
        Target 99th-percentile concentrations [hematoxylin, eosin].
        Controls overall stain intensity of the output. Higher values
        produce darker/more saturated results. Default is [2.8, 1.5]
        to avoid washed-out normalization on MoNuSeg-like images.
        You can tune this per dataset, e.g. [2.4, 1.3] (lighter)
        or [3.0, 1.6] (stronger).

    Returns
    -------
    normalized : np.ndarray
        Color-normalized RGB image
    h_image : np.ndarray
        Hematoxylin channel image (grayscale)
    e_image : np.ndarray
        Eosin channel image (grayscale)
    """
    # Default target stain vectors (H&E reference from paper)
    # HEref = [0.5626, 0.7201, 0.4062; 0.2159, 0.8012, 0.5581]
    HEref = np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]])

    def _estimate_stain_vectors(img, alpha=1, beta=0.15):
        """Estimate stain vectors using SVD on OD space."""
        od = _get_optical_density(img)
        od_flat = od.reshape(-1, 3)

        # Remove transparent pixels
        od_thresh = od_flat[np.all(od_flat > beta, axis=1)]

        if len(od_thresh) < 10:
            # Fallback to default vectors
            return HEref.copy()

        # SVD
        try:
            _, _, Vt = np.linalg.svd(od_thresh, full_matrices=False)
        except np.linalg.LinAlgError:
            return HEref.copy()

        # Project on plane of first two SVD directions
        plane = Vt[:2, :]
        proj = od_thresh @ plane.T

        # Find angles
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        # Get robust extreme angles (alpha percentile)
        min_angle = np.percentile(angles, alpha)
        max_angle = np.percentile(angles, 100 - alpha)

        # Convert back to OD space
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

        # Ensure vectors point in the positive OD direction
        if np.sum(v1) < 0:
            v1 = -v1
        if np.sum(v2) < 0:
            v2 = -v2

        # Assign H vs E using cosine similarity to reference vectors
        # (more robust than comparing a single channel)
        ref_H = HEref[0]  # [0.5626, 0.7201, 0.4062]
        sim_v1_H = abs(np.dot(v1, ref_H)) / (np.linalg.norm(v1) * np.linalg.norm(ref_H) + 1e-10)
        sim_v2_H = abs(np.dot(v2, ref_H)) / (np.linalg.norm(v2) * np.linalg.norm(ref_H) + 1e-10)

        if sim_v1_H >= sim_v2_H:
            stain_vectors = np.array([v1, v2])
        else:
            stain_vectors = np.array([v2, v1])

        return _normalize_rows(stain_vectors)

    def _get_concentrations(img, stain_vectors):
        """Get stain concentration matrix."""
        od = _get_optical_density(img)
        od_flat = od.reshape(-1, 3)
        # Solve: OD = C * stain_vectors -> C = OD * pinv(stain_vectors)
        C = od_flat @ np.linalg.pinv(stain_vectors)
        return C

    # Estimate source stain vectors
    source_stain = _estimate_stain_vectors(image, alpha, beta)

    # Get source concentrations
    source_conc = _get_concentrations(image, source_stain)

    # Normalize concentrations using percentiles
    max_conc_source = np.percentile(source_conc, 99, axis=0)

    if target_image is not None:
        target_stain = _estimate_stain_vectors(target_image, alpha, beta)
        if max_conc_target is None:
            target_conc = _get_concentrations(target_image, target_stain)
            max_conc_target = np.percentile(target_conc, 99, axis=0)
        else:
            max_conc_target = np.asarray(max_conc_target, dtype=np.float64)
    else:
        target_stain = HEref
        if max_conc_target is None:
            max_conc_target = np.array([2.8, 1.5])
        else:
            max_conc_target = np.asarray(max_conc_target, dtype=np.float64)

    # Normalize
    max_conc_source[max_conc_source == 0] = 1
    conc_normalized = source_conc * (max_conc_target / max_conc_source)

    # Reconstruct (Eq. 6 in paper): I_norm = I0 * exp(-W_T * C_norm)
    od_normalized = conc_normalized @ target_stain
    normalized = 255 * np.exp(-od_normalized * np.log(10))
    normalized = normalized.reshape(image.shape)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Extract hematoxylin and eosin channels
    h_conc = conc_normalized[:, 0].reshape(image.shape[:2])
    e_conc = conc_normalized[:, 1].reshape(image.shape[:2])

    # Convert to image-like representation
    h_image = np.clip(255 * np.exp(-h_conc * np.log(10)), 0, 255).astype(np.uint8)
    e_image = np.clip(255 * np.exp(-e_conc * np.log(10)), 0, 255).astype(np.uint8)

    return normalized, h_image, e_image


# ============================================================================
# Method 3: Reinhard Normalization (Color Transfer)
# ============================================================================
def normalize_reinhard(image, target_image=None):
    """
    Reinhard color normalization (Lab color space transfer).

    Transfers the mean and standard deviation of color channels
    from a target image to the source image in Lab color space.

    Parameters
    ----------
    image : np.ndarray
        Source RGB image
    target_image : np.ndarray or None
        Target image. If None, uses default statistics.

    Returns
    -------
    normalized : np.ndarray
        Color-normalized RGB image
    """
    # Convert to Lab
    source_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)

    if target_image is not None:
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float64)
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
    else:
        # Default target statistics (typical H&E slide)
        target_mean = np.array([148.60, 133.00, 145.00])
        target_std = np.array([41.56, 9.01, 6.67])

    source_mean = np.mean(source_lab, axis=(0, 1))
    source_std = np.std(source_lab, axis=(0, 1))

    # Avoid division by zero
    source_std[source_std == 0] = 1

    # Transfer: (x - source_mean) / source_std * target_std + target_mean
    for i in range(3):
        source_lab[:, :, i] = (
            (source_lab[:, :, i] - source_mean[i]) / source_std[i]
        ) * target_std[i] + target_mean[i]

    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    normalized = cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)

    return normalized


# ============================================================================
# Method 4: Vahadane-like Normalization (Sparse NMF approach)
# ============================================================================
def normalize_vahadane_simple(image, target_image=None):
    """
    Simplified Vahadane-style normalization using dictionary learning.

    Uses Non-negative Matrix Factorization (NMF) as a simplified
    alternative to the full sparse NMF (SPAMS library dependency).

    Parameters
    ----------
    image : np.ndarray
        Source RGB image
    target_image : np.ndarray or None
        Target image

    Returns
    -------
    normalized : np.ndarray
        Color-normalized RGB image
    h_image : np.ndarray
        Hematoxylin channel
    e_image : np.ndarray
        Eosin channel
    """
    from sklearn.decomposition import NMF

    def _get_stain_matrix(img, n_components=2):
        """Extract stain matrix using NMF on optical density."""
        od = _get_optical_density(img)
        od_flat = od.reshape(-1, 3)

        # Remove near-zero rows (background)
        mask = np.all(od_flat > 0.05, axis=1)
        od_filtered = od_flat[mask]

        if len(od_filtered) < 100:
            return np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]])

        # NMF decomposition
        model = NMF(n_components=n_components, init="nndsvd", max_iter=300, random_state=42)
        try:
            model.fit(od_filtered)
            W = model.components_  # (2, 3)
        except Exception:
            return np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]])

        # Normalize rows
        return _normalize_rows(W)

    source_stain = _get_stain_matrix(image)

    if target_image is not None:
        target_stain = _get_stain_matrix(target_image)
    else:
        target_stain = np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]])

    # Get source concentrations
    od = _get_optical_density(image)
    od_flat = od.reshape(-1, 3)
    source_conc = od_flat @ np.linalg.pinv(source_stain)

    # Normalize concentrations
    max_conc_s = np.percentile(source_conc, 99, axis=0)
    max_conc_s[max_conc_s == 0] = 1
    max_conc_t = np.array([2.8, 1.5])

    conc_normalized = source_conc * (max_conc_t / max_conc_s)

    # Reconstruct
    od_normalized = conc_normalized @ target_stain
    normalized = 255 * np.exp(-od_normalized * np.log(10))
    normalized = normalized.reshape(image.shape)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Extract channels
    h_conc = conc_normalized[:, 0].reshape(image.shape[:2])
    e_conc = conc_normalized[:, 1].reshape(image.shape[:2])
    h_image = np.clip(255 * np.exp(-h_conc * np.log(10)), 0, 255).astype(np.uint8)
    e_image = np.clip(255 * np.exp(-e_conc * np.log(10)), 0, 255).astype(np.uint8)

    return normalized, h_image, e_image


# ============================================================================
# Convenience function
# ============================================================================
def normalize_image(image, method="macenko", target_image=None, **kwargs):
    """
    Apply stain normalization to an image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image
    method : str
        One of 'clahe', 'macenko', 'reinhard', 'vahadane'
    target_image : np.ndarray or None
        Reference image for normalization

    Returns
    -------
    result : dict
        Contains 'normalized', and optionally 'hematoxylin', 'eosin'
    """
    result = {}

    if method == "clahe":
        norm, h_ch = normalize_clahe(image, **kwargs)
        result["normalized"] = norm
        result["hematoxylin"] = h_ch

    elif method == "macenko":
        norm, h_img, e_img = normalize_macenko(image, target_image, **kwargs)
        result["normalized"] = norm
        result["hematoxylin"] = h_img
        result["eosin"] = e_img

    elif method == "reinhard":
        norm = normalize_reinhard(image, target_image)
        result["normalized"] = norm

    elif method == "vahadane":
        norm, h_img, e_img = normalize_vahadane_simple(image, target_image, **kwargs)
        result["normalized"] = norm
        result["hematoxylin"] = h_img
        result["eosin"] = e_img

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return result
