"""
Stain deconvolution methods for H&E histopathology images.

Implements:
1. Ruifrok Color Deconvolution (standard basis vectors)
2. PCA-based Color Deconvolution (adaptive, paper Method 2)

Reference: Paper Section 3.4.2
Color deconvolution separates H&E stained images into individual
hematoxylin and eosin channels using known or estimated stain vectors.
"""

import numpy as np
import cv2


# Standard stain vectors from literature (Ruifrok & Johnston, 2001)
STAIN_VECTORS = {
    "ruifrok": {
        "hematoxylin": np.array([0.6500286, 0.704031, 0.2860126]),
        "eosin": np.array([0.07200929, 0.99003, 0.10567]),
        "dab": np.array([0.26814753, 0.57031375, 0.77642715]),
    },
    "he_default": {
        "hematoxylin": np.array([0.6500286, 0.704031, 0.2860126]),
        "eosin": np.array([0.07200929, 0.99003, 0.10567]),
        "residual": np.array([0.0, 0.0, 0.0]),
    },
}


def _normalize_vector(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def deconvolve_ruifrok(image):
    """
    Ruifrok & Johnston color deconvolution with standard H&E vectors.

    This is the classical method using predetermined stain vectors.

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
        Input RGB image

    Returns
    -------
    result : dict
        'hematoxylin': grayscale hematoxylin image
        'eosin': grayscale eosin image
        'residual': residual channel
        'normalized': reconstructed normalized image
    """
    # Build stain matrix M (3x3)
    h_vec = _normalize_vector(STAIN_VECTORS["ruifrok"]["hematoxylin"])
    e_vec = _normalize_vector(STAIN_VECTORS["ruifrok"]["eosin"])

    # Third vector: orthogonal to H and E
    residual_vec = np.cross(h_vec, e_vec)
    residual_vec = _normalize_vector(residual_vec)

    # Stain Color Appearance (SCA) matrix
    M = np.array([h_vec, e_vec, residual_vec])

    # Deconvolution matrix (inverse of M)
    D = np.linalg.inv(M)

    # Convert to Optical Density
    img_float = image.astype(np.float64) + 1
    od = -np.log10(img_float / 255.0)

    # Deconvolve: C = D * OD
    od_flat = od.reshape(-1, 3)
    concentrations = od_flat @ D.T
    concentrations = np.clip(concentrations, 0, None)

    h, w = image.shape[:2]

    # Extract individual channels
    h_conc = concentrations[:, 0].reshape(h, w)
    e_conc = concentrations[:, 1].reshape(h, w)
    r_conc = concentrations[:, 2].reshape(h, w)

    # Convert concentration back to image (transmittance)
    h_img = np.clip(255 * 10 ** (-h_conc), 0, 255).astype(np.uint8)
    e_img = np.clip(255 * 10 ** (-e_conc), 0, 255).astype(np.uint8)
    r_img = np.clip(255 * 10 ** (-r_conc), 0, 255).astype(np.uint8)

    # Invert so that stain = dark (for visualization as grayscale)
    h_gray = 255 - h_img
    e_gray = 255 - e_img

    # Reconstruct normalized RGB from H and E only
    od_recon = concentrations[:, :2] @ M[:2, :]
    normalized = np.clip(255 * 10 ** (-od_recon), 0, 255)
    normalized = normalized.reshape(image.shape).astype(np.uint8)

    return {
        "hematoxylin": h_gray,
        "eosin": e_gray,
        "residual": r_img,
        "normalized": normalized,
        "h_concentration": h_conc,
        "e_concentration": e_conc,
    }


def deconvolve_pca(image, beta=0.15):
    """
    PCA-based color deconvolution (adaptive stain vector estimation).

    Based on paper Method 2: estimates optimal stain vectors from the
    image itself using PCA/SVD in OD space.

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
        Input RGB image
    beta : float
        Threshold to remove transparent pixels in OD space

    Returns
    -------
    result : dict
        'hematoxylin': grayscale hematoxylin image
        'eosin': grayscale eosin image
        'normalized': reconstructed normalized image
        'stain_vectors': estimated stain vectors
    """
    h, w = image.shape[:2]

    # Convert to OD
    img_float = image.astype(np.float64) + 1
    od = -np.log10(img_float / 255.0)
    od_flat = od.reshape(-1, 3)

    # Remove transparent pixels
    mask = np.all(od_flat > beta, axis=1)
    od_filtered = od_flat[mask]

    if len(od_filtered) < 50:
        # Fallback to Ruifrok vectors
        return deconvolve_ruifrok(image)

    # PCA via SVD
    od_mean = od_filtered.mean(axis=0)
    od_centered = od_filtered - od_mean

    try:
        U, S, Vt = np.linalg.svd(od_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return deconvolve_ruifrok(image)

    # Project onto plane of first 2 principal components
    plane = Vt[:2, :]  # (2, 3)
    proj = od_filtered @ plane.T  # (N, 2)

    # Find angles to first direction
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    # Get robust extremes
    min_angle = np.percentile(angles, 1)
    max_angle = np.percentile(angles, 99)

    # Convert back to OD space
    v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
    v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

    # Ensure vectors point in the positive OD direction
    if np.sum(v1) < 0:
        v1 = -v1
    if np.sum(v2) < 0:
        v2 = -v2

    v1 = _normalize_vector(v1)
    v2 = _normalize_vector(v2)

    # Assign H vs E using cosine similarity to reference vectors
    ref_H = np.array([0.6500286, 0.704031, 0.2860126])
    ref_H = _normalize_vector(ref_H)
    sim_v1_H = abs(np.dot(v1, ref_H))
    sim_v2_H = abs(np.dot(v2, ref_H))

    if sim_v1_H >= sim_v2_H:
        stain_vectors = np.array([v1, v2])
    else:
        stain_vectors = np.array([v2, v1])

    # Third vector (residual)
    v3 = np.cross(stain_vectors[0], stain_vectors[1])
    v3 = _normalize_vector(v3)

    M = np.vstack([stain_vectors, [v3]])
    D = np.linalg.inv(M)

    # Deconvolve
    concentrations = od_flat @ D.T
    concentrations = np.clip(concentrations, 0, None)

    h_conc = concentrations[:, 0].reshape(h, w)
    e_conc = concentrations[:, 1].reshape(h, w)

    h_img = np.clip(255 * 10 ** (-h_conc), 0, 255).astype(np.uint8)
    e_img = np.clip(255 * 10 ** (-e_conc), 0, 255).astype(np.uint8)

    h_gray = 255 - h_img
    e_gray = 255 - e_img

    # Reconstruct
    od_recon = concentrations[:, :2] @ M[:2, :]
    normalized = np.clip(255 * 10 ** (-od_recon), 0, 255)
    normalized = normalized.reshape(image.shape).astype(np.uint8)

    return {
        "hematoxylin": h_gray,
        "eosin": e_gray,
        "normalized": normalized,
        "stain_vectors": stain_vectors,
        "h_concentration": h_conc,
        "e_concentration": e_conc,
    }


def deconvolve_image(image, method="ruifrok", **kwargs):
    """
    Apply stain deconvolution to an image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image
    method : str
        One of 'ruifrok' or 'pca'

    Returns
    -------
    result : dict
    """
    if method == "ruifrok":
        return deconvolve_ruifrok(image, **kwargs)
    elif method == "pca":
        return deconvolve_pca(image, **kwargs)
    else:
        raise ValueError(f"Unknown deconvolution method: {method}")
