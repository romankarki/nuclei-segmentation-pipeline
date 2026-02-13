"""
Data loading utilities for MoNuSeg and TNBC datasets.

Handles loading images and corresponding masks from both datasets,
with train/test splitting and batch iteration support.
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_image(path):
    """Load an image as a numpy RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask(path):
    """Load a mask as a binary numpy array (0 or 1)."""
    mask = Image.open(path).convert("L")
    mask = np.array(mask)
    # Binarize: anything > 0 is foreground
    return (mask > 0).astype(np.uint8)


def get_dataset_paths(data_root, dataset_name="monuseg"):
    """
    Get matched lists of image and mask file paths for a dataset.

    Parameters
    ----------
    data_root : str
        Root directory containing the data folders (e.g., '../../data')
    dataset_name : str
        One of 'monuseg' or 'tnbc'

    Returns
    -------
    image_paths : list of str
    mask_paths : list of str
    """
    img_dir = os.path.join(data_root, f"{dataset_name}_images")
    mask_dir = os.path.join(data_root, f"{dataset_name}_masks")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )

    # Verify filenames match
    for ip, mp in zip(image_paths, mask_paths):
        img_name = os.path.basename(ip)
        mask_name = os.path.basename(mp)
        if img_name != mask_name:
            raise ValueError(f"Filename mismatch: {img_name} vs {mask_name}")

    return image_paths, mask_paths


def load_dataset(data_root, dataset_name="monuseg", max_images=None):
    """
    Load all images and masks for a dataset.

    Parameters
    ----------
    data_root : str
        Root directory containing the data folders
    dataset_name : str
        One of 'monuseg' or 'tnbc'
    max_images : int or None
        Limit number of images to load (for quick testing)

    Returns
    -------
    images : list of np.ndarray (H, W, 3)
    masks : list of np.ndarray (H, W)
    names : list of str
    """
    image_paths, mask_paths = get_dataset_paths(data_root, dataset_name)

    if max_images is not None:
        image_paths = image_paths[:max_images]
        mask_paths = mask_paths[:max_images]

    images = []
    masks = []
    names = []

    for ip, mp in zip(image_paths, mask_paths):
        images.append(load_image(ip))
        masks.append(load_mask(mp))
        names.append(os.path.splitext(os.path.basename(ip))[0])

    return images, masks, names


def split_dataset(images, masks, names, test_size=0.3, random_state=42):
    """
    Split dataset into train and test sets.

    Returns
    -------
    dict with keys: 'train_images', 'train_masks', 'train_names',
                    'test_images', 'test_masks', 'test_names'
    """
    indices = list(range(len(images)))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    return {
        "train_images": [images[i] for i in train_idx],
        "train_masks": [masks[i] for i in train_idx],
        "train_names": [names[i] for i in train_idx],
        "test_images": [images[i] for i in test_idx],
        "test_masks": [masks[i] for i in test_idx],
        "test_names": [names[i] for i in test_idx],
    }


def load_all_datasets(data_root, max_images=None):
    """
    Load both MoNuSeg and TNBC datasets.

    Returns
    -------
    dict mapping dataset name to (images, masks, names)
    """
    datasets = {}
    for name in ["monuseg", "monuseg_full", "tnbc"]:
        try:
            images, masks, names = load_dataset(data_root, name, max_images)
            datasets[name] = {"images": images, "masks": masks, "names": names}
            print(f"Loaded {name}: {len(images)} images, resolution {images[0].shape[1]}x{images[0].shape[0]}")
        except FileNotFoundError as e:
            print(f"Warning: Could not load {name}: {e}")
    return datasets
