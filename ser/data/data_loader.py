"""
Data Loader for Speech Emotion Recognition (SER) Tool
Upgraded version with:
- Class balancing
- Data augmentation (noise, jitter)
- Feature normalization
- Safety checks for shape/dtype consistency
- Proper padding for variable-length 2D features with dimension consistency
"""

import os
import glob
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ser.utils import get_logger
from ser.features.feature_extractor import extract_feature
from ser.config import Config

logger: logging.Logger = get_logger(__name__)

# -----------------------------
# Process Single File
# -----------------------------
def process_file(file: str, observed_emotions: List[str]) -> Optional[Tuple[np.ndarray, str]]:
    """Extracts features for one audio file if emotion is valid."""
    try:
        file_name: str = os.path.basename(file)
        emotion: Optional[str] = Config.EMOTIONS.get(file_name.split("-")[2])

        if not emotion or emotion not in observed_emotions:
            return None

        features: np.ndarray = extract_feature(file)

        # Skip if empty or not numeric
        if features is None or features.size == 0 or not np.issubdtype(features.dtype, np.number):
            logger.warning(f"Skipping file {file} due to invalid features.")
            return None

        # Ensure 2D shape (time, features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        return (features, emotion)

    except Exception as e:
        logger.error(f"Failed to process file {file}: {e}")
        return None

# -----------------------------
# Padding for variable-length 2D features with consistent frequency bins
# -----------------------------
def pad_features(features: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pads a list of 2D features (time, feat_dim) to the same length and consistent frequency bins."""
    if not features:
        return np.array([]), {}
    
    # Find maximum time steps and minimum frequency bins for consistency
    max_time_steps = max(f.shape[0] for f in features)
    min_freq_bins = min(f.shape[1] for f in features)
    
    logger.info(f"Padding features to max_time={max_time_steps}, min_freq={min_freq_bins}")
    
    # Log dimension distribution for debugging
    time_lengths = [f.shape[0] for f in features]
    freq_dims = [f.shape[1] for f in features]
    
    logger.debug(f"Time steps distribution: min={min(time_lengths)}, max={max(time_lengths)}, avg={np.mean(time_lengths):.1f}")
    logger.debug(f"Frequency bins distribution: min={min(freq_dims)}, max={max(freq_dims)}, avg={np.mean(freq_dims):.1f}")
    
    # Create padded array with consistent dimensions
    padded = np.zeros((len(features), max_time_steps, min_freq_bins), dtype=np.float32)
    
    for i, f in enumerate(features):
        time_steps = min(f.shape[0], max_time_steps)
        freq_bins = min(f.shape[1], min_freq_bins)
        padded[i, :time_steps, :freq_bins] = f[:time_steps, :freq_bins]
    
    padding_info = {
        'max_time_steps': max_time_steps,
        'min_freq_bins': min_freq_bins,
        'original_time_lengths': time_lengths,
        'original_freq_dims': freq_dims
    }
    
    return padded, padding_info

# -----------------------------
# Validate feature consistency
# -----------------------------
def validate_features(features: List[np.ndarray]) -> bool:
    """Check if all features have consistent frequency dimensions."""
    if not features:
        return False
    
    first_shape = features[0].shape
    for i, f in enumerate(features[1:], 1):
        if f.shape[1] != first_shape[1]:
            logger.warning(f"Inconsistent frequency dimensions: "
                          f"File 0 has {first_shape[1]} bins, File {i} has {f.shape[1]} bins")
            return False
    return True

# -----------------------------
# Load Data with Balancing + Augmentation + Padding
# -----------------------------
def load_data(test_size: float = 0.2) -> Optional[List]:
    observed_emotions: List[str] = list(Config.EMOTIONS.values())

    data_path_pattern: str = (
        f"{Config.DATASET['folder']}/"
        f"{Config.DATASET['subfolder_prefix']}/"
        f"{Config.DATASET['extension']}"
    )
    files: List[str] = glob.glob(data_path_pattern)
    logger.info(f"Found {len(files)} audio files to process.")

    # Parallel processing
    with mp.Pool(int(Config.MODELS_CONFIG["num_cores"])) as pool:
        data = pool.map(partial(process_file, observed_emotions=observed_emotions), files)

    # Remove invalid entries
    data = [item for item in data if item is not None]
    if not data:
        logger.warning("No valid data processed.")
        return None

    features, labels = zip(*data)
    features = list(features)
    labels = np.array(labels)

    # Validate feature consistency
    if not validate_features(features):
        logger.warning("Features have inconsistent dimensions. Using minimum frequency bins for padding.")

    # Pad sequences to consistent dimensions
    features, padding_info = pad_features(features)
    logger.info(f"Padded features shape: {features.shape}")

    # -----------------------------
    # Balance classes
    # -----------------------------
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = min(counts) if len(counts) > 0 else 0
    
    if min_count == 0:
        logger.error("No samples found for some emotions. Check your dataset.")
        return None

    balanced_features, balanced_labels = [], []

    for emo in observed_emotions:
        emo_indices = np.where(labels == emo)[0]
        if len(emo_indices) == 0:
            continue
            
        emo_features = features[emo_indices]
        emo_labels = labels[emo_indices]

        if len(emo_features) > min_count:
            idx = np.random.choice(len(emo_features), min_count, replace=False)
            balanced_features.extend(emo_features[idx])
            balanced_labels.extend(emo_labels[idx])
        else:
            balanced_features.extend(emo_features)
            balanced_labels.extend(emo_labels)

    features = np.array(balanced_features)
    labels = np.array(balanced_labels)

    # -----------------------------
    # Light augmentation (jitter)
    # -----------------------------
    if len(features) > 0:
        aug_features, aug_labels = [], []
        for feat, label in zip(features, labels):
            jittered = feat + np.random.normal(0, 0.01, feat.shape)
            aug_features.append(jittered)
            aug_labels.append(label)

        features = np.vstack([features, np.array(aug_features)])
        labels = np.hstack([labels, np.array(aug_labels)])

    # -----------------------------
    # Normalize & clean
    # -----------------------------
    if len(features) > 0:
        scaler = StandardScaler()
        # reshape for scaler: (samples*timesteps, feat_dim)
        reshaped = features.reshape(-1, features.shape[-1])
        scaled = scaler.fit_transform(reshaped)
        features = scaled.reshape(features.shape)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # -----------------------------
    # Encode labels
    # -----------------------------
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    logger.info(f"Classes: {list(le.classes_)}")
    logger.info(f"Final features shape: {features.shape}, Labels shape: {labels.shape}")

    # -----------------------------
    # Return train/test split
    # -----------------------------
    if len(features) == 0:
        logger.error("No features available after processing.")
        return None

    return train_test_split(
        features, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )