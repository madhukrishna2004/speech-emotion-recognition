"""
Emotion Classification Model for Speech Emotion Recognition (SER)
High-accuracy pipeline using CNN + BiLSTM + Temporal Attention.

- Expects variable-length sequences padded in data_loader to (N, T, F)
- Scales features with a StandardScaler fit over all time steps
- Uses Conv1D -> BiLSTM -> Attention -> Dense classifier
- Strong regularization, LR scheduling, early stopping, class weights
- Saves: best model (.keras), final model (.keras), scaler.pkl, classes.pkl
"""

import os
import json
import math
import pickle
import logging
import warnings
import numpy as np
from typing import Tuple, List, Optional, Dict

# TF / Keras
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, MaxPooling1D,
    Dropout, Bidirectional, LSTM, Dense, Layer, Softmax
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Project
from ser.utils import get_logger, read_audio_file
from ser.config import Config
from ser.data import load_data
from ser.features.feature_extractor import extended_extract_feature

logger: logging.Logger = get_logger(__name__)

# -----------------------------
# Utilities
# -----------------------------
def _ensure_models_folder() -> str:
    folder = Config.MODELS_CONFIG["models_folder"]
    os.makedirs(folder, exist_ok=True)
    return folder

def _save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def _get_class_names() -> List[str]:
    """
    Assumes LabelEncoder on string labels during data prep.
    """
    return sorted(list(set(Config.EMOTIONS.values())))

def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    return {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

def _fit_scaler_3d(X: np.ndarray):
    """
    Fit a StandardScaler on (N, T, F) by flattening time dimension.
    """
    from sklearn.preprocessing import StandardScaler
    N, T, F = X.shape
    scaler = StandardScaler()
    scaler.fit(X.reshape(N * T, F))
    return scaler

def _transform_with_scaler_3d(X: np.ndarray, scaler) -> np.ndarray:
    """
    Transform (N, T, F) using a scaler fit on flattened (N*T, F).
    """
    N, T, F = X.shape
    Xf = scaler.transform(X.reshape(N * T, F)).reshape(N, T, F)
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Xf

def _pad_sequence_2d(seq: np.ndarray, max_T: int, max_F: int) -> np.ndarray:
    """
    Pad or truncate a 2D (T, F) feature sequence to (max_T, max_F).
    """
    T = min(seq.shape[0], max_T)
    F = min(seq.shape[1], max_F)
    out = np.zeros((max_T, max_F), dtype=np.float32)
    out[:T, :F] = seq[:T, :F]
    return out

def _pad_1d(vec: np.ndarray, target_F: int) -> np.ndarray:
    """
    Pad or truncate a 1D frame vector to length target_F.
    """
    v = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if v.ndim != 1:
        v = v.reshape(-1)
    if v.shape[0] == target_F:
        return v
    if v.shape[0] < target_F:
        out = np.zeros((target_F,), dtype=np.float32)
        out[: v.shape[0]] = v
        return out
    # truncate
    return v[:target_F]

def _safe_input_tf(model: Model) -> Tuple[int, int]:
    """
    Safely read (T, F) from model.input_shape which can be:
    (None, T, F) or a list/tuple for multi-input.
    """
    in_shape = model.input_shape
    if isinstance(in_shape, (list, tuple)) and isinstance(in_shape[0], (list, tuple)):
        in_shape = in_shape[0]
    # in_shape like (None, T, F)
    T_train_max = int(in_shape[-2])
    F_train_max = int(in_shape[-1])
    return T_train_max, F_train_max

# -----------------------------
# Temporal Attention layer
# -----------------------------
class TemporalAttention(Layer):
    """
    Simple temporal attention: produces a weighted sum (context) over time.
    Input:  (batch, time, features)
    Output: (batch, features)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = None
        self.softmax = Softmax(axis=1)

    def build(self, input_shape):
        self.score_dense = Dense(1, activation="tanh")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        scores = self.score_dense(inputs)      # (B, T, 1)
        weights = self.softmax(scores)         # (B, T, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)  # (B, F)
        return context

    def get_config(self):
        return super().get_config()

# -----------------------------
# Model
# -----------------------------
def build_cnn_bilstm_attention(input_shape: Tuple[int, int], num_classes: int) -> Model:
    """
    input_shape = (T, F)
    """
    inp = Input(shape=input_shape)

    x = Conv1D(256, 5, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # Temporal attention over time dimension
    x = TemporalAttention()(x)  # (B, features)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------
# Train
# -----------------------------
def train_model() -> None:
    logger.info("Loading dataset...")
    data_split = load_data(test_size=0.15)
    if data_split is None:
        raise RuntimeError("Dataset could not be loaded.")
    X_train, X_val, y_train, y_val = data_split

    # Expect (N, T, F)
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (N, T, F), got {X_train.shape}")
    if X_val.ndim != 3:
        raise ValueError(f"X_val must be 3D (N, T, F), got {X_val.shape}")

    logger.info(f"Dataset loaded. Train: {X_train.shape}, Val: {X_val.shape}")

    # Scale over feature dim using all time steps
    scaler = _fit_scaler_3d(X_train)
    X_train = _transform_with_scaler_3d(X_train, scaler)
    X_val   = _transform_with_scaler_3d(X_val, scaler)

    # Labels
    y_train = np.asarray(y_train).astype("int64")
    y_val   = np.asarray(y_val).astype("int64")

    num_classes = int(np.max([y_train.max(), y_val.max()])) + 1
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_cnn_bilstm_attention(input_shape, num_classes)
    model.summary(print_fn=lambda s: logger.info(s))

    # Callbacks & paths
    models_dir = _ensure_models_folder()
    best_path   = os.path.join(models_dir, "ser_cnn_bilstm_att_best.keras")
    final_path  = os.path.join(models_dir, "ser_cnn_bilstm_att_final.keras")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    classes_path= os.path.join(models_dir, "classes.pkl")

    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    class_weights = _compute_class_weights(y_train)
    logger.info(f"Using class weights: {class_weights}")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        shuffle=True,
        callbacks=cbs,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final model and artifacts
    model.save(final_path)
    _save_pickle(scaler, scaler_path)
    classes = _get_class_names()
    _save_pickle(classes, classes_path)

    logger.info(f"Best model saved to: {best_path}")
    logger.info(f"Final model saved to: {final_path}")
    logger.info(f"Scaler saved to: {scaler_path}")
    logger.info(f"Classes saved to: {classes_path}")

# -----------------------------
# Load
# -----------------------------
def load_ser_model(best: bool = True) -> Model:
    models_dir = _ensure_models_folder()
    path = os.path.join(
        models_dir,
        "ser_cnn_bilstm_att_best.keras" if best else "ser_cnn_bilstm_att_final.keras"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trained model not found at {path}")
    model = load_model(path, custom_objects={"TemporalAttention": TemporalAttention})
    logger.info(f"Loaded SER model from {path}")
    return model

def load_scaler_and_classes():
    models_dir = _ensure_models_folder()
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    classes_path = os.path.join(models_dir, "classes.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(classes_path):
        raise FileNotFoundError("Scaler or classes file missing. Train the model first.")
    scaler = _load_pickle(scaler_path)
    classes = _load_pickle(classes_path)
    return scaler, classes

# -----------------------------
# Predict (utterance-level)
# -----------------------------
def predict_emotions(file: str) -> List[Tuple[str, float, float]]:
    """
    Predict the dominant emotion over the whole file.
    Returns: List of (emotion_name, start_time, end_time)
    """
    logger.info("Starting emotion prediction...")
    model = load_ser_model(best=True)
    scaler, class_names = load_scaler_and_classes()

    # Extract frame-level features (1D vectors; lengths may differ!)
    frames = extended_extract_feature(file, frame_size=3, frame_stride=1, augment=False)
    if not frames:
        raise RuntimeError("No features extracted from audio.")

    # Read expected (T, F) from model
    T_train_max, F_train_max = _safe_input_tf(model)  # (T, F)

    # 1) Normalize each frame vector length to F_train_max
    padded_frames = [_pad_1d(f, F_train_max) for f in frames]  # list of (F_train_max,)
    # 2) Stack into (T, F_train_max)
    seq = np.stack(padded_frames, axis=0)  # (T, F)
    # 3) Pad time to T_train_max
    seq = _pad_sequence_2d(seq, max_T=T_train_max, max_F=F_train_max)  # (T, F)

    # 4) Add batch dim -> (1, T, F)
    seq = seq[None, ...]
    # 5) Scale exactly like training
    seq = _transform_with_scaler_3d(seq, scaler)  # keeps (1, T, F)

    # Predict
    probs = model.predict(seq, verbose=0)[0]  # (C,)
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    # Duration for timestamp
    audio, sr = read_audio_file(file)
    duration = float(len(audio)) / float(sr)

    return [(pred_name, 0.0, duration)]

# -----------------------------
# Evaluate helper (optional)
# -----------------------------
def evaluate_on_validation() -> Dict[str, float]:
    """
    Quick evaluation of the best model on a fresh split.
    """
    data_split = load_data(test_size=0.15)
    if data_split is None:
        raise RuntimeError("Dataset could not be loaded.")
    X_train, X_val, y_train, y_val = data_split

    scaler = _fit_scaler_3d(X_train)
    X_val = _transform_with_scaler_3d(X_val, scaler)

    model = load_ser_model(best=True)
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation — loss: {loss:.4f}, accuracy: {acc*100:.2f}%")
    return {"val_loss": float(loss), "val_accuracy": float(acc)}
