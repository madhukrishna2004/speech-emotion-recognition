import numpy as np
import librosa
import soundfile as sf
import os
import logging
import warnings
from typing import List

from ser.utils import get_logger, read_audio_file
from ser.config import Config

logger: logging.Logger = get_logger(__name__)

os.makedirs(Config.TMP_FOLDER, exist_ok=True)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


# -----------------------
# Audio Augmentation Helpers
# -----------------------
def augment_audio(audio, sr):
    """Apply random augmentations to improve generalization."""
    # Random pitch shift
    if np.random.rand() < 0.3:
        steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr, n_steps=steps)

    # Random time stretch
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate)

    # Random noise injection
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.005, len(audio))
        audio = audio + noise

    return audio


# -----------------------
# SpecAugment for Mel
# -----------------------
def spec_augment(mel_spectrogram, num_masks=2, freq_masking=8, time_masking=8):
    """Apply frequency and time masking to mel spectrogram."""
    mel = mel_spectrogram.copy()
    num_mel_channels, num_frames = mel.shape

    for _ in range(num_masks):
        # Frequency masking
        f = np.random.randint(0, freq_masking)
        f0 = np.random.randint(0, num_mel_channels - f)
        mel[f0:f0 + f, :] = 0

        # Time masking
        t = np.random.randint(0, time_masking)
        t0 = np.random.randint(0, num_frames - t)
        mel[:, t0:t0 + t] = 0

    return mel


# -----------------------
# Feature Extraction
# -----------------------
def extract_feature(file: str, augment: bool = False) -> np.ndarray:
    """
    Extracts Mel-spectrogram features with optional augmentation and normalization.
    Returns 2D array suitable for CNN input.
    """
    try:
        audio, sr = read_audio_file(file)
    except Exception as err:
        logger.error(f"Error reading file {file}: {err}")
        return None

    if augment:
        audio = augment_audio(audio, sr)

    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if augment and np.random.rand() < 0.5:
        mel_spec_db = spec_augment(mel_spec_db)

    # Normalize per sample
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-6)

    return mel_spec_db.astype(np.float32)


def extended_extract_feature(audiofile: str, frame_size: int = 3, frame_stride: int = 1, augment: bool = False) -> List[np.ndarray]:
    """Extract features over sliding frames with augmentation."""
    temp_filename = f"{Config.TMP_FOLDER}/temp.wav"
    features_list = []
    audio, sr = read_audio_file(audiofile)
    frame_length = int(frame_size * sr)
    frame_step = int(frame_stride * sr)
    num_frames = int(np.ceil(len(audio) / frame_step))

    for frame in range(num_frames):
        start = frame * frame_step
        end = min(start + frame_length, len(audio))
        frame_data = audio[start:end]

        sf.write(temp_filename, frame_data, sr)
        feat = extract_feature(temp_filename, augment=augment)
        if feat is not None:
            features_list.append(feat)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return features_list
