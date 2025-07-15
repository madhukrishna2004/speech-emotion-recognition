"""
Transcript Extraction for Speech Emotion Recognition (SER) Tool

This module provides functions to extract transcripts from audio files using the Whisper
model. It includes functions to load the Whisper model and extract and format the transcript.

Functions:
    - load_whisper_model: Loads the Whisper model specified in the configuration.
    - extract_transcript: Extracts the transcript from an audio file using the Whisper model.
    - format_transcript: Formats the transcript into a list of tuples containing the word,
                         start time, and end time.
"""

import logging
from typing import Tuple, List, Any, Optional
import warnings

from halo import Halo
import whisper
from whisper.model import Whisper

from ser.utils import get_logger
from ser.config import Config

logger: logging.Logger = get_logger(__name__)


def load_whisper_model() -> Whisper:
    """
    Loads the Whisper model specified in the configuration.

    Returns:
        whisper.Whisper: Loaded Whisper model.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="whisper")
            model: Whisper = whisper.load_model(
                name=Config.MODELS_CONFIG["whisper_model"]["name"]
            )
        return model
    except Exception as err:
        logger.error(f"Failed to load Whisper model: {err}", exc_info=True)
        raise


def extract_transcript(
    file_path: str, language: str = Config.DEFAULT_LANGUAGE
) -> List[Tuple[str, float, float]]:
    """
    Extracts the transcript from an audio file using the Whisper model.

    Arguments:
        file_path (str): Path to the audio file.
        language (str): Language of the audio.

    Returns:
        list: List of tuples (word, start_time, end_time).
    """
    try:
        return _extract_transcript(file_path, language)
    except Exception as err:
        logger.error(f"Failed to extract transcript: {err}", exc_info=True)
        raise


def _extract_transcript(
    file_path: str, language: str
) -> List[Tuple[str, float, float]]:
    with Halo(text="Loading the Whisper model...", spinner="dots", text_color="green"):
        model: Whisper = load_whisper_model()

    logger.info("Whisper model loaded successfully.")

    try:
        with Halo(text="Transcribing the audio file...", spinner="dots", text_color="green"):
            result = __transcribe_file(model, language, file_path)
        logger.info("Audio file transcription process completed.")

        if result and "segments" in result:
            formatted_transcript = format_transcript(result)
        else:
            logger.info("Transcript is empty.")
            return [("", 0, 0)]

        logger.debug("Transcript output formatted successfully.")
    except Exception as err:
        logger.error(f"Error generating the transcript: {err}", exc_info=True)
        raise

    logger.info("Transcript extraction process completed successfully.")
    return formatted_transcript


def __transcribe_file(
    model: Whisper, language: str, file_path: str
) -> Optional[dict]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = model.transcribe(
                audio=file_path,
                language=language,
                verbose=False,
                word_timestamps=True
            )
    except Exception as err:
        logger.error(f"Error processing speech extraction: {err}", exc_info=True)
        return None
    return result


def format_transcript(result: dict) -> List[Tuple[str, float, float]]:
    """
    Formats the transcript into a list of tuples containing the word,
    start time, and end time.

    Args:
        result (dict): The transcript result.

    Returns:
        List[Tuple[str, float, float]]: Formatted transcript with timestamps.
    """
    try:
        words: List[Tuple[str, float, float]] = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info["word"]
                start = word_info["start"]
                end = word_info["end"]
                words.append((word, start, end))
    except Exception as err:
        logger.error(f"Error extracting words from result: {err}", exc_info=True)
        raise

    return words if words else [("", 0, 0)]
