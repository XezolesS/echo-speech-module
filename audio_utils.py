"""
Common audio utilities for the echo-speech-module.

Provides small helpers to load audio, remove silent segments (spoken audio),
perform STT via `speech_recognition` (using a normalized temporary WAV), and
detect onsets. These centralize shared logic used by multiple analysis
modules to keep the codebase DRY.
"""
from __future__ import annotations

import os
import tempfile
from os import PathLike
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import speech_recognition as sr


def load_audio(audio_file_path: str | PathLike) -> Tuple[np.ndarray, int | float]:
    """Load audio at native sampling rate.

    Raises FileNotFoundError if `audio_file_path` does not exist. Other
    exceptions from `librosa.load` are propagated to the caller.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(audio_file_path)

    y, sampling_rate = librosa.load(audio_file_path, sr=None)
    return y, sampling_rate


def compute_spoken_audio(y: np.ndarray, top_db: int = 40) -> np.ndarray:
    """Return a concatenated audio array containing only non-silent frames.

    If no voiced intervals are found an empty numpy array is returned.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    if intervals is None or len(intervals) == 0:
        return np.array([], dtype=y.dtype)

    return np.concatenate([y[s:e] for s, e in intervals])


def transcribe_audio_file(
        audio_file_path: str | PathLike,
        language: str = "ko-KR") -> str:
    """Transcribe audio using `speech_recognition` + Google Web Speech.

    The audio is loaded and normalized, written to a temporary WAV and fed to
    the recognizer. This function re-raises the same exceptions from
    `speech_recognition` so callers can decide how to handle them.
    """
    y, sr_native = load_audio(audio_file_path)
    r = sr.Recognizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        normalized_y = librosa.util.normalize(y)
        temp_full_wav_path = os.path.join(temp_dir, "full_audio.wav")
        sf.write(temp_full_wav_path, normalized_y, sr_native)

        with sr.AudioFile(temp_full_wav_path) as source:
            r.adjust_for_ambient_noise(source, duration=1)

        with sr.AudioFile(temp_full_wav_path) as source:
            audio_data = r.record(source)
            # Let caller handle UnknownValueError / RequestError
            return r.recognize_google(audio_data, language=language)


def detect_onsets(
        spoken_audio: np.ndarray,
        sampling_rate: int | float,
        hop_length: int = 512,
        backtrack: bool = True):
    """Compute onset strength envelope and detect onset frames.

    Returns a tuple: (onset_envelope, onset_frames)
    """
    onset_env = librosa.onset.onset_strength(
        y=spoken_audio, sr=sampling_rate, hop_length=hop_length
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sampling_rate,
        units="frames", backtrack=backtrack, hop_length=hop_length
    )

    return onset_env, onset_frames
