"""
Intensity analyzer module.

Author:
    김찬희
"""

from os import PathLike

import librosa
import numpy as np
from speech_recognition import RequestError, UnknownValueError

from .audio_utils import (compute_spoken_audio, detect_onsets, load_audio,
                          transcribe_audio_file)
from .response import ErrorResponse, IntensityResponse, Response


def analyze_intensity(audio_file_path: str | PathLike) -> Response:
    """
    Analyze per-character intensity (volume in dB) from an audio file.

    This function performs:
    - Audio loading and transcription.
    - Silence removal to extract spoken-only audio.
    - Onset detection to estimate character boundaries.
    - RMS-to-dB conversion for each detected character segment.
    - Reconstruction of intensity values with spacing preserved.

    Parameters:
    - audio_file_path (str | PathLike): Path to the input audio file.

    Returns:
    - IntensityResponse: On success, containing per-character volume estimates.
    - ErrorResponse: On failure (e.g., silent audio, recognition error).
    """
    y, sampling_rate = load_audio(audio_file_path)

    # Transcribe audio
    try:
        text_full = transcribe_audio_file(
            audio_file_path, language='ko-KR').strip()
    except (UnknownValueError, RequestError) as e:
        return ErrorResponse(error_name=e.__class__.__name__,
                             error_details=e.args[0] if len(e.args) > 0 else "No details.")

    text_no_spaces = text_full.replace(" ", "")

    # Get non-silent audio frames
    spoken_audio = compute_spoken_audio(y, top_db=40)
    if spoken_audio.size == 0:
        return ErrorResponse(
            error_name="Cannot Remove Silent Intervals",
            error_details="An unknown error occured while removing silent intervals from audio frame."
        )

    hop_length = 512
    onset_strength, onset_frames = detect_onsets(
        spoken_audio, sampling_rate, hop_length=hop_length)

    if len(onset_frames) > len(text_no_spaces) - 1:
        sorted_onset_frames = sorted(
            onset_frames, key=lambda frame: onset_strength[frame], reverse=True)
        boundary_frames = sorted(sorted_onset_frames[:len(text_no_spaces)-1])
    else:
        boundary_frames = list(onset_frames)

    boundary_samples = librosa.frames_to_samples(boundary_frames)
    boundary_samples = np.asarray(boundary_samples, dtype=int).flatten()
    boundaries = np.concatenate(
        [np.array([0], dtype=boundary_samples.dtype), boundary_samples,
         np.array([len(spoken_audio)], dtype=boundary_samples.dtype)]
    )

    # estimate intensity per character
    char_volumes = []
    for i, char in enumerate(text_no_spaces):
        if i >= len(boundaries) - 1:
            break

        start_sample = int(boundaries[i])
        end_sample = int(boundaries[i+1])
        char_audio = spoken_audio[start_sample:end_sample]

        if char_audio.size > 0:
            rms = np.mean(librosa.feature.rms(y=char_audio).flatten())
            min_rms_threshold = 1e-4
            if rms < min_rms_threshold:
                volume = -100.0
            else:
                # Convert RMS amplitude to decibels (dB) using 20 * log10(rms) for audio intensity
                volume = 20 * np.log10(rms)

            char_volumes.append((char, round(volume, 2)))

    # reconstruct, add paddings between words
    words_list = text_full.split()
    response = IntensityResponse()
    text_cursor = 0
    for word in words_list:
        for char in word:
            if text_cursor < len(char_volumes):
                response.add_char_volume(
                    char=char_volumes[text_cursor][0],
                    volume=char_volumes[text_cursor][1])
                text_cursor += 1

        # add space
        if word != words_list[-1]:
            response.add_char_volume(char=' ', volume=-100.0)

    response.set_value("status", "SUCCESS")

    return response
