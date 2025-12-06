"""
Speechrate analyzer module.

Author:
    김찬희
"""

import time
from os import PathLike

import speech_recognition as sr

from audio_utils import compute_spoken_audio, load_audio, transcribe_audio_file
from response import ErrorResponse, Response, SpeechrateResponse


def analyze_speechrate(audio_file_path: str | PathLike) -> Response:
    """
    Analyze speech rate (WPM & CPS) from an audio file.

    This function performs:
    - Audio loading and preprocessing.
    - Speech recognition to obtain transcription.
    - Silence removal to extract spoken-only audio.
    - Calculation of:
        - Words per minute (WPM)
        - Characters per second (CPS)
        - Total spoken duration (seconds)
        - Word and character counts

    Parameters:
    - audio_file_path (str | PathLike): Path to the input audio file.

    Returns:
    - SpeechrateResponse: On success, containing speech-rate metrics.
    - ErrorResponse: On failure (e.g., recognition error, silence removal error).
    """
    start_time = time.time()

    y, sampling_rate = load_audio(audio_file_path)

    # Transcribe using the shared helper (speech_recognition)
    try:
        text_full = transcribe_audio_file(
            audio_file_path, language='ko-KR').strip()
    except (sr.UnknownValueError, sr.RequestError) as e:
        return ErrorResponse(error_name=e.__class__.__name__,
                             error_details=e.args[0] if len(e.args) > 0 else "No details.")

    # Get non-silent audio frames
    spoken_audio = compute_spoken_audio(y, top_db=40)
    if spoken_audio.size == 0:
        return ErrorResponse(
            error_name="Cannot Remove Silent Intervals",
            error_details="An unknown error occured while removing silent intervals from audio frame."
        )

    # Load audio and compute speech-only audio
    total_speech_time_seconds = round(
        len(spoken_audio) / sampling_rate, 2) if spoken_audio.size > 0 else 0.0
    if total_speech_time_seconds <= 0:
        return ErrorResponse(
            error_name="Invalid Total Speech Time",
            error_details=f"Total speech time (seconds) is not a positive number: {total_speech_time_seconds}"
        )

    end_time = time.time()
    analysis_time = end_time - start_time

    total_words = len(text_full.split())
    total_characters = len(text_full.replace(" ", ""))

    wpm = (total_words / total_speech_time_seconds) * 60
    cps = total_characters / total_speech_time_seconds

    response = SpeechrateResponse(status="SUCCESS",
                                  wpm=wpm,
                                  cps=cps,
                                  total_speech_time=total_speech_time_seconds,
                                  total_words=total_words,
                                  total_characters=total_characters,
                                  analysis_time=analysis_time,
                                  transcript=text_full)

    return response
