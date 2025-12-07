"""
Articulation analyzer module.

Author:
    김유환
"""

from os import PathLike

import Levenshtein
import librosa
from speech_recognition import RequestError, UnknownValueError

from .audio_utils import load_audio, transcribe_audio_file
from .response import ArticulationResponse, ErrorResponse, Response


def analyze_articulation(
        audio_file_path: str | PathLike,
        reference_text: str | None = None) -> Response:
    """
    Analyze articulation quality and fluency features from a speech audio file.

    This function extracts various speech metrics such as articulation rate,
    pause ratio, and (optionally) transcription accuracy using Levenshtein
    distance when a reference script is provided. Speech segments are detected
    by removing silence using `librosa.effects.split`, and the function uses a
    Korean speech recognizer (`ko-KR`) for transcription.

    Args:
        audio_file_path (str | PathLike):
            Path to the input audio file to analyze (e.g., `.wav`, `.flac`).
        reference_text (str, optional):
            The expected spoken script. If provided, the function computes
            accuracy and character error rate (CER) by comparing the reference
            text to the transcribed output.

    Returns:
        Response:
            An `ArticulationResponse` object containing:
                - status (str): Processing result status.
                - duration (float): Total duration of the audio in seconds.
                - articulation_rate (float): Syllables per second of spoken audio.
                - pause_ratio (float): Ratio of silence duration to total duration.
                - accuracy_score (float): Percentage accuracy vs. reference text.
                - char_error_rate (float): CER based on Levenshtein distance.
                - transcription (str): Recognized text from the audio.

    Raises:
        UnknownValueError:
            Raised when the audio cannot be reliably transcribed.
        RequestError:
            Raised when the transcription service encounters a request failure.

    Notes:
        - Articulation rate is estimated by counting characters in the
          transcribed Korean text (spaces removed). This is accurate for Korean
          because each Hangul syllable block corresponds to one syllable.
          Results will be less accurate for languages where characters do not
          map 1:1 to syllables.
        - Silence detection uses librosa's energy-based splitting with `top_db=40`.
    """
    y, sampling_rate = load_audio(audio_file_path)

    # Get length of spoken audio
    intervals = librosa.effects.split(y, top_db=40)
    speech_duration = 0
    for start, end in intervals:
        speech_duration += (end - start) / sampling_rate

    total_duration = librosa.get_duration(y=y, sr=sampling_rate)
    pause_duration = total_duration - speech_duration
    pause_ratio = pause_duration / total_duration if total_duration > 0 else 0

    # Transcribe audio
    try:
        transcribed_text = transcribe_audio_file(
            audio_file_path, language='ko-KR').strip()
    except (UnknownValueError, RequestError) as e:
        return ErrorResponse(error_name=e.__class__.__name__,
                             error_details=e.args[0] if len(e.args) > 0 else "No details.")

    # Calculate articulation rate (syllables per second)
    # This heuristic works well for Korean but is inaccurate for languages
    # where characters do not map 1:1 to syllables (e.g., English).
    num_syllables = len(transcribed_text.replace(" ", ""))
    articulation_rate = num_syllables / speech_duration if speech_duration > 0 else 0

    # Evaluate accuracy (if reference_text is given)
    if reference_text:
        ref_clean = reference_text.strip()
        hyp_clean = transcribed_text.strip()

        distance = Levenshtein.distance(ref_clean, hyp_clean)
        length = len(ref_clean)

        cer = distance / length if length > 0 else 0
        accuracy = max(0, (1 - cer) * 100)
    else:
        cer = 0
        accuracy = 0

    response = ArticulationResponse(
        status="SUCCESS",
        duration=total_duration,
        articulation_rate=articulation_rate,
        pause_ratio=pause_ratio,
        accuracy_score=accuracy,
        char_error_rate=cer,
        transcription=transcribed_text
    )

    return response
