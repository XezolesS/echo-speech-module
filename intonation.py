"""
Intonation analyzer module.

This module computes character-aligned prosodic summaries (duration, F0)
and a pitch contour mapped to a character axis. It relies on shared
helpers from `audio_utils` and the `analyze_intensity` result which
provides per-character volumes and the character sequence.

Author:
    김유환
"""

from os import PathLike
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from audio_utils import compute_spoken_audio, detect_onsets, load_audio
from intensity import analyze_intensity
from response import ErrorResponse, IntonationResponse, Response


def select_boundaries_for_chars(
    spoken_len: int,
    onset_envelope: np.ndarray,
    onset_frames: np.ndarray,
    hop_length: int,
    target_char_count: int,
) -> np.ndarray:
    """
    Generate segment boundaries aligned to the number of non-space characters.

    The function attempts to use detected onset frames as boundaries. If
    there are more onsets than needed, it keeps only the strongest ones
    (based on onset energy). If fewer are available, it falls back to
    evenly spaced boundaries.

    Args:
        spoken_len (int):
            Length of the spoken audio in samples.
        onset_envelope (np.ndarray):
            Onset strength envelope (one value per frame).
        onset_frames (np.ndarray):
            Indices of detected onset frames.
        hop_length (int):
            Hop length used for converting frames to sample indices.
        target_char_count (int):
            Number of non-space characters to segment.

    Returns:
        np.ndarray:
            A 1-D integer array of sample indices with length
            ``target_char_count + 1`` defining segment boundaries.
    """
    if target_char_count <= 0:
        return np.array([0, spoken_len])

    # Convert onset frames to sample indices
    onset_strength = onset_envelope
    if onset_frames.size > target_char_count - 1:
        # Select top N-1 strongest onsets, then sort
        sorted_frames = sorted(
            onset_frames, key=lambda fr: onset_strength[fr], reverse=True
        )[: (target_char_count - 1)]
        boundary_frames = np.array(sorted(sorted_frames))
        boundary_samples = librosa.frames_to_samples(
            boundary_frames, hop_length=hop_length)
    else:
        # Use all available onset frames
        boundary_samples = librosa.frames_to_samples(
            onset_frames, hop_length=hop_length)

    # Add start/end boundaries
    boundaries = np.concatenate(
        [np.array([0]), boundary_samples.astype(int), np.array([spoken_len])]
    )

    # If boundary count mismatches, fall back to uniform partitioning
    needed = target_char_count + 1
    if len(boundaries) != needed:
        boundaries = np.linspace(0, spoken_len, num=needed, dtype=int)

    return boundaries


def pyin_f0(
    y: np.ndarray,
    sampling_rate: int | float,
    frame_length: int = 2048,
    hop_length: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frame-wise F0 (pitch) using librosa.pyin.

    Unvoiced frames are returned as NaN so downstream median or smoothing
    operations can naturally ignore them.

    Args:
        y (np.ndarray):
            Audio signal.
        sampling_rate (int | float):
            Sampling rate of the audio.
        frame_length (int):
            Frame length for PYIN.
        hop_length (int):
            Hop length for PYIN.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - times (float seconds): one time stamp per frame.
            - f0_hz (float Hz): detected F0 values (NaN for unvoiced).
    """
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sampling_rate, hop_length=hop_length)

    return times, f0


def summarize_char_level_prosody(
    spoken_audio: np.ndarray,
    sampling_rate: int | float,
    chars: List[str],
    volumes: List[float],
    hop_length: int = 256,
) -> Dict[str, Any]:
    """
    Produce character-aligned prosodic summaries (duration, F0, volume).

    This function:
      1. Detects onset boundaries and aligns them to the number of non-space
         characters.
      2. Computes F0 (pitch) for the full spoken audio.
      3. Assigns each non-space character a segment and computes:
         - duration (seconds)
         - representative F0 (median over segment)
         - volume (precomputed)
      4. Builds a pitch contour aligned to character-axis coordinates.

    Characters that are spaces receive:
         duration = 0, f0 = None

    Args:
        spoken_audio (np.ndarray):
            The silence-trimmed audio signal.
        sampling_rate (int | float):
            Sampling rate of the audio.
        chars (List[str]):
            List of characters including spaces.
        volumes (List[float]):
            Precomputed per-character loudness (dB).
        hop_length (int):
            Hop length for onset detection and F0 analysis.

    Returns:
        Dict[str, Any]:
            {
                "char_summary": List[Dict],
                "pitch_contour_char": {
                    "char_axis": List[float],
                    "f0_hz": List[Optional[float]],
                    "chars": List[str]
                }
            }
    """
    # 1) Detect onset envelope and peak frames
    onset_env, onset_frames = detect_onsets(
        spoken_audio, sampling_rate, hop_length=hop_length)

    # Count non-space characters
    nonspace_count = sum(ch != " " for ch in chars)

    # Select suitable boundaries
    boundaries = select_boundaries_for_chars(
        spoken_len=len(spoken_audio),
        onset_envelope=onset_env,
        onset_frames=onset_frames,
        hop_length=hop_length,
        target_char_count=nonspace_count,
    )

    # 2) Compute F0 contour for the whole spoken audio
    times, f0_hz = pyin_f0(
        spoken_audio, sampling_rate=sampling_rate, hop_length=hop_length)

    # Frame start-sample positions for boundary mapping
    frame_samples = librosa.frames_to_samples(
        np.arange(len(f0_hz)), hop_length=hop_length)

    # 3) Character-level summary
    char_summary = []
    seg_idx = 0
    nonspace_indices = [i for i, ch in enumerate(chars) if ch != " "]
    nonspace_segments: List[Tuple[int, int, int]] = []

    for ch, vol in zip(chars, volumes):
        if ch == " ":
            # Spaces receive zero-duration and no F0
            char_summary.append(
                {"char": ch, "volume_db": vol, "duration_sec": 0.0, "f0_hz": None}
            )
            continue

        if seg_idx >= len(boundaries) - 1:
            # If out of boundaries, assign zero-length
            char_summary.append(
                {"char": ch, "volume_db": vol, "duration_sec": 0.0, "f0_hz": None}
            )
            continue

        s = int(boundaries[seg_idx])
        e = int(boundaries[seg_idx + 1])

        # Record segment for pitch-contour mapping
        if seg_idx < len(nonspace_indices):
            nonspace_segments.append((s, e, nonspace_indices[seg_idx]))
        seg_idx += 1

        # ㅇuration in seconds for this character segment
        duration = max(0.0, (e - s) / sampling_rate)

        # F0 median for frames inside this segment
        in_seg = (frame_samples >= s) & (frame_samples < e)
        f0_vals = f0_hz[in_seg]
        if f0_vals.size > 0:
            f0_valid = f0_vals[~np.isnan(f0_vals)]
            f0_rep = float(np.nanmedian(f0_valid)
                           ) if f0_valid.size > 0 else None
        else:
            f0_rep = None

        char_summary.append(
            {"char": ch, "volume_db": vol, "duration_sec": round(
                duration, 3), "f0_hz": f0_rep}
        )

    # 4) Build pitch contour on character axis
    char_axis_pos: List[float] = []
    char_axis_f0: List[Any] = []
    seg_ptr = 0

    for fs, f0 in zip(frame_samples, f0_hz):
        # Advance segment pointer if frame is beyond the current boundary
        while seg_ptr < len(nonspace_segments) and fs >= nonspace_segments[seg_ptr][1]:
            seg_ptr += 1

        if seg_ptr >= len(nonspace_segments):
            break

        s, e, char_idx = nonspace_segments[seg_ptr]

        # Skip frames not inside the segment
        if fs < s or fs >= e or e <= s:
            continue

         # Fractional progress in this character segment
        frac = float((fs - s) / (e - s)) if (e - s) > 0 else 0.0
        x = float(char_idx) + frac
        char_axis_pos.append(x)
        char_axis_f0.append(None if np.isnan(f0) else float(f0))

    pitch_contour_char = {
        "char_axis": char_axis_pos,
        "f0_hz": char_axis_f0,
        "chars": chars,
    }

    return {
        "char_summary": char_summary,
        "pitch_contour_char": pitch_contour_char,
    }


def analyze_intonation(audio_file_path: str | PathLike) -> Response:
    """
    Analyze character-level intonation patterns for the given audio file.

    Workflow:
        1. Run `analyze_intensity()` to obtain character list and per-character volumes.
        2. Remove silence from the audio to obtain spoken-only frames.
        3. Generate character-aligned prosody summaries (duration, F0, volume).
        4. Build character-axis pitch contours.
        5. Return an IntonationResponse containing the full analysis.

    Args:
        audio_file_path (str):
            Path to the input audio file.

    Returns:
        Response:
            - IntonationResponse on success
            - ErrorResponse if silence removal or intensity analysis fails
    """
    # 1) Get intensity results
    res_intensity = analyze_intensity(audio_file_path)
    if isinstance(res_intensity, ErrorResponse):
        return res_intensity

    char_volumes = res_intensity.get_value("char_volumes")
    chars = [d["char"] for d in char_volumes]
    volumes = [d["volume"] for d in char_volumes]

    # 2) Load audio and extract non-silent portion
    y, sampling_rate = load_audio(audio_file_path)
    spoken_audio = compute_spoken_audio(y, top_db=40)
    if spoken_audio.size == 0:
        return ErrorResponse(
            error_name="Cannot Remove Silent Intervals",
            error_details="An unknown error occured while removing silent intervals from audio frame."
        )

    # 3) Character-aligned prosody
    result = summarize_char_level_prosody(
        spoken_audio=spoken_audio, sampling_rate=sampling_rate,
        chars=chars, volumes=volumes, hop_length=256
    )

    # 4) Build final response
    response = IntonationResponse(
        status="SUCCESS",
        char_summary=result["char_summary"],
        pitch_contour_char=result["pitch_contour_char"]
    )

    return response
