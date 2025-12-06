"""Intonation analysis utilities.

This module computes character-aligned prosodic summaries (duration, F0)
and a pitch contour mapped to a character axis. It relies on shared
helpers from `audio_utils` and the `analyze_intensity` result which
provides per-character volumes and the character sequence.
"""

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
    """Generate sample-index boundaries matching non-space character count.

    The function attempts to use detected onsets as boundaries. If there
    are more onsets than required it selects the strongest ones by onset
    energy; if there are fewer, it falls back to an even partitioning.

    Args:
        spoken_len: length of the spoken audio in samples.
        onset_envelope: onset strength envelope (per frame).
        onset_frames: detected onset frame indices.
        hop_length: hop length used when converting frames -> samples.
        target_char_count: number of non-space characters to segment.

    Returns:
        A 1-D integer numpy array of sample indices with length
        ``target_char_count + 1`` representing segment boundaries.
    """
    if target_char_count <= 0:
        return np.array([0, spoken_len])

    # onset frame -> sample
    onset_strength = onset_envelope
    if onset_frames.size > target_char_count - 1:
        # onset 강도 기준 상위 N-1 선택 후 정렬
        sorted_frames = sorted(
            onset_frames, key=lambda fr: onset_strength[fr], reverse=True
        )[: (target_char_count - 1)]
        boundary_frames = np.array(sorted(sorted_frames))
        boundary_samples = librosa.frames_to_samples(
            boundary_frames, hop_length=hop_length)
    else:
        boundary_samples = librosa.frames_to_samples(
            onset_frames, hop_length=hop_length)

    # 앞/뒤 경계 추가
    boundaries = np.concatenate(
        [np.array([0]), boundary_samples.astype(int), np.array([spoken_len])]
    )

    # 목표 개수와 다르면 균등분할로 재보정
    needed = target_char_count + 1
    if len(boundaries) != needed:
        boundaries = np.linspace(0, spoken_len, num=needed, dtype=int)

    return boundaries


def pyin_f0(y: np.ndarray, sampling_rate: int | float, frame_length: int = 2048, hop_length: int = 256
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frame-wise F0 (Hz) using librosa.pyin.

    Returns frame times (seconds) and F0 values (Hz). Unvoiced frames are
    left as NaN so callers can ignore them for median computations.
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

    # Keep NaNs for unvoiced frames so downstream median ignores them.
    return times, f0


def summarize_char_level_prosody(
    spoken_audio: np.ndarray,
    sampling_rate: int | float,
    chars: List[str],
    volumes: List[float],
    hop_length: int = 256,
) -> Dict[str, Any]:
    """Produce per-character prosody summary aligned to ``chars``.

    For each character (including spaces) the function returns a dict with
    volume (dB), duration (seconds) and a representative F0 (Hz) where
    available. Spaces receive duration 0 and f0=None.
    """
    # 1) onset envelope and frame indices (frame-units)
    onset_env, onset_frames = detect_onsets(
        spoken_audio, sampling_rate, hop_length=hop_length)

    nonspace_count = sum(ch != " " for ch in chars)
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

    # Map each F0 frame to its starting sample index for segment membership tests
    frame_samples = librosa.frames_to_samples(
        np.arange(len(f0_hz)), hop_length=hop_length)

    # 3) 문자 단위 요약 (공백은 구간 배정 없이 pass)
    # seg_idx: 공백 제외 문자마다 하나의 세그먼트 소비
    # nonspace_indicies: non-space 문자 인덱스 목록 (원본 chars 축 기준)
    # nonspace_segments: 각 non-space 문자에 대응하는 (시작샘플, 끝샘플, 원본 문자 인덱스) 목록 구축
    char_summary = []
    seg_idx = 0
    nonspace_indices = [i for i, ch in enumerate(chars) if ch != " "]
    nonspace_segments: List[Tuple[int, int, int]] = []
    for ch, vol in zip(chars, volumes):
        if ch == " ":
            char_summary.append(
                {"char": ch, "volume_db": vol, "duration_sec": 0.0, "f0_hz": None}
            )
            continue

        if seg_idx >= len(boundaries) - 1:
            # 경계가 모자라면 잔여는 0 길이 처리
            char_summary.append(
                {"char": ch, "volume_db": vol, "duration_sec": 0.0, "f0_hz": None}
            )
            continue

        s = int(boundaries[seg_idx])
        e = int(boundaries[seg_idx + 1])

        # non-space 문자 세그먼트 기록 (원본 문자 인덱스는 nonspace_indices로부터 매핑)
        if seg_idx < len(nonspace_indices):
            nonspace_segments.append((s, e, nonspace_indices[seg_idx]))
        seg_idx += 1

        # duration in seconds for this character segment
        duration = max(0.0, (e - s) / sampling_rate)

        # f0: 해당 구간에 걸친 frame들 선택 후 중앙값
        # Select frames whose start-sample falls inside the character segment
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

    # 4) 글자 축(character axis)으로의 pitch contour 구성
    # - 각 frame을 해당 non-space 문자 세그먼트에 매핑
    # - 글자 인덱스 + 구간 내 진행도(0~1)를 x좌표로 사용
    char_axis_pos: List[float] = []
    char_axis_f0: List[Any] = []
    seg_ptr = 0
    # nonspace_segments는 s(샘플) 기준으로 정렬되어 있음
    for fs, f0 in zip(frame_samples, f0_hz):
        # NaN F0도 contour에는 포함(시각화를 위해 None으로 변환)
        # 현재 frame이 속한 세그먼트를 찾기 위해 seg_ptr를 전진
        while seg_ptr < len(nonspace_segments) and fs >= nonspace_segments[seg_ptr][1]:
            seg_ptr += 1

        if seg_ptr >= len(nonspace_segments):
            break

        s, e, char_idx = nonspace_segments[seg_ptr]

        # 세그먼트 시작 전이거나 비정상 구간인 경우 skip
        if fs < s or fs >= e or e <= s:
            continue

        frac = float((fs - s) / (e - s)) if (e - s) > 0 else 0.0
        x = float(char_idx) + frac
        char_axis_pos.append(x)
        char_axis_f0.append(None if np.isnan(f0) else float(f0))

    pitch_contour_char = {
        "char_axis": char_axis_pos,  # 예: 0.0 ~ len(chars)-1 범위의 연속 좌표
        "f0_hz": char_axis_f0,
        "chars": chars,  # 라벨링 용이하도록 원본 문자 배열 제공
    }

    return {
        "char_summary": char_summary,
        "pitch_contour_char": pitch_contour_char,
    }


def analyze_intonation(audio_file_path: str) -> Response:
    """
    1) intensity + 문자 리스트: 기존 analyze_intensity() 호출
    2) spoken 오디오 생성
    3) 문자 축에 맞춘 duration/F0 계산
    4) 결과 + 시각화
    """
    # Get intensity
    res_intensity = analyze_intensity(audio_file_path)
    if isinstance(res_intensity, ErrorResponse):
        return res_intensity

    char_volumes = res_intensity.get_value("char_volumes")
    chars = [d["char"] for d in char_volumes]
    volumes = [d["volume"] for d in char_volumes]

    # Get non-silent audio frames and sampling rate
    y, sampling_rate = load_audio(audio_file_path)
    spoken_audio = compute_spoken_audio(y, top_db=40)
    if spoken_audio.size == 0:
        return ErrorResponse(
            error_name="Cannot Remove Silent Intervals",
            error_details="An unknown error occured while removing silent intervals from audio frame."
        )

    # character-aligned prosody summary
    result = summarize_char_level_prosody(
        spoken_audio=spoken_audio, sampling_rate=sampling_rate,
        chars=chars, volumes=volumes, hop_length=256
    )

    response = IntonationResponse(
        status="SUCCESS",
        char_summary=result["char_summary"],
        pitch_contour_char=result["pitch_contour_char"]
    )

    return response
