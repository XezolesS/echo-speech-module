import os
from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
from intensity import analyze_intensity

def compute_spoken_audio(y: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
    """무성 구간을 제거한 spoken 오디오를 반환."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if intervals is None or len(intervals) == 0:
        return np.array([], dtype=y.dtype)
    return np.concatenate([y[s:e] for s, e in intervals])

def select_boundaries_for_chars(
    spoken_len: int,
    onset_envelope: np.ndarray,
    onset_frames: np.ndarray,
    hop_length: int,
    target_char_count: int,
) -> np.ndarray:
    """
    문자 수(공백 제외)에 맞도록 경계(sample 인덱스)를 생성.
    onset 수가 많으면 강한 onset 위주로 선택하고, 모자라면 균등분할로 보정.
    반환: 길이 (target_char_count+1)의 sample 인덱스 배열
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
        boundary_samples = librosa.frames_to_samples(boundary_frames, hop_length=hop_length)
    else:
        boundary_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

    # 앞/뒤 경계 추가
    boundaries = np.concatenate(
        [np.array([0]), boundary_samples.astype(int), np.array([spoken_len])]
    )

    # 목표 개수와 다르면 균등분할로 재보정
    needed = target_char_count + 1
    if len(boundaries) != needed:
        boundaries = np.linspace(0, spoken_len, num=needed, dtype=int)

    return boundaries

def pyin_f0(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 256
           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    librosa.pyin으로 F0(Hz) 시퀀스를 계산.
    반환: (times_sec, f0_hz)  (NaN 포함)
    """
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    # 무성 구간은 NaN 그대로 두되, 후처리 시 중앙값 계산에서 제외
    return times, f0

def summarize_char_level_prosody(
    spoken: np.ndarray,
    sr: int,
    chars: List[str],
    volumes: List[float],
    hop_length: int = 256,
) -> Dict[str, Any]:
    """
    문자(공백 포함) 축에 맞춘 prosody 요약을 만든다.
    - 공백(' ')은 duration=0, f0=None 처리
    - 공백 제외 문자수에 맞춰 경계를 만든 뒤, 순서대로 문자에 대응
    """
    # 1) onset 기반 경계 생성 (공백 제외 개수에 맞춤)
    onset_env = librosa.onset.onset_strength(y=spoken, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units="frames", backtrack=True
    )

    nonspace_count = sum(ch != " " for ch in chars)
    boundaries = select_boundaries_for_chars(
        spoken_len=len(spoken),
        onset_envelope=onset_env,
        onset_frames=onset_frames,
        hop_length=hop_length,
        target_char_count=nonspace_count,
    )

    # 2) 전체 spoken에 대해 f0 시퀀스 계산
    times, f0_hz = pyin_f0(spoken, sr=sr, hop_length=hop_length)

    # frame -> sample 범위 매핑용 헬퍼
    frame_samples = librosa.frames_to_samples(np.arange(len(f0_hz)), hop_length=hop_length)

    # 3) 문자 단위 요약 (공백은 구간 배정 없이 pass)
    char_summary = []
    seg_idx = 0  # 공백 제외 문자마다 하나의 세그먼트 소비
    # non-space 문자 인덱스 목록 (원본 chars 축 기준)
    nonspace_indices = [i for i, ch in enumerate(chars) if ch != " "]
    # 각 non-space 문자에 대응하는 (시작샘플, 끝샘플, 원본 문자 인덱스) 목록 구축
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

        # duration
        duration = max(0.0, (e - s) / sr)

        # f0: 해당 구간에 걸친 frame들 선택 후 중앙값
        # frame이 sample s~e에 들어오는지로 마스크
        in_seg = (frame_samples >= s) & (frame_samples < e)
        f0_vals = f0_hz[in_seg]
        if f0_vals.size > 0:
            f0_valid = f0_vals[~np.isnan(f0_vals)]
            f0_rep = float(np.nanmedian(f0_valid)) if f0_valid.size > 0 else None
        else:
            f0_rep = None

        char_summary.append(
            {"char": ch, "volume_db": vol, "duration_sec": round(duration, 3), "f0_hz": f0_rep}
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
        if fs < s or e <= s:
            # 세그먼트 시작 전이거나 비정상 구간인 경우 skip
            continue
        if fs >= e:
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
        "char_level": char_summary,
        "pitch_contour_char": pitch_contour_char,
    }

def analyze_intonation(audio_path: str) -> Dict[str, Any]:
    """
    1) intensity + 문자 리스트: 기존 analyze_intensity() 호출
    2) spoken 오디오 생성
    3) 문자 축에 맞춘 duration/F0 계산
    4) 결과 + 시각화
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    iv = analyze_intensity(audio_path)
    if not iv:
        raise RuntimeError("analyze_intensity() 결과가 비어 있습니다. (STT 실패/무성 구간 등)")

    chars = [d["char"] for d in iv]
    volumes = [float(d["volume"]) for d in iv]

    # 2) 오디오 로드 및 spoken 생성
    y, sr_native = librosa.load(audio_path, sr=None)
    spoken = compute_spoken_audio(y, sr_native, top_db=40)
    if spoken.size == 0:
        raise RuntimeError("유의미한 발화(spoken) 구간을 찾지 못했습니다.")

    # 3) 문자 축 prosody 요약
    result = summarize_char_level_prosody(
        spoken=spoken, sr=sr_native, chars=chars, volumes=volumes, hop_length=256
    )

    return result