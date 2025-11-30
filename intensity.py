import json
import os
import tempfile
from os import PathLike
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
import speech_recognition as sr


def get_normal_intensity() -> dict:
    """
    20대 한국인 남녀의 정상 발성 크기 기준값을 반환합니다.

    Returns:
        dict: 남성 및 여성의 평균(mean)과 표준편차(stddev)를 포함한 딕셔너리
    """
    return {
        "male_20": {
            "mean": 70.0,
            "stddev": 6.0
        },
        "female_20": {
            "mean": 68.0,
            "stddev": 6.0
        }
    }


def analyze_intensity(audio_file_path: str | PathLike) -> List[Dict[str, Any]]:
    if not os.path.exists(audio_file_path):
        return []

    try:
        y, sr_native = librosa.load(audio_file_path, sr=None)
    except Exception as e:
        return []

    # speech-to-text
    full_text = ""
    r = sr.Recognizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        normalized_y = librosa.util.normalize(y)
        temp_full_wav_path = os.path.join(temp_dir, "full_audio.wav")
        sf.write(temp_full_wav_path, normalized_y, sr_native)

        with sr.AudioFile(temp_full_wav_path) as source:
            r.adjust_for_ambient_noise(source, duration=1)

        with sr.AudioFile(temp_full_wav_path) as source:
            audio_data = r.record(source)
            try:
                full_text = r.recognize_google(audio_data, language='ko-KR')
            except sr.UnknownValueError:
                return []
            except sr.RequestError as e:
                return []

    # non-silent intervals
    intervals = librosa.effects.split(y, top_db=40)

    if not intervals.any():
        return []

    spoken_audio = np.concatenate([y[start:end] for start, end in intervals])
    text_no_spaces = full_text.replace(" ", "")

    if not text_no_spaces:
        return []

    onset_strength = librosa.onset.onset_strength(y=spoken_audio, sr=sr_native)

    # onset detection
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_strength, sr=sr_native, units='frames', backtrack=True)

    if len(onset_frames) > len(text_no_spaces) - 1:
        sorted_onset_frames = sorted(
            onset_frames, key=lambda frame: onset_strength[frame], reverse=True)
        boundary_frames = sorted(sorted_onset_frames[:len(text_no_spaces)-1])
    else:
        boundary_frames = list(onset_frames)

    boundary_samples = librosa.frames_to_samples(boundary_frames)
    boundaries = np.concatenate(([0], boundary_samples, [len(spoken_audio)]))

    # estimate intensity per character
    char_volumes = []
    for i, char in enumerate(text_no_spaces):
        if i >= len(boundaries) - 1:
            break

        start_sample = int(boundaries[i])
        end_sample = int(boundaries[i+1])
        char_audio = spoken_audio[start_sample:end_sample]

        if char_audio.size > 0:
            rms = np.mean(librosa.feature.rms(y=char_audio))
            volume = 20 * np.log10(rms + 1e-9)
            char_volumes.append({'char': char, 'volume': round(volume, 2)})

    # reconstruct, add paddings between words
    words_list = full_text.split()
    final_char_volumes: List[Dict[str, Any]] = []
    text_cursor = 0
    for word in words_list:
        for char in word:
            if text_cursor < len(char_volumes):
                final_char_volumes.append(char_volumes[text_cursor])
                text_cursor += 1
        if word != words_list[-1]:
            final_char_volumes.append({'char': ' ', 'volume': -100.0})

    if not final_char_volumes:
        return []

    return final_char_volumes
