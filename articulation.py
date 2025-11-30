import os

import Levenshtein
import librosa
import numpy as np
import speech_recognition as sr


def analyze_articulation(audio_file_path, reference_text=None):
    """
    오디오 파일을 입력받아 조음 정확성 및 유창성 지표를 분석합니다.

    Args:
        audio_file_path (str): 분석할 오디오 파일의 경로 (.wav, .flac 등)
        reference_text (str, optional): 발화해야 할 원본 대본. 
                                      제공 시 정확도(Accuracy) 지표가 계산됩니다.

    Returns:
        dict: 분석 결과가 담긴 딕셔너리
    """

    result = {
        "status": "success",
        "transcription": "",          # 인식된 텍스트
        "duration_sec": 0.0,          # 전체 길이
        "articulation_rate": 0.0,     # 조음 속도 (휴식 제외 음절/초)
        "pause_ratio": 0.0,           # 전체 시간 중 침묵 비율
        "accuracy_score": None,       # 정확도 (0~100)
        "cer": None,                  # 문자 오류율 (Character Error Rate)
        "message": ""
    }

    # 1. 파일 존재 확인
    if not os.path.exists(audio_file_path):
        result["status"] = "error"
        result["message"] = "파일을 찾을 수 없습니다."
        return result

    try:
        # 2. 음향 신호 분석 (Librosa 사용)
        # y: 오디오 시계열 데이터, sr_rate: 샘플링 레이트
        y, sr_rate = librosa.load(audio_file_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr_rate)
        result["duration_sec"] = round(total_duration, 2)

        # 침묵 구간(Silence)과 발화 구간(Speech) 분리
        # top_db: 침묵으로 간주할 데시벨 임계값 (조절 가능)
        intervals = librosa.effects.split(y, top_db=20)

        # 발화된 구간(순수 말하기 시간)의 총 길이 계산
        speech_duration = 0
        for start, end in intervals:
            speech_duration += (end - start) / sr_rate

        pause_duration = total_duration - speech_duration

        # 침묵 비율 계산
        if total_duration > 0:
            result["pause_ratio"] = float(
                round(pause_duration / total_duration, 3))

        # 3. 음성 인식 (Speech Recognition 사용)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)

        try:
            # Google Web Speech API 사용 (한국어 설정)
            recognized_text = recognizer.recognize_google(
                audio_data, language='ko-KR')
            result["transcription"] = recognized_text
        except sr.UnknownValueError:
            result["status"] = "warning"
            result["message"] = "음성을 인식할 수 없습니다 (명확하지 않음)."
            return result
        except sr.RequestError as e:
            result["status"] = "error"
            result["message"] = f"API 요청 실패: {e}"
            return result

        # 4. 발화 속도 분석 (음절 단위)
        # 공백을 제거한 순수 글자 수 계산 (한국어 기준)
        num_syllables = len(recognized_text.replace(" ", ""))

        if speech_duration > 0:
            result["articulation_rate"] = float(
                round(num_syllables / speech_duration, 2))  # 순수 발화 시간 기준

        # 5. 정확성 분석 (Reference Text가 있는 경우)
        if reference_text:
            # 텍스트 정규화 (공백 제거 후 비교가 일반적임)
            ref_clean = reference_text.strip()
            hyp_clean = recognized_text.strip()

            # Levenshtein 거리 계산 (편집 거리)
            distance = Levenshtein.distance(ref_clean, hyp_clean)
            length = len(ref_clean)

            # CER (Character Error Rate) 계산
            # CER = (삽입 + 삭제 + 대체) / 원본 길이
            cer = distance / length if length > 0 else 0
            result["cer"] = round(cer, 4)

            # 정확도 점수 (1 - CER) * 100, 음수가 되지 않도록 처리
            accuracy = max(0, (1 - cer) * 100)
            result["accuracy_score"] = round(accuracy, 2)

    except (OSError, ValueError, TypeError, librosa.util.exceptions.ParameterError) as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result
