from os import PathLike

import Levenshtein
import librosa
from speech_recognition import RequestError, UnknownValueError

from audio_utils import load_audio, transcribe_audio_file

from response import ArticulationResponse, ErrorResponse, Response


def analyze_articulation(audio_file_path: str | PathLike, reference_text: str | None = None) -> Response:
    """
    오디오 파일을 입력받아 조음 정확성 및 유창성 지표를 분석합니다.

    Args:
        audio_file_path (str): 분석할 오디오 파일의 경로 (.wav, .flac 등)
        reference_text (str, optional): 발화해야 할 원본 대본. 
                                      제공 시 정확도(Accuracy) 지표가 계산됩니다.

    Returns:
        dict: 분석 결과가 담긴 딕셔너리
    """
    # 2. 음향 신호 분석 (Librosa 사용)
    # y: 오디오 시계열 데이터, sr_rate: 샘플링 레이트
    y, sampling_rate = load_audio(audio_file_path)
    total_duration = librosa.get_duration(y=y, sr=sampling_rate)

    # 침묵 구간(Silence)과 발화 구간(Speech) 분리
    # top_db: 침묵으로 간주할 데시벨 임계값 (조절 가능)
    intervals = librosa.effects.split(y, top_db=20)

    # 발화된 구간(순수 말하기 시간)의 총 길이 계산
    speech_duration = 0
    for start, end in intervals:
        speech_duration += (end - start) / sampling_rate

    pause_duration = total_duration - speech_duration

    # 침묵 비율 계산
    pause_ratio = pause_duration / total_duration if total_duration > 0 else 0

    # 3. 음성 인식 (Speech Recognition 사용)
    try:
        transcribed_text = transcribe_audio_file(
            audio_file_path, language='ko-KR').strip()
    except (UnknownValueError, RequestError) as e:
        return ErrorResponse(error_name=e.__class__.__name__, error_details=e.args[0])

    # 4. 발화 속도 분석 (음절 단위)
    # 공백을 제거한 순수 글자 수 계산 (한국어 기준)
    num_syllables = len(transcribed_text.replace(" ", ""))
    articulation_rate = num_syllables / speech_duration if speech_duration > 0 else 0

    # 5. 정확성 분석 (Reference Text가 있는 경우)
    if reference_text:
        # 텍스트 정규화 (공백 제거 후 비교가 일반적임)
        ref_clean = reference_text.strip()
        hyp_clean = transcribed_text.strip()

        # Levenshtein 거리 계산 (편집 거리)
        distance = Levenshtein.distance(ref_clean, hyp_clean)
        length = len(ref_clean)

        # CER (Character Error Rate) 계산
        # CER = (삽입 + 삭제 + 대체) / 원본 길이
        cer = distance / length if length > 0 else 0

        # 정확도 점수 (1 - CER) * 100, 음수가 되지 않도록 처리
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
