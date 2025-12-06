**에코 스피치 모듈**

**개요:**
- **프로젝트:** 음성 훈련 및 피드백을 위한 간단한 분석 도구 모음
- **주요 기능:** 문자별 음성 세기(intensity), 발화 속도(speech-rate: WPM/CPS), 문자 축 정렬 인토네이션(duration/F0), 조음(articulation) 정확도
- **언어:** Python (`librosa`, `speech_recognition`, `soundfile`, `numpy` 등 사용)

**빠른 시작**
- **의존성 설치:** `requirements.txt`를 참고하여 가상환경에 설치하세요.

```pwsh
# 예시 (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **분석 실행 (CLI):** 진입점은 `run.py`입니다. 필요한 모듈 조합을 동시에 실행할 수 있습니다.

```pwsh
# intensity + speechrate + intonation + articulation 한 번에 실행
python run.py -lsia input.wav

# intonation 만 실행
python run.py -i input.wav
```

**상위 파일 설명**
- `run.py`: 입력 검사, 선택된 모듈 실행(동시 처리) 및 JSON 응답 반환을 담당하는 CLI 래퍼
- `audio_utils.py`: 오디오 로드, 무성(침묵) 제거, onset 검출, STT 래퍼 등 공통 헬퍼 집합
- `response/`: 모듈별 구조화된 출력 클래스( `IntensityResponse`, `SpeechrateResponse`, `IntonationResponse`, `ArticulationResponse`, `ErrorResponse` )

**모듈별 처리 흐름**

1) 문자별 세기 분석 (Intensity)
- **함수:** `intensity.analyze_intensity(audio_file_path)` → `IntensityResponse` 또는 `ErrorResponse`
- **목적:** 인식된 텍스트의 각 문자(공백 제외)에 대해 음량(dB)을 추정하고, 단어 간 공백은 낮은 값(-100 dB 등)으로 표시
- **처리 단계:**
  - 오디오 로드: `audio_utils.load_audio()` (librosa, 원래 샘플링 레이트 유지)
  - 음성 인식: `audio_utils.transcribe_audio_file()` (한국어 `ko-KR`) — 인식 실패 시 예외를 `ErrorResponse`로 반환
  - 무성 제거: `audio_utils.compute_spoken_audio()` (librosa.effects.split, 기본 `top_db=40`) — 발화가 없으면 실패
  - onset 검출: `audio_utils.detect_onsets()`로 onset 강도(프레임 단위)와 onset 프레임을 얻음. 텍스트의 공백 제거 문자 수에 맞춰 경계를 선택함. 검출된 onset이 많으면 강도 상위 onset을 사용, 부족하면 모두 사용
  - 프레임→샘플 변환: `librosa.frames_to_samples()`로 프레임 인덱스를 샘플 인덱스로 변환하고 시작/끝 경계 추가
  - 문자별 RMS→dB: 각 문자 세그먼트에서 RMS(평균)를 계산하고 `20 * log10(rms)`로 dB 변환; 매우 작은 RMS는 -100 dB 등으로 클리핑
  - 재구성: 단어 사이 공백 문자를 삽입하고 `IntensityResponse`에 문자별 볼륨을 추가
- **출력:** `IntensityResponse` (`char_volumes` 리스트 포함)
- **주의:** onset 기반 정렬은 휴리스틱이며 연결음이 강한 발화에서는 부정확할 수 있습니다. STT 정합성도 결과에 큰 영향을 줍니다.

2) 발화 속도 분석 (Speechrate)
- **함수:** `speechrate.analyze_speechrate(audio_file_path)` → `SpeechrateResponse` 또는 `ErrorResponse`
- **목적:** 발음된 구간 기준으로 WPM(분당 단어 수) 및 CPS(초당 문자 수)를 계산
- **처리 단계:**
  - 오디오 로드: `audio_utils.load_audio()`
  - 음성 인식: `audio_utils.transcribe_audio_file()`으로 전체 전사 얻음
  - 무성 제거: `audio_utils.compute_spoken_audio()`로 발화 구간을 이어붙여 총 발화 시간(초) 추정
  - 지표 계산:
    - `total_words = len(transcript.split())`
    - `total_characters = len(transcript.replace(" ", ""))`
    - `WPM = (total_words / total_speech_time_seconds) * 60`
    - `CPS = total_characters / total_speech_time_seconds`
  - 응답 포장: `SpeechrateResponse`에 결과와 분석 시간, 전사 포함
- **출력:** `SpeechrateResponse`
- **주의:** 현재 STT는 Google Web Speech(네트워크 필요)를 사용합니다. 오프라인 번들용으로는 VOSK 같은 대체를 고려하세요. 발화 시간은 에너지 기반으로 추정하므로 ASR 세그먼트 타임스탬프와는 차이가 있을 수 있습니다.

3) 인토네이션 분석 (Intonation)
- **함수:** `intonation.analyze_intonation(audio_file_path)` → `IntonationResponse` 또는 `ErrorResponse`
- **목적:** 문자 축(character axis)에 정렬된 prosody 요약(문자별 duration, 대표 F0) 및 시각화용 pitch contour 생성
- **처리 단계:**
  - 먼저 intensity 실행: `intensity.analyze_intensity()`를 호출하여 문자 시퀀스와 문자별 볼륨을 확보. 실패 시 해당 `ErrorResponse` 반환
  - 오디오 로드/무성 제거: `audio_utils.load_audio()` + `compute_spoken_audio()`
  - 경계 생성: `detect_onsets()`로 onset envelope/frames를 계산하고 `select_boundaries_for_chars()`를 호출하여 비공백 문자 수에 맞춘 샘플 경계 생성(강한 onset 우선 또는 균등 분할)
  - 피치 추출: `librosa.pyin`으로 F0 프레임 시퀀스 계산(무성 프레임은 NaN)
  - 문자 단위 요약: 각 비공백 문자 세그먼트에 대해 duration(초)과 대표 F0(세그먼트 내 유성 프레임의 중앙값) 계산. 공백은 duration=0, f0=None
  - 문자 축 pitch contour: 각 F0 프레임을 해당 문자 세그먼트에 매핑하고 (문자 인덱스 + 구간 내 진행도) 좌표를 만들어 시각화용 배열 생성
  - 응답 포장: `IntonationResponse`에 `char_summary`(문자별 duration/volume/f0)와 `pitch_contour_char` 포함
- **출력:** `IntonationResponse`
- **주의:** PYIN의 피치 추정은 녹음 품질에 민감합니다. 무성 프레임은 NaN으로 남겨지므로 후처리 시 무시해야 합니다. 경계 매핑은 휴리스틱이므로 완벽한 정렬을 보장하지 않습니다.

4) 조음 분석 (Articulation)
- **함수:** `articulation.analyze_articulation(audio_file_path, reference_text=None)` → `ArticulationResponse` 또는 `ErrorResponse`
- **목적:** 발화의 조음 속도/유창성 지표 산출 및(선택적으로) 참조 대본과의 정확도 비교
- **처리 단계:**
  - 오디오 로드: `audio_utils.load_audio()`
  - 무성 제거: `librosa.effects.split`으로 발화 구간을 찾아 총 발화 시간 계산
  - 음성 인식: `audio_utils.transcribe_audio_file()`
  - 조음 속도 산출: 전사에서 공백 제거 문자 수(한국어의 경우 음절 수와 근사) ÷ 발화 시간 → 음절/초
  - 휴지 비율: `(total_duration - speech_duration) / total_duration`
  - 정확도(선택): `reference_text`가 주어지면 Levenshtein 거리로 CER을 계산하고 정확도 `(1 - CER) * 100` 산출
  - 응답 포장: `ArticulationResponse`
- **출력:** `ArticulationResponse`
- **주의:** 문자 기반 음절 계산은 언어에 따라 부적절할 수 있습니다(한국어는 비교적 적합).

**기타 코드 및 유틸**
- `audio_utils.py` 핵심 함수:
  - `load_audio(path)` — `librosa.load`로 오디오를 로드
  - `compute_spoken_audio(y, top_db)` — `librosa.effects.split` 결과를 이어붙여 발화만 반환
  - `transcribe_audio_file(path, language)` — 오디오 정규화 후 임시 WAV로 `speech_recognition`에 전달하여 전사 반환(예외는 호출자 처리)
  - `detect_onsets(spoken_audio, sr, hop_length)` — onset envelope와 onset frame 인덱스 반환
- `response/` 디렉터리: 모듈별 출력 클래스를 제공하며 `Response.to_json()`으로 직렬화 가능

**CLI 사용 및 동시성**
- `run.py`는 입력을 검증하고 요청된 분석 작업을 모아서 `concurrent.futures.ThreadPoolExecutor`로 병렬 실행합니다. 각 작업은 `Response` 객체 또는 `ErrorResponse`를 반환하며, 최종적으로 합쳐진 JSON을 출력합니다.
- 스레드 수는 `--max-workers` 옵션으로 제어할 수 있습니다.

**트러블슈팅 & 팁**
- STT가 자주 실패하면 네트워크 상태를 확인하거나 오프라인 ASR(VOSK 등) 도입을 고려하세요.
- 잡음이 많은 녹음은 사전 잡음 제거 또는 `top_db` 파라미터 조정이 필요합니다.
- PYIN 피치는 SNR과 샘플링레이트에 민감합니다. 필요하면 `sr=16000`로 리샘플링 후 처리하세요.
