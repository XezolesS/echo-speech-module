**Echo Speech Module**

**Overview:**
- **Project:** A small toolkit to analyze spoken audio for training and feedback on speech habits.
- **Main features:** per-character intensity (loudness), speech-rate (WPM/CPS), character-aligned intonation (duration / pitch), and articulation accuracy.
- **Primary language:** Python (uses `librosa`, `speech_recognition`, `soundfile`, and NumPy).

**Quick Start**
- **Install dependencies:** See `requirements.txt` and install into a virtual environment.

```pwsh
# Example (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Run analyses (CLI):** the entrypoint is `run.py`. You can run any combination of modules concurrently.

```pwsh
# Analyze intensity + speechrate + intonation + articulation
python run.py -lsia input.wav

# Analyze only intonation
python run.py -i input.wav
```

**Top-level files**
- **`run.py`**: CLI wrapper that validates input, dispatches requested modules and returns a JSON `Response` object. It now runs selected analyses concurrently using a thread pool.
- **`audio_utils.py`**: Shared audio helpers (load audio, remove silence, detect onsets, and a small STT wrapper using `speech_recognition`). Centralized to keep modules DRY.
- **`response/`**: Response classes for structured outputs: `IntensityResponse`, `SpeechrateResponse`, `IntonationResponse`, `ArticulationResponse`, and `ErrorResponse`.

**Detailed module workflows**

**Intensity Analysis**
- **Entry:** `intensity.analyze_intensity(audio_file_path)` → returns an `IntensityResponse` or `ErrorResponse`.
- **Input:** one audio file (wav / supported extension).
- **Goal:** estimate per-character loudness (dB) aligned with transcribed text.
- **Processing steps:**
  - **Load audio:** uses `audio_utils.load_audio()` (librosa, native sampling rate).
  - **Transcription:** `audio_utils.transcribe_audio_file()` produces a Korean (`ko-KR`) transcript string. Caller handles `UnknownValueError` and `RequestError` (returned as `ErrorResponse`).
  - **Silence removal:** `audio_utils.compute_spoken_audio()` concatenates non-silent intervals (librosa.effects.split with `top_db=40`). If no voiced audio is found the function returns `ErrorResponse`.
  - **Onset detection:** `audio_utils.detect_onsets()` computes the onset strength envelope and detects onset frames. The algorithm then selects boundary frames to match the number of non-space characters: if there are more onsets than needed, pick the strongest (by envelope energy) up to `len(text_no_spaces)-1`, otherwise keep all detected onsets.
  - **Frame→sample conversion:** convert selected onset frames to sample indices with `librosa.frames_to_samples()` and build boundary array including start and end samples.
  - **Per-character RMS → dB:** for each non-space character segment, compute RMS energy (librosa.feature.rms), convert to dB via `20 * log10(rms)`; very low RMS is clipped to a fixed low value (e.g., -100 dB) to mark silence.
  - **Reconstruction:** re-insert spaces between words in the transcript, assigning a placeholder low volume (e.g., -100 dB) for space characters, and pack results into `IntensityResponse`.
- **Output:** `IntensityResponse` with `char_volumes` list of `{char, volume}` entries and `status`.
- **Notes & caveats:**
  - Onset detection is heuristic — alignment is approximate, especially for connected speech. The algorithm reduces/expands boundaries to match character counts.
  - Accuracy depends on the STT transcript (word/character counts affect segmentation).

**Speechrate Analysis**
- **Entry:** `speechrate.analyze_speechrate(audio_file_path)` → returns a `SpeechrateResponse` or `ErrorResponse`.
- **Input:** one audio file.
- **Goal:** compute words-per-minute (WPM) and characters-per-second (CPS) for the spoken portions of audio.
- **Processing steps:**
  - **Load audio:** `audio_utils.load_audio()`.
  - **Transcription:** `audio_utils.transcribe_audio_file()` to obtain the full transcript string.
  - **Silence removal:** `audio_utils.compute_spoken_audio()` concatenates voiced intervals to estimate total spoken duration (in seconds). This approach approximates the sum of segments Whisper provided previously.
  - **Metric computation:**
    - `total_words = len(transcript.split())`
    - `total_characters = len(transcript.replace(" ", ""))`
    - `WPM = (total_words / total_speech_time_seconds) * 60`
    - `CPS = total_characters / total_speech_time_seconds`
  - **Response packaging:** return a `SpeechrateResponse` containing `wpm`, `cps`, `total_speech_time`, `total_words`, `total_characters`, `analysis_time`, and `transcript`.
- **Output:** `SpeechrateResponse` with metrics and transcript.
- **Notes & caveats:**
  - This implementation uses Google Web Speech (via `speech_recognition`) for transcription which requires network access. For offline workflows consider integrating VOSK or an offline ASR model.
  - Speech duration is approximated from voiced samples (energy-based) rather than explicit ASR segments, which is adequate for rate metrics.

**Intonation Analysis**
- **Entry:** `intonation.analyze_intonation(audio_file_path)` → returns `IntonationResponse` or `ErrorResponse`.
- **Input:** one audio file.
- **Goal:** produce character-aligned prosodic summaries: per-character duration, representative F0 and a character-axis pitch contour for visualization.
- **Processing steps:**
  - **Intensity first:** calls `intensity.analyze_intensity()` to obtain `chars` and per-character volumes. If intensity fails (STT failure or silence) it returns the underlying `ErrorResponse`.
  - **Load audio / spoken extraction:** `audio_utils.load_audio()` and `compute_spoken_audio()` to get voiced audio and sampling rate.
  - **Onset-based boundaries:** compute an onset envelope and frames (`detect_onsets`) and call the helper `select_boundaries_for_chars()` which attempts to map non-space character count to sample boundaries by selecting strong onsets or falling back to uniform partitioning when onsets are insufficient.
  - **Pitch extraction:** use `librosa.pyin` to compute an F0 contour (per-frame Hz values), keeping NaN for unvoiced frames.
  - **Character-level summarization:** for each non-space character segment, compute duration (seconds) and representative F0 (median of voiced frames within the segment). Spaces receive duration 0 and `f0=None`.
  - **Character-axis pitch contour:** map each F0 frame to a continuous `char_axis` coordinate (character index + intra-segment progress fraction) and produce aligned `f0_hz` values for plotting.
  - **Response packaging:** `IntonationResponse` with `char_summary` list and `pitch_contour_char` dict.
- **Output:** `IntonationResponse` containing character-level duration, volume and representative pitch, and a pitch contour mapped to characters for visualization.
- **Notes & caveats:**
  - Pitch estimation (PYIN) is sensitive to recording quality; unvoiced frames are left as NaN and treated as `None` in the final contour.
  - Boundary selection is heuristic to match character counts and works best for well-articulated speech with clear onsets.

**Articulation Analysis**
- **Entry:** `articulation.analyze_articulation(audio_file_path, reference_text=None)` → returns `ArticulationResponse` or `ErrorResponse`.
- **Input:** one audio file and optional `reference_text` (the expected script).
- **Goal:** evaluate articulation speed/fluency and optionally accuracy vs. a reference using Levenshtein distance.
- **Processing steps:**
  - **Load audio:** `audio_utils.load_audio()`.
  - **Silence removal:** `librosa.effects.split` and sum voiced intervals to compute `speech_duration`.
  - **Transcription:** `audio_utils.transcribe_audio_file()` to obtain the recognized transcript.
  - **Articulation rate:** count non-space characters (heuristic for Korean syllables) and divide by `speech_duration` → syllables/sec.
  - **Pause ratio:** `(total_duration - speech_duration) / total_duration`.
  - **Accuracy (optional):** when `reference_text` is provided compute Levenshtein distance and derive CER and accuracy percentage `(1 - CER) * 100`.
  - **Response packaging:** `ArticulationResponse` containing `duration`, `articulation_rate`, `pause_ratio`, `accuracy_score`, `char_error_rate`, and `transcription`.
- **Output:** `ArticulationResponse`.
- **Notes & caveats:**
  - Character-based syllable counting is language-specific; for Korean this is reasonable (Hangul syllable blocks typically correspond to syllables) but is not portable to all languages.

**Other code & utilities**
- **`audio_utils.py`**: central helpers:
  - `load_audio(path)` — loads audio using `librosa.load` at native SR.
  - `compute_spoken_audio(y, top_db)` — returns voiced audio by concatenating `librosa.effects.split` intervals.
  - `transcribe_audio_file(path, language)` — normalizes audio to a temp WAV and runs `speech_recognition` (Google Web Speech) returning the transcript or raising SR exceptions.
  - `detect_onsets(spoken_audio, sr, hop_length)` — returns onset envelope and onset frame indices.
- **`response/`**: typed response classes that encapsulate module outputs and support JSON serialization via `Response.to_json()`.

**CLI usage and concurrency**
- `run.py` validates input, builds a set of requested analysis tasks, and executes them concurrently using `concurrent.futures.ThreadPoolExecutor`. Each task returns Response objects (or `ErrorResponse`) and the final aggregate `Response` is printed as JSON.
- Control maximum concurrency via `--max-workers`.

**Troubleshooting & Tips**
- If STT fails often, check network access (Google Web Speech requires connectivity) or consider switching to an offline ASR (VOSK) for privacy and reliability.
- For noisy recordings adjust silence `top_db` thresholds or preprocess with noise reduction.
- Pitch (PYIN) is sensitive to sampling rate and SNR; consider pre-filtering or using `sr=16000` common for speech models.

