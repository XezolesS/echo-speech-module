import whisper
import os
import sys
import time
from os import PathLike

def analyze_speechrate(audio_file_path: str | PathLike) -> dict:
    if not os.path.exists(audio_file_path):
        return {"status": "error", "message": f"File not found: {audio_file_path}"}

    try:
        model = whisper.load_model("base")

        start_time = time.time()
        result = model.transcribe(str(audio_file_path), language="ko")
        end_time = time.time()
        analysis_time = round(end_time - start_time, 2)

        full_transcript = result.get("text", "").strip()
        segments = result.get("segments", [])

        if not segments or not full_transcript:
            return {"status": "error", "message": "No speech detected"}

        total_speech_time_seconds = sum(seg['end'] - seg['start'] for seg in segments)
        total_speech_time_seconds = round(total_speech_time_seconds, 2)

        if total_speech_time_seconds <= 0:
            return {"status": "error", "message": "Invalid speech duration"}

        total_words = len(full_transcript.split())
        total_characters = len(full_transcript.replace(" ", ""))

        wpm = round((total_words / total_speech_time_seconds) * 60, 2)
        cps = round(total_characters / total_speech_time_seconds, 2)

        result_data = {
            "status": "success",
            "file_name": str(audio_file_path),
            "wpm": wpm,
            "cps": cps,
            "total_speech_time_seconds": total_speech_time_seconds,
            "total_words": total_words,
            "total_characters_no_space": total_characters,
            "analysis_time_seconds": analysis_time,
            "transcript": full_transcript
        }

        return result_data

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python speechrate.py <audio_file_path>")
        sys.exit(1) 

    audio_file_to_test = sys.argv[1]
    result = analyze_speechrate(audio_file_to_test)
    
    print(json.dumps(result, ensure_ascii=False, indent=4))