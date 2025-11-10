import whisper
import os
import sys
import time
import json

def measure_speech_speed(audio_file_path: str):
    
    if not os.path.exists(audio_file_path):
        print(f"오류: '{audio_file_path}' 파일을 찾을 수 없습니다.")
        print("스크립트와 같은 폴더에 파일이 있는지 확인하세요.")
        return

    try:
        print("Whisper 모델을 로드하는 중입니다... (base 모델)")
        model = whisper.load_model("base")
        print("모델 로드 완료.")

        print(f"'{audio_file_path}' 파일 분석을 시작합니다...")
        start_time = time.time()

        result = model.transcribe(audio_file_path, language="ko")
        
        end_time = time.time()
        analysis_time = round(end_time - start_time, 2)
        print(f"분석 완료. (소요 시간: {analysis_time}초)")

        full_transcript = result.get("text", "").strip()
        segments = result.get("segments", [])

        if not segments or not full_transcript:
            print("\n[결과]")
            print("오디오에서 음성을 인식할 수 없었습니다.")
            return

        total_speech_time_seconds = sum(seg['end'] - seg['start'] for seg in segments)
        total_speech_time_seconds = round(total_speech_time_seconds, 2)

        if total_speech_time_seconds <= 0:
            print("\n[결과]")
            print("유효한 발화 시간을 계산할 수 없습니다.")
            return

        total_words = len(full_transcript.split())
        total_characters = len(full_transcript.replace(" ", ""))

        wpm = round((total_words / total_speech_time_seconds) * 60, 2)
        cps = round(total_characters / total_speech_time_seconds, 2)

        result_data = {
            "file_name": audio_file_path,
            "wpm": wpm,
            "cps": cps,
            "total_speech_time_seconds": total_speech_time_seconds,
            "total_words": total_words,
            "total_characters_no_space": total_characters,
            "analysis_time_seconds": analysis_time,
            "transcript": full_transcript
        }

        print("\n--- 발화 속도 측정 결과 ---")
        print(f"  > 분당 단어 수 (WPM): {result_data['wpm']}")
        print(f"  > 초당 음절 수 (CPS): {result_data['cps']}")
        print(f"  > 총 순수 발화 시간: {result_data['total_speech_time_seconds']}초")
        print(f"  > 총 단어 수: {result_data['total_words']}개")
        print(f"  > 총 음절 수 (공백 제외): {result_data['total_characters_no_space']}자")
        print("\n--- 전체 변환 텍스트 ---")
        print(result_data['transcript'])

        json_file_path = os.path.splitext(audio_file_path)[0] + ".json"
        
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
            print(f"\n결과를 '{json_file_path}' 파일로 저장했습니다.")
        except IOError as e:
            print(f"\nJSON 파일 저장 중 오류 발생: {e}")

    except Exception as e:
        print(f"스크립트 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python speechrate.py <오디오_파일_경로>")
        print("예: python speechrate.py test5.mp3")
        sys.exit(1) 

    audio_file_to_test = sys.argv[1]
    measure_speech_speed(audio_file_to_test)