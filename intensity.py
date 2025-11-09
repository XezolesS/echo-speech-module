import os
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
import tempfile
import colorama
from colorama import Fore
import json
import sys

colorama.init(autoreset=True)

def analyze_audio_volume(audio_file_path):
    if not os.path.exists(audio_file_path):
        print(f"오류: '{audio_file_path}' 파일을 찾을 수 없습니다.")
        print("분석할 오디오 파일을 스크립트와 같은 폴더에 위치시키고 파일명을 확인해주세요.")
        return

    try:
        y, sr_native = librosa.load(audio_file_path, sr=None)
    except Exception as e:
        print(f"오디오 파일을 불러오는 중 오류가 발생했습니다: {e}")
        return

    full_text = ""
    r = sr.Recognizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        normalized_y = librosa.util.normalize(y)
        temp_full_wav_path = os.path.join(temp_dir, "full_audio.wav")
        sf.write(temp_full_wav_path, normalized_y, sr_native)

        with sr.AudioFile(temp_full_wav_path) as source:
            print("파일의 노이즈 레벨을 분석하여 감지 기준을 조정합니다...")
            r.adjust_for_ambient_noise(source, duration=1)

        with sr.AudioFile(temp_full_wav_path) as source:
            print("조정된 기준으로 전체 오디오를 다시 읽어옵니다...")
            audio_data = r.record(source)
            try:
                print("오디오 전체를 분석하여 텍스트로 변환하는 중...")
                full_text = r.recognize_google(audio_data, language='ko-KR')
                print(f'-> 인식된 전체 텍스트: "{full_text}"')
            except sr.UnknownValueError:
                print("오디오에서 음성을 인식할 수 없습니다. 파일의 내용이나 볼륨을 확인해주세요.")
                return
            except sr.RequestError as e:
                print(f"Google 음성 인식 서비스에 연결할 수 없습니다; {e}")
                return

    intervals = librosa.effects.split(y, top_db=40)
    
    if not intervals.any():
        print("음성 구간을 찾을 수 없습니다.")
        return

    spoken_audio = np.concatenate([y[start:end] for start, end in intervals])
    text_no_spaces = full_text.replace(" ", "")
    
    if not text_no_spaces:
        print("인식된 텍스트가 비어있습니다.")
        return

    onset_strength = librosa.onset.onset_strength(y=spoken_audio, sr=sr_native)
    
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr_native, units='frames', backtrack=True)
    
    if len(onset_frames) > len(text_no_spaces) - 1:
        sorted_onset_frames = sorted(onset_frames, key=lambda frame: onset_strength[frame], reverse=True)
        boundary_frames = sorted(sorted_onset_frames[:len(text_no_spaces)-1])
    else:
        boundary_frames = list(onset_frames)
        
    boundary_samples = librosa.frames_to_samples(boundary_frames)
    boundaries = np.concatenate(([0], boundary_samples, [len(spoken_audio)]))
    
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

    words_list = full_text.split()
    final_char_volumes = []
    text_cursor = 0
    for word in words_list:
        for char in word:
            if text_cursor < len(char_volumes):
                final_char_volumes.append(char_volumes[text_cursor])
                text_cursor += 1
        if word != words_list[-1]:
            final_char_volumes.append({'char': ' ', 'volume': -100.0})

    if not final_char_volumes:
        print("분석할 텍스트가 없습니다.")
        return

    json_output_path = os.path.splitext(audio_file_path)[0] + ".json"
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_char_volumes, f, ensure_ascii=False, indent=4)
        print(f"-> 분석 결과가 '{json_output_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"JSON 파일 저장 중 오류가 발생했습니다: {e}")

    volumes = [item['volume'] for item in final_char_volumes if item['char'] != ' ']
    if not volumes:
        print("인식된 텍스트에 음량이 없습니다.")
        return

    min_vol, max_vol = min(volumes), max(volumes)

    print("\n")
    console_output = ""
    for item in final_char_volumes:
        char = item['char']
        volume = item['volume']
        
        if char == ' ':
            console_output += ' '
            continue

        if (max_vol - min_vol) != 0:
            normalized_volume = (volume - min_vol) / (max_vol - min_vol)
        else:
            normalized_volume = 0.5
            
        if normalized_volume > 0.75:
            color = Fore.RED
        elif normalized_volume > 0.25:
            color = Fore.GREEN
        else:
            color = Fore.YELLOW
            
        console_output += f"{color}{char}"
    
    print(console_output)
    print("\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python intensity.py <오디오_파일_경로>")
        print("예: python intensity.py test5.mp3")
        sys.exit(1) 

    audio_file = sys.argv[1] 
    analyze_audio_volume(audio_file)