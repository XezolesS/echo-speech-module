from .response import Response


class SpeechrateResponse(Response):
    """
    A wrapper class of speech-rate analysis response.

    This class contains:
    - status (str): Processing result status.
    - wpm (float): Words per minute during spoken audio.
    - cps (float): Characters per second during spoken audio.
    - total_speech_time (float): Duration of spoken-only segments in seconds.
    - total_words (int): Number of words in transcribed text.
    - total_characters (int): Number of non-whitespace characters.
    - analysis_time (float): Total time taken to perform analysis.
    - transcript (str): Recognized text from the audio.
    """

    def __init__(self,
                 status: str = "UNKNOWN",
                 wpm: float = 0.0,
                 cps: float = 0.0,
                 total_speech_time: float = 0.0,
                 total_words: int = 0,
                 total_characters: int = 0,
                 analysis_time: float = 0.0,
                 transcript: str = ""):
        super().__init__(status=status,
                         wpm=wpm,
                         cps=cps,
                         total_speech_time=total_speech_time,
                         total_words=total_words,
                         total_characters=total_characters,
                         analysis_time=analysis_time,
                         transcript=transcript)
