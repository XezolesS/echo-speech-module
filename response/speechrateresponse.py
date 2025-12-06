from .response import Response


class SpeechrateResponse(Response):
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
