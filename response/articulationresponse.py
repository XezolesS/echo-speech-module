from .response import Response


class ArticulationResponse(Response):
    """
    A wrapper class of articulation response.

    This class contains:
    - status (str): Processing result status.
    - duration (float): Total duration of the audio in seconds.
    - articulation_rate (float): Syllables per second of spoken audio.
    - pause_ratio (float): Ratio of silence duration to total duration.
    - accuracy_score (float): Percentage accuracy vs. reference text.
    - char_error_rate (float): CER based on Levenshtein distance.
    - transcription (str): Recognized text from the audio.
    """

    def __init__(self,
                 status: str,
                 duration: float,
                 articulation_rate: float,
                 pause_ratio: float,
                 accuracy_score: float,
                 char_error_rate: float,
                 transcription: str):
        super().__init__(status=status,
                         duration=duration,
                         articulation_rate=articulation_rate,
                         pause_ratio=pause_ratio,
                         accuracy_score=accuracy_score,
                         char_error_rate=char_error_rate,
                         transcription=transcription)
