from .response import Response


class ArticulationResponse(Response):
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
