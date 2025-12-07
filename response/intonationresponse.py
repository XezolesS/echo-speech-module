from typing import Any

from .response import Response


class IntonationResponse(Response):
    """
    A wrapper class for intonation analysis results.

    This class contains:
    - status (str): Processing result status.
    - char_summary (list[dict[str, Any]]):
        Per-character prosody features including:
            - char (str): The character.
            - volume_db (float): Estimated loudness in dB.
            - duration_sec (float): Duration of the character segment.
            - f0_hz (float | None): Representative F0 (pitch) value in Hz,
              or None if unvoiced.
    - pitch_contour_char (dict[str, Any]):
        Character-axis pitch contour information, including:
            - char_axis (list[float]):
                Continuous x-axis values mapping F0 frames to character positions.
            - f0_hz (list[float | None]):
                F0 values aligned to the character axis, with None for
                unvoiced frames.
            - chars (list[str]):
                The original character sequence for reference.

    This response structure is designed for downstream visualization,
    UI rendering, or evaluators requiring pitch-aligned character-level data.
    """

    def __init__(self,
                 status: str = "",
                 char_summary: list[dict[str, Any]] | None = None,
                 pitch_contour_char: dict[str, Any] | None = None):
        super().__init__(status=status,
                         char_summary=char_summary,
                         pitch_contour_char=pitch_contour_char)
