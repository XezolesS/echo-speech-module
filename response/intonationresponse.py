from typing import Any

from .response import Response


class IntonationResponse(Response):
    def __init__(self,
                 status: str = "",
                 char_summary: list[dict[str, Any]] | None = None,
                 pitch_contour_char: dict[str, Any] | None = None):
        super().__init__(status=status,
                         char_summary=char_summary,
                         pitch_contour_char=pitch_contour_char)
