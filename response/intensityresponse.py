import numpy as np
from .response import Response


class IntensityResponse(Response):
    """
    A wrapper class of per-character intensity analysis response.

    This class contains:
    - status (str): Processing result status.
    - char_volumes (list[dict]): A list of character-volume mappings, where each entry is:
        {
            "char": str,      # The character
            "volume": float   # Estimated intensity in decibels (dB)
        }

    Methods:
    - add_char_volume(char, volume):
        Adds a character and its corresponding dB volume to the response.
    """

    def __init__(self, status: str = "UNKNOWN"):
        super().__init__(status=status, char_volumes=[])

    def add_char_volume(self, char: str, volume: float | np.float32):
        if isinstance(volume, np.float32):
            volume = volume.item()

        char_volumes: list = self.get_value("char_volumes")
        char_volumes.append({"char": char, "volume": volume})
