import numpy as np
from .response import Response


class IntensityResponse(Response):
    def __init__(self):
        super().__init__(char_volumes=[])

    def add_char_volume(self, char: str, volume: float | np.float32):
        if isinstance(volume, np.float32):
            volume = volume.item()

        char_volumes: list = self.get_value("char_volumes")
        char_volumes.append({"char": char, "volume": volume})
