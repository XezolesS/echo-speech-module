from datetime import datetime

from .response import Response


class ErrorResponse(Response):
    def __init__(self, error_name: str, error_details: str):
        super().__init__(
            status="ERROR",
            error_name=error_name,
            error_details=error_details,
            time=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
