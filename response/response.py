"""
The response module for building structured response data.

Author:
    Joe "XezolesS" K.   tndid7876@gmail.com
"""

import json
from typing import Any


class Response:
    """
    A base class for a response.
    """

    def __init__(self, **kwargs):
        self.__data = {}

        for key, value in kwargs.items():
            self.__data[key] = value

    def __str__(self) -> str:
        return str(self.__data)

    def get_data(self) -> dict[str, Any]:
        """
        Get a response data in dictionary.

        Returns:
            dict[str, Any]: Response data in dict
        """
        return self.__data

    def get_value(self, key: str) -> Any:
        """
        Get a value for response data corresponds to a key.

        Args:
            key (str): The key of a data

        Returns:
            Any: The value of a data
        """
        return self.__data[key]

    def set_value(self, key: str, value: Any) -> None:
        """
        Set a value for response data corresponds to a key.

        Args:
            key (str): The key of a data
            value (Any): The value of a data
        """
        self.__data[key] = value

    def to_json(self) -> str:
        """
        Stringify response data to JSON

        Returns:
            str: JSON string
        """
        return json.JSONEncoder(indent=2).encode(self.__data)
