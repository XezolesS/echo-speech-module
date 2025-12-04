"""
run.py

This script is a entry point of echo-speech-module.

Usage:
    run.py [-h] [-l] [-s] [-i] [-a] input_file                                                                                                                                      

    Process sound input to various form of speech data.

    positional arguments:
    input_file          Input sound file to process. (.wav)

    options:
    -h, --help          show this help message and exit
    -l, --intensity     Analyze speech intensity(loudness).
    -s, --speechrate    Analyze speechrate.
    -i, --intonation    Analyze intonation.
    -a, --articulation  Analyze articulation.
        
Author:
    Joe "XezolesS" K.   tndid7876@gmail.com
"""

import argparse
import logging
import sys
from pathlib import Path

from articulation import analyze_articulation
from intensity import analyze_intensity
from intonation import analyze_intonation
from response import ErrorResponse, Response
from speechrate import analyze_speechrate

SUPPORTED_EXTENSIONS = {
    ".wav"
}

logger = logging.getLogger(name="echosm")
logger.setLevel(logging.DEBUG)

log_handler = logging.StreamHandler(sys.stdout)
log_handler.setLevel(logging.DEBUG)
log_formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s]\t%(message)s")
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "input_file",
        nargs=1,
        help="Input sound file to process. (.wav)"
    )
    parser.add_argument(
        "-l", "--intensity",
        action="store_true",
        default=False,
        help="Analyze speech intensity(loudness)."
    )
    parser.add_argument(
        "-s", "--speechrate",
        action="store_true",
        default=False,
        help="Analyze speechrate."
    )
    parser.add_argument(
        "-i", "--intonation",
        action="store_true",
        default=False,
        help="Analyze intonation."
    )
    parser.add_argument(
        "-a", "--articulation",
        action="store_true",
        default=False,
        help="Analyze articulation."
    )

    # print help when no argument given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def main() -> str | None:
    # init argument parser
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Process sound input to various form of speech data."
    )
    args = parse_args(parser)
    logger.debug("Arguments parsed: %s", str(args))

    # print help when no input_file is given
    if args.input_file is None:
        logger.error("No input file has given.")
        parser.print_help()
        return None

    # print help when no process option is given
    if (args.intensity or args.speechrate or args.intonation or args.articulation) is False:
        logger.error("Need at least 1 process option.")
        parser.print_help()
        return None

    # process file path
    file_path = Path(args.input_file[0])

    if not file_path.exists():
        logger.error("File '%s' does not exists.", file_path.name)
        return ErrorResponse(
            error_name="File Not Found",
            error_details=f"File '{file_path.name}' does not exists."
        ).to_json()

    if not file_path.is_file():
        logger.error("'%s' is not a file.", file_path.absolute())
        return ErrorResponse(
            error_name="Not A File",
            error_details=f"'{file_path.absolute()}' is not a file."
        ).to_json()

    if file_path.suffix not in SUPPORTED_EXTENSIONS:
        ext_str = ", ".join(SUPPORTED_EXTENSIONS)
        logger.error("File '%s' is not supported.\nSupports:\n%s",
                     file_path.name, ext_str)
        return ErrorResponse(
            error_name="File Not Supported",
            error_details=f"File '{file_path.name}' is not supported. Supports: {ext_str}"
        ).to_json()

    response = Response()

    # process intensity
    if args.intensity:
        logger.info("Starts to process intensity")

        res_intensity = analyze_intensity(audio_file_path=file_path)
        response.set_value("intensity", res_intensity)

    # process speech rate
    if args.speechrate:
        logger.info("Starts to process speech rate")

        res_speechrate = analyze_speechrate(audio_file_path=file_path)
        response.set_value("speechrate", res_speechrate)

    # process intonation
    if args.intonation:
        logger.info("Starts to process intonation")

        res_intonation = analyze_intonation(audio_path=file_path)
        response.set_value("intonation", res_intonation)

    # process articulation
    if args.articulation:
        logger.info("Starts to process articulation")

        res_articulation = analyze_articulation(
            audio_file_path=file_path, reference_text=None)
        response.set_value("articulation", res_articulation)

    return response.to_json()


if __name__ == "__main__":
    result = main()

    if result is not None:
        print(result)
