"""
run.py

Usage examples:
    python run.py a
    python run.py --number 3 a b c
"""

import argparse
import sys
from pathlib import Path


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "input_file",
        nargs=1,
        help="Input sound file to process. (.wav)"
    )
    parser.add_argument(
        "-l", "--intensity",
        action='store_true',
        default=None,
        help="Analyze speech intensity(loudness)."
    )
    parser.add_argument(
        "-s", "--speechrate",
        action='store_true',
        default=None,
        help="Analyze speechrate."
    )
    parser.add_argument(
        "-i", "--intonation",
        action='store_true',
        default=None,
        help="Analyze intonation."
    )
    parser.add_argument(
        "-a", "--articulation",
        action='store_true',
        default=None,
        help="Analyze articulation."
    )

    # print help when no argument given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Process sound input to various form of speech data."
    )
    args = parse_args(parser)

    # print help when no input_file given
    if args.input_file is None:
        parser.print_help()
        sys.exit(1)

    # process file path
    file_path = Path(args.input_file[0])
    if not file_path.is_file():
        print(f'{file_path} is not a file.')
        sys.exit(2)

    if file_path.suffix != '.wav':
        print(f'{file_path.suffix} file is not supported.'
              '\nSupports: .wav')
        sys.exit(2)

    # process intensity
    if args.intensity:
        pass

    # process speech rate
    if args.speechrate:
        pass

    # process intonation
    if args.intonation:
        pass

    # process articulation
    if args.articulation:
        pass


if __name__ == "__main__":
    main()
