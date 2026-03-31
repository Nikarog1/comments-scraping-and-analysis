from __future__ import annotations

import sys

from yt_comments.cli.helpers import _configure_logging
from yt_comments.cli.parser import build_parser

from dotenv import load_dotenv
load_dotenv()


    
def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
        
    parser = build_parser()
    args = parser.parse_args(argv)
    
    _configure_logging(args.verbose)
    
    return args.func(args)