import argparse
import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yt-comments")
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="enable debug logging"
    )
    
    subparser = parser.add_subparsers(dest="command", required=True)
    
    scrape = subparser.add_parser(
        "scrape", 
        help="scrape comments from a YouTube video"
    )
    scrape.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    scrape.add_argument(
        "--limit", 
        type=int, 
        default=200, 
        help="maximum number of comments to fetch"
    )
    
    #2DO: add subparser for analysis etc. 
    
    return parser

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)
    
def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
        
    parser = build_parser()
    args = parser.parse_args(argv)
    
    _configure_logging(args.verbose)
    
    if args.command == "scrape":
        print(args.video)
        return 0
    
    return 2