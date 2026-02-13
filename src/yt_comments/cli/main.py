import os

import argparse
import logging
import sys

from datetime import datetime, timezone

from yt_comments.ingestion.models import Comment
from yt_comments.storage.comments_repository import JSONLCommentsRepository
from yt_comments.ingestion.scrape_service import ScrapeResult, ScrapeCommentsService
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient
from yt_comments.ingestion.youtube_client import StubYouTubeClient

from dotenv import load_dotenv
load_dotenv()



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
        video_id = args.video
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if api_key: 
             client = YouTubeApiClient(api_key=api_key) 
        else:
             logging.warning("API key not found. Using StubYouTubeClient.")
             client = StubYouTubeClient()
        
        repo = JSONLCommentsRepository()
        service = ScrapeCommentsService(client=client, repo=repo)

        result = service.run(video_id, overwrite=True)

        print(f"Saved {result.saved_count} comments to: {result.path}")
        return 0
    
    return 2