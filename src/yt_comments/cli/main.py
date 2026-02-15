import os
import argparse
import logging
import sys

from yt_comments.ingestion.scrape_service import ScrapeCommentsService
from yt_comments.ingestion.video_id_extractor import extract_video_id
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient
from yt_comments.ingestion.youtube_client import StubYouTubeClient

from yt_comments.preprocessing.preprocess_service import PreprocessCommentsService
from yt_comments.preprocessing.text_preprocessor import TextPreprocessor

from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository

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
    
    # SCRAPE
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
    scrape.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="bronze output directory"
    )
    scrape.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overwrite existing bronze file",
    )
    
    # PREPROCESS
    preprocess = subparser.add_parser(
        "preprocess", 
        help="build Silver parquet from Bronze JSONL"
    )
    preprocess.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    preprocess.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="bronze input directory"
        )
    preprocess.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="silver output directory"
        )
    preprocess.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="rows per parquet write batch"
        )
    preprocess.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overwrite existing silver file",
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
        video_id = extract_video_id(args.video)
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if api_key: 
             client = YouTubeApiClient(api_key=api_key) 
        else:
             logging.warning("API key not found. Using StubYouTubeClient.")
             client = StubYouTubeClient()
        
        repo = JSONLCommentsRepository()
        service = ScrapeCommentsService(client=client, repo=repo)

        result = service.run(video_id, overwrite=True, limit=args.limit)

        print(f"Saved {result.saved_count} comments to: {result.path}")
        return 0
    
    if args.command == "preprocess":
        video_id = extract_video_id(args.video)

        bronze_repo = JSONLCommentsRepository(args.bronze_dir)
        silver_repo = ParquetSilverCommentsRepository(args.silver_dir)
        tp = TextPreprocessor()

        service = PreprocessCommentsService(
            bronze_repo=bronze_repo,
            silver_repo=silver_repo,
            text_preprocessor=tp,
        )

        out_path = service.run(video_id, overwrite=args.overwrite, batch_size=args.batch_size)
        print(f"Saved Silver parquet to: {out_path}")
        return 0
    
    return 2