from __future__ import annotations

import os
import argparse
import logging
import sys

from pathlib import Path
from datetime import datetime, timezone

from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.basic_stats.service import BasicStatsService
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.analysis.tfidf.service import TfidfService

from yt_comments.ingestion.scrape_service import ScrapeCommentsService
from yt_comments.ingestion.video_id_extractor import extract_video_id
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient
from yt_comments.ingestion.youtube_client import StubYouTubeClient

from yt_comments.preprocessing.preprocess_service import PreprocessCommentsService
from yt_comments.preprocessing.text_preprocessor import TextPreprocessor

from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.gold_basic_stats_parquet_repository import ParquetBasicStatsRepository
from yt_comments.storage.gold_tfidf_keywords_parquet_repository import ParquetTfidfKeywordsRepository
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
        help="Enable debug logging"
    )
    
    subparser = parser.add_subparsers(dest="command", required=True)
    
    # SCRAPE
    scrape = subparser.add_parser(
        "scrape", 
        help="Scrape comments from a YouTube video"
    )
    scrape.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    scrape.add_argument(
        "--limit", 
        type=int, 
        default=200, 
        help="Maximum number of comments to fetch"
    )
    scrape.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Bronze output directory"
    )
    scrape.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing bronze file",
    )
    
    # PREPROCESS
    preprocess = subparser.add_parser(
        "preprocess", 
        help="Build Silver parquet from Bronze JSONL"
    )
    preprocess.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    preprocess.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Bronze input directory"
        )
    preprocess.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Silver output directory"
        )
    preprocess.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Rows per parquet write batch"
        )
    preprocess.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing silver file",
    )
    
    # STATS
    b_stats = subparser.add_parser(
        "stats", 
        help="Compute Gold v1 basic stats from Silver"
    )
    b_stats.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    b_stats.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    b_stats.add_argument(
        "--top-n", 
        type=int, 
        default=30, 
        help="Top N tokens (default: 30)"
    )
    b_stats.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Min token length (default: 2)"
    )
    b_stats.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Do not drop numeric tokens"
    )
    b_stats.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Do not lowercase before tokenization"
    )
    b_stats.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Do not drop stopwords"
    )
    b_stats.add_argument(
        "--lang", 
        default="en",
        help="Stopwords language (default: en)"
    )
    b_stats.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Arrow batch size (default: 5000)"
    )
    
    # TFIDF
    tfidf = subparser.add_parser(
        "tfidf", 
        help="Compute Gold v2 TF-IDF from Silver"
    )
    tfidf.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    tfidf.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    tfidf.add_argument(
        "--top-k", 
        type=int, 
        default=30, 
        help="Top K keywords (default: 30)"
    )
    tfidf.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Min token length (default: 2)"
    )
    tfidf.add_argument(
        "--min-df", 
        type=int, 
        default=2, 
        help="Min token df, i.e., in how many documents token must appear (default: 2)"
    )
    tfidf.add_argument(
        "--max-df", 
        type=float, 
        default=0.9, 
        help="Max token df fraction, i.e., drop tokens appearing in more documents in percents (default: 0.9)"
    )
    tfidf.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Do not drop numeric tokens"
    )
    tfidf.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Do not lowercase before tokenization"
    )
    tfidf.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Do not drop stopwords"
    )
    tfidf.add_argument(
        "--lang", 
        default="en",
        help="Stopwords language (default: en)"
    )
    tfidf.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Arrow batch size (default: 5000)"
    )
    tfidf.add_argument(
        "--ngram-min", 
        type=int, 
        default=1, 
        help="Min ngram to be extracted (default: 1)"
    )
    tfidf.add_argument(
        "--ngram-max", 
        type=int, 
        default=1, 
        help="Max ngram to be extracted (default: 1 - meaning extraction of unigrams only)"
    )
    tfidf.add_argument(
        "--min-ngram-df", 
        type=int, 
        default=2, 
        help="Min df of ngram, i.e., in how many documents ngram must appear (default: 2)"
    )
    
    return parser
    
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

    if args.command == "stats":
        video_id = extract_video_id(args.video)
        
        data_root = Path(args.data_root)
        silver_path = _silver_parquet_path(data_root, video_id)
        if not silver_path.exists():
            parser.error(f"Silver file not found: {silver_path}")
        
        svc = BasicStatsService(preprocess_version="v1")
        cfg = BasicStatsConfig(
            top_n_tokens=args.top_n,
            min_token_len=args.min_token_len,
            drop_numeric_tokens=not args.keep_numeric,
            lowercase=not args.no_lowercase,
            drop_stopwords=not args.keep_stopwords,
            stopwords_lang=args.lang,
        )
                
        b_stats = svc.compute_for_video(
            video_id=video_id,
            silver_parquet_path=str(silver_path),
            config=cfg,
            created_at_utc=datetime.now(timezone.utc),
            batch_size=args.batch_size
        )
        
        repo = ParquetBasicStatsRepository(data_root=data_root)
        repo.save(b_stats)
        
        print(f"video_id: {b_stats.video_id}")
        print(f"rows: {b_stats.row_count} | empty_text: {b_stats.empty_text_count}")
        print(f"tokens: total={b_stats.total_token_count} | unique={b_stats.unique_token_count}")
        if b_stats.top_tokens:
            top_preview = ", ".join(f"{t.token}:{t.count}" for t in b_stats.top_tokens[:10])
            print(f"top_tokens: {top_preview}")
        else:
            print("top_tokens: (none)")
        return 0
    
    if args.command == "tfidf":
        video_id = extract_video_id(args.video)
        
        data_root = Path(args.data_root)
        silver_path = _silver_parquet_path(data_root, video_id)
        if not silver_path.exists():
            parser.error(f"Silver file not found: {silver_path}")
            
        if args.ngram_min < 1:
            raise SystemExit("--ngram-min must be >= 1")
        if args.ngram_max < args.ngram_min:
            raise SystemExit("--ngram-max must be >= --ngram-min")
        if args.min_ngram_df < 1:
            raise SystemExit("--min-ngram-df must be >= 1")

        svc = TfidfService(preprocess_version="v1")
        cfg = TfidfConfig(
            top_k=args.top_k,
            min_token_len=args.min_token_len,
            drop_numeric_tokens=not args.keep_numeric,
            lowercase=not args.no_lowercase,
            drop_stopwords=not args.keep_stopwords,
            stopwords_lang=args.lang,
            min_df=args.min_df,
            max_df=args.max_df,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_ngram_df=args.min_ngram_df,  
        )
        
        tfidf = svc.compute_for_video(
            video_id=video_id,
            silver_parquet_path=str(silver_path),
            config=cfg,
            created_at_utc=datetime.now(timezone.utc),
            batch_size=args.batch_size  
        )
        
        repo = ParquetTfidfKeywordsRepository(data_root=data_root)
        repo.save(tfidf)
        
        print(f"video_id: {tfidf.video_id}")
        print(f"rows: {tfidf.row_count} | empty_text: {tfidf.empty_text_count} | docs_used: {tfidf.doc_count_non_empty}")
        print("top_keywords:")
        
        if not tfidf.keywords:
            print(" (none)")
        else:
            for i, kw in enumerate(tfidf.keywords, start=1):
                print(
                    f" {i:>2}. {kw.token:<15} "
                    f"score={kw.score:.3f} "
                    f"df={kw.df}"
                )
        return 0
        
    return 2


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)
    
def _silver_parquet_path(data_root: Path, video_id: str) -> Path:
    return data_root / "silver" / video_id / "comments.parquet"

    