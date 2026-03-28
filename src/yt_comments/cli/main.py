from __future__ import annotations

import os
import argparse
import logging
import sys

from pathlib import Path
from datetime import datetime, timezone

from yt_comments.analysis.features import hash_config
from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.basic_stats.service import BasicStatsService
from yt_comments.analysis.channel_runs.models import ChannelRunSummary
from yt_comments.analysis.channel_stats.service import ChannelTokenStatsService
from yt_comments.analysis.corpus.service import CorpusService
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.analysis.tfidf.service import TfidfService

from yt_comments.ingestion.channel_ref_parser import parse_channel_ref
from yt_comments.ingestion.channel_video_discovery_service import ChannelVideoDiscoveryService
from yt_comments.ingestion.models import ChannelVideoDiscovery
from yt_comments.ingestion.scrape_service import ScrapeCommentsService
from yt_comments.ingestion.video_id_extractor import extract_video_id
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient
from yt_comments.ingestion.youtube_client import StubYouTubeClient

from yt_comments.nlp.stopwords import STOPWORDS

from yt_comments.preprocessing.preprocess_service import PreprocessCommentsService
from yt_comments.preprocessing.text_preprocessor import TextPreprocessor

from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.gold_basic_stats_parquet_repository import ParquetBasicStatsRepository
from yt_comments.storage.gold_channel_ref_mapping_repository import JSONChannelRefRepository
from yt_comments.storage.gold_channel_run_summary_repository import JSONChannelRunSummaryRepository
from yt_comments.storage.gold_channel_token_stats_repository import ParquetChannelTokenStatsRepository
from yt_comments.storage.gold_corpus_df_parquet_repository import ParquetCorpusDfRepository
from yt_comments.storage.gold_tfidf_keywords_parquet_repository import ParquetTfidfKeywordsRepository
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository

from dotenv import load_dotenv
load_dotenv()



logger = logging.getLogger(__name__)
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
        default=5000, 
        help="Maximum number of comments to fetch"
    )
    scrape.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Bronze output directory (default: 'data/bronze')"
    )
    scrape.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing bronze file",
    )
    scrape.set_defaults(func=run_scrape)
    
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
        help="Bronze output directory (default: 'data/bronze')"
        )
    preprocess.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Silver output directory (default: 'data/silver)"
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
    preprocess.set_defaults(func=run_preprocess)
    
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
    b_stats.set_defaults(func=run_stats)
    
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
    tfidf.add_argument(
        "--use-corpus",
        action="store_true",
        help="Use global corpus across all silver comments (default: False)"
    )
    tfidf.add_argument(
        "--keep-sentiment",
        action="store_true",
        help="Keep sentiment words in final result (default: False)"
    )
    tfidf.add_argument(
        "--stemming-mode",
        default="none", 
        help="Normalize words so related forms (-ing, -ed, -s) collapse into one base form (default: none)\nCurrent supported mods: stem_en"
    )
    tfidf.set_defaults(func=run_tfidf)

    # CORPUS
    corpus = subparser.add_parser(
        "corpus", 
        help="Compute corpus across silver comments"
    )
    corpus.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    corpus.add_argument(
        "--min-df", 
        type=int, 
        default=2, 
        help="Min token df, i.e., in how many documents token must appear (default: 2)"
    )
    corpus.add_argument(
        "--max-df", 
        type=float, 
        default=0.9, 
        help="Max token df fraction, i.e., drop tokens appearing in more documents in percents (default: 0.9)"
    )
    corpus.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Do not drop numeric tokens"
    )
    corpus.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Do not lowercase before tokenization"
    )
    corpus.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Do not drop stopwords"
    )
    corpus.add_argument(
        "--lang", 
        default="en",
        help="Stopwords language (default: en)"
    )
    corpus.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Arrow batch size (default: 5000)"
    )
    corpus.add_argument(
        "--ngram-min", 
        type=int, 
        default=1, 
        help="Min ngram to be extracted (default: 1)"
    )
    corpus.add_argument(
        "--ngram-max", 
        type=int, 
        default=1, 
        help="Max ngram to be extracted (default: 1 - meaning extraction of unigrams only)"
    )
    corpus.add_argument(
        "--min-ngram-df", 
        type=int, 
        default=2, 
        help="Min df of ngram, i.e., in how many documents ngram must appear (default: 2)"
    )
    corpus.set_defaults(func=run_corpus)
    
    # DISCOVER_VIDEOS
    discover_vids = subparser.add_parser(
        "discover-videos", 
        help="List available videos on the provided channel"
    )
    discover_vids.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    discover_vids.add_argument(
        "--limit", 
        type=int, 
        default=100, 
        help="Maximum number of videos to list"
    )
    discover_vids.add_argument(
        "--published-after", 
        type=_parse_cli_datetime, 
        help="Include videos published AFTER the specified date"
    )
    discover_vids.add_argument(
        "--published-before", 
        type=_parse_cli_datetime, 
        help="Include videos published BEFORE the specified date"
    )
    discover_vids.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    discover_vids.set_defaults(func=run_discover_vids)
    
    # SCRAPE-CHANNEL
    scrape_channel = subparser.add_parser(
        "scrape-channel", 
        help="Scrape comments from videos from a YouTube channel"
    )
    scrape_channel.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    scrape_channel.add_argument(
        "--video-limit", 
        type=int, 
        default=100, 
        help="Maximum number of videos to list"
    )
    scrape_channel.add_argument(
        "--comments-limit", 
        type=int, 
        default=5000, 
        help="Maximum number of comments to fetch from each video"
    )
    scrape_channel.add_argument(
        "--published-after", 
        type=_parse_cli_datetime, 
        help="Include videos published AFTER the specified date"
    )
    scrape_channel.add_argument(
        "--published-before", 
        type=_parse_cli_datetime, 
        help="Include videos published BEFORE the specified date"
    )
    scrape_channel.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    scrape_channel.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Bronze output directory (default: 'data/bronze')"
    )
    scrape_channel.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing bronze file",
    )
    scrape_channel.set_defaults(func=run_scrape_channel)

    # PREPROCESS-CHANNEL
    preprocess_channel = subparser.add_parser(
        "preprocess-channel", 
        help="Build Silver parquet from Bronze JSONL from a YouTube channel comments"
    )
    preprocess_channel.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    preprocess_channel.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    preprocess_channel.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Bronze output directory (default: 'data/bronze')"
        )
    preprocess_channel.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Silver output directory (default: 'data/silver)"
        )
    preprocess_channel.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Rows per parquet write batch"
        )
    preprocess_channel.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing silver file",
    )
    preprocess_channel.set_defaults(func=run_preprocess_channel)

    # CHANNEL STATS
    c_stats = subparser.add_parser(
        "stats-channel", 
        help="Compute Gold v4 channel token stats from Silver"
    )
    c_stats.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    c_stats.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    c_stats.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Silver output directory (default: 'data/silver)"
        )
    c_stats.add_argument(
        "--top-n", 
        type=int, 
        default=30, 
        help="Top N tokens (default: 30)"
    )
    c_stats.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Min token length (default: 2)"
    )
    c_stats.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Do not drop numeric tokens"
    )
    c_stats.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Do not lowercase before tokenization"
    )
    c_stats.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Do not drop stopwords"
    )
    c_stats.add_argument(
        "--lang", 
        default="en",
        help="Stopwords language (default: en)"
    )
    c_stats.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Arrow batch size (default: 5000)"
    )
    c_stats.add_argument(
        "--stemming-mode",
        default="none", 
        help="Normalize words so related forms (-ing, -ed, -s) collapse into one base form (default: none)\nCurrent supported mods: stem_en"
    )
    c_stats.set_defaults(func=run_channel_stats)
    
    return parser

    
def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
        
    parser = build_parser()
    args = parser.parse_args(argv)
    
    _configure_logging(args.verbose)
    
    return args.func(args)


def run_scrape(args: argparse.Namespace) -> int:
    video_id = extract_video_id(args.video)

    logger.info("Looking up YouTube API key")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key: 
            logger.info("Using YouTube API Client")
            client = YouTubeApiClient(api_key=api_key) 
    else:
            logger.error("YouTube API key not found")
            return 2

    repo = JSONLCommentsRepository(data_dir=args.bronze_dir)
    logger.info("Starting comment scrape | video_id=%s", video_id)
    result = _scrape_video(video_id=video_id, client=client, repo=repo, limit=args.limit, overwrite=args.overwrite)
    logger.info("Comment scrape completed | video_id=%s saved_count=%s path=%s", video_id, result.saved_count, result.path)

    print(f"Saved {result.saved_count} comments to: {result.path}")
    return 0


def run_preprocess(args: argparse.Namespace) -> int:
    video_id = extract_video_id(args.video)

    logger.info("Initializing repositories and text preprocessor")
    bronze_repo = JSONLCommentsRepository(args.bronze_dir)
    silver_repo = ParquetSilverCommentsRepository(args.silver_dir)
    tp = TextPreprocessor()

    logger.info("Initializing preprocess comments service")
    service = PreprocessCommentsService(
        bronze_repo=bronze_repo,
        silver_repo=silver_repo,
        text_preprocessor=tp,
    )

    logger.info("Starting preprocess | video_id=%s", video_id)
    out_path = service.run(video_id, overwrite=args.overwrite, batch_size=args.batch_size)
    logger.info("Preprocess completed | video_id=%s output_path=%s", video_id, out_path)

    print(f"Saved Silver parquet to: {out_path}")
    return 0


def run_stats(args: argparse.Namespace) -> int:
    video_id = extract_video_id(args.video)

    data_root = Path(args.data_root)

    silver_path = _silver_parquet_path(data_root, video_id)
    if not silver_path.exists():
        logger.error("Silver parquet not found | video_id=%s path=%s", video_id, silver_path)
        return 2

    stopwords_hash = str(hash_config(sorted(STOPWORDS[args.lang])))

    svc = BasicStatsService()
    cfg = BasicStatsConfig(
        top_n_tokens=args.top_n,
        min_token_len=args.min_token_len,
        drop_numeric_tokens=not args.keep_numeric,
        lowercase=not args.no_lowercase,
        drop_stopwords=not args.keep_stopwords,
        stopwords_lang=args.lang,
        stopwords_hash=stopwords_hash,
    )

    logger.info("Starting basic stats computation | video_id=%s", video_id)
    b_stats = svc.compute_for_video(
        video_id=video_id,
        silver_parquet_path=str(silver_path),
        config=cfg,
        created_at_utc=datetime.now(timezone.utc),
        batch_size=args.batch_size
    )

    repo = ParquetBasicStatsRepository(data_root=data_root)
    repo.save(b_stats)
    logger.info(
        "Basic stats completed | video_id=%s rows=%s total_tokens=%s unique_tokens=%s",
        b_stats.video_id,
        b_stats.row_count,
        b_stats.total_token_count,
        b_stats.unique_token_count,
    )

    print(f"video_id: {b_stats.video_id}")
    print(f"rows: {b_stats.row_count} | empty_text: {b_stats.empty_text_count}")
    print(f"tokens: total={b_stats.total_token_count} | unique={b_stats.unique_token_count}")
    if b_stats.top_tokens:
        top_preview = ", ".join(f"{t.token}:{t.count}" for t in b_stats.top_tokens[:10])
        print(f"top_tokens: {top_preview}")
    else: 
        print("top_tokens: (none)")
    return 0


def run_tfidf(args: argparse.Namespace) -> int:
    video_id = extract_video_id(args.video)
    
    data_root = Path(args.data_root)
    silver_path = _silver_parquet_path(data_root, video_id)
    if not silver_path.exists():
        logger.error("Silver parquet not found | video_id=%s path=%s", video_id, silver_path)
        return 2
        
    if args.ngram_min < 1:
        logger.error("Invalid argument | --ngram-min must be >= 1")
        return 2

    if args.ngram_max < args.ngram_min:
        logger.error("Invalid argument | --ngram-max must be >= --ngram-min")
        return 2

    if args.min_ngram_df < 1:
        logger.error("Invalid argument | --min-ngram-df must be >= 1")
        return 2
    
    if args.use_corpus:
        logger.info("Loading global corpus")
        corpus = ParquetCorpusDfRepository(data_root=data_root).load()
    else:
        corpus = None
        
    stopwords_hash = str(hash_config(sorted(STOPWORDS[args.lang])))

    svc = TfidfService()
    cfg = TfidfConfig(
        top_k=args.top_k,
        min_token_len=args.min_token_len,
        drop_numeric_tokens=not args.keep_numeric,
        lowercase=not args.no_lowercase,
        drop_stopwords=not args.keep_stopwords,
        stopwords_lang=args.lang,
        stopwords_hash=stopwords_hash,
        normalization=args.stemming_mode,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_ngram_df=args.min_ngram_df,  
    )
    
    logger.info(
        "Starting TF-IDF computation | video_id=%s use_corpus=%s",
        video_id,
        args.use_corpus,
    )
    tfidf = svc.compute_for_video(
        video_id=video_id,
        silver_parquet_path=str(silver_path),
        config=cfg,
        created_at_utc=datetime.now(timezone.utc),
        batch_size=args.batch_size,
        global_corpus=corpus,
        unfilter_sentiment=not args.keep_sentiment,  
    )
    
    repo = ParquetTfidfKeywordsRepository(data_root=data_root)
    repo.save(tfidf)
    logger.info(
        "TF-IDF completed | video_id=%s rows=%s docs_used=%s keywords=%s",
        tfidf.video_id,
        tfidf.row_count,
        tfidf.doc_count_non_empty,
        len(tfidf.keywords),
    )
    
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


def run_corpus(args: argparse.Namespace) -> int:
        
    data_root = Path(args.data_root)
        
    if args.ngram_min < 1:
        logger.error("Invalid argument | --ngram-min must be >= 1")
        return 2

    if args.ngram_max < args.ngram_min:
        logger.error("Invalid argument | --ngram-max must be >= --ngram-min")
        return 2

    if args.min_ngram_df < 1:
        logger.error("Invalid argument | --min-ngram-df must be >= 1")
        return 2

    stopwords_hash = str(hash_config(sorted(STOPWORDS[args.lang])))
    
    corpus = CorpusService(data_root=data_root)
    cfg = TfidfConfig(
        drop_numeric_tokens=not args.keep_numeric,
        lowercase=not args.no_lowercase,
        drop_stopwords=not args.keep_stopwords,
        stopwords_lang=args.lang,
        stopwords_hash=stopwords_hash,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_ngram_df=args.min_ngram_df,  
    )
    
    logger.info("Starting corpus build")
    result = corpus.build(config=cfg, batch_size=args.batch_size)
    
    repo = ParquetCorpusDfRepository(data_root=data_root)
    repo.save(result)
    logger.info(
        "Corpus build completed | videos=%s features=%s",
        result.video_count,
        len(result.tokens),
    )
    
    print("corpus_df:")
    print(f"videos: {result.video_count}")
    print(f"features: {len(result.tokens)}")
    
    return 0


def run_discover_vids(args: argparse.Namespace) -> int:
    logger.info("Looking up YouTube API key")
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    channel_id = args.channelId

    if api_key:
            logger.info("Using YouTube API client | channel_ref=%s", args.channelId)
            parsed_channel_id = parse_channel_ref(args.channelId)
            client = YouTubeApiClient(api_key=api_key)
            channel_id = client.resolve_channel_id(parsed_channel_id)
            _save_channel_id_ref_mapping(data_root=args.data_root, raw_input=parsed_channel_id.value, channel_id=channel_id)
            logger.info("Resolved channel reference | input=%s channel_id=%s", args.channelId, channel_id)
    else:
            logger.error("YouTube API key not found")
            return 2
    
    request = ChannelVideoDiscovery(
        channel_id=channel_id,
        published_after=args.published_after,
        published_before=args.published_before,
        limit=args.limit,    
    )
    
    service = ChannelVideoDiscoveryService(client=client, request=request)
    logger.info("Starting channel video discovery | channel_id=%s", channel_id)
    result = service.run()
    logger.info("Channel video discovery completed | channel_id=%s videos=%s", channel_id, result.video_count)
    
    for video in result.videos:
        if video.published_at:
            date_str = video.published_at.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = "N/A"
        print(f"{date_str} | {video.video_id} | {video.title}")

    print(f"Total videos={result.video_count}")
        
    return 0


def run_scrape_channel(args: argparse.Namespace) -> int:
    logger.info("Looking up YouTube API key")
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    channel_id = args.channelId

    if api_key:
            logger.info("Using YouTube API client | channel_ref=%s", args.channelId)
            client = YouTubeApiClient(api_key=api_key)
            parsed_channel_id = parse_channel_ref(args.channelId)
            channel_id = client.resolve_channel_id(parsed_channel_id)
            _save_channel_id_ref_mapping(data_root=args.data_root, raw_input=parsed_channel_id.value, channel_id=channel_id)
            logger.info("Resolved channel reference | input=%s channel_id=%s", args.channelId, channel_id)
    else:
            logger.error("YouTube API key not found")
            return 2
    
    request = ChannelVideoDiscovery(
        channel_id=channel_id,
        published_after=args.published_after,
        published_before=args.published_before,
        limit=args.video_limit,    
    )
    
    service = ChannelVideoDiscoveryService(client=client, request=request)
    logger.info("Starting channel video discovery | channel_id=%s", channel_id)
    videos = service.run()
    logger.info("Channel video discovery completed | channel_id=%s videos=%s", channel_id, videos.video_count)
    
    repo = JSONLCommentsRepository(data_dir=args.bronze_dir)
    
    logger.info("Starting channel scrape | channel_id=%s", channel_id)
    start_at_utc = datetime.now(tz=timezone.utc)
    comments_count = 0
    errors = 0
    video_ids = []
    for video in videos.videos:
        try:
            result = _scrape_video(
                video_id=video.video_id,
                client=client,
                repo=repo,
                limit=args.comments_limit,
                overwrite=args.overwrite
            )
            comments_count += result.saved_count
            video_ids.append(video.video_id)
            print(f"{video.video_id} | title={video.title} | comments={result.saved_count} | path={result.path}")
        except Exception as e:
             errors += 1
             logger.exception("Video scrape failed | video_id=%s", video.video_id)
             print(f"Failed to scrape | video_id={video.video_id} | error={e}")
    finished_at_utc = datetime.now(tz=timezone.utc)
    logger.info(
        "Channel scrape completed | channel_id=%s videos=%s comments=%s errors=%s",
        channel_id,
        videos.video_count,
        comments_count,
        errors,
    )
    print(f"TOTAL | videos={videos.video_count} | comments={comments_count} | errors={errors}")

    try:
        summary = ChannelRunSummary(
            channel_id=channel_id,
            started_at_utc=start_at_utc,
            finished_at_utc=finished_at_utc,
            video_ids=tuple(video_ids),
            video_count=videos.video_count,
            comment_count=comments_count,
            error_count=errors,
            video_limit=args.video_limit,
            comment_limit=args.comments_limit,
            published_after=args.published_after,
            published_before=args.published_before,
        )
        repo = JSONChannelRunSummaryRepository(data_root=args.data_root)
        path = repo.save(summary)
        logger.info("Channel run summary saved | channel_id=%s path=%s", channel_id, path)
        print(f"Metadata saved to: {path}")
        
    except Exception:
         logger.exception("Failed to save channel run summary | channel_id=%s", channel_id)
    
    return 0

def run_preprocess_channel(args: argparse.Namespace) -> int:
    logger.info("Loading latest channel run summary | channel_id=%s", args.channelId)
    
    channel_id = _load_channel_id_ref_mapping(data_root=args.data_root, raw_input=args.channelId)
    logger.info(
    "Resolved channel reference | channel_ref=%s | channel_id=%s",
    args.channelId,
    channel_id,
    )
    
    repo = JSONChannelRunSummaryRepository(data_root=args.data_root)
    summary = repo.load_latest(channel_id=channel_id)
    logger.info(
        "Loaded channel run summary | channel_id=%s | videos=%d",
        channel_id,
        len(summary.video_ids),
    )
    
    bronze_repo = JSONLCommentsRepository(args.bronze_dir)
    silver_repo = ParquetSilverCommentsRepository(args.silver_dir)
    tp = TextPreprocessor()
    service = PreprocessCommentsService(
            bronze_repo=bronze_repo,
            silver_repo=silver_repo,
            text_preprocessor=tp,
    )

    logger.info(
        "Starting channel preprocessing | channel_id=%s | videos=%d",
        channel_id,
        len(summary.video_ids),
    )
    errors = 0
    for video_id in summary.video_ids:
        try:
            logger.info("Preprocessing video | video_id=%s", video_id)
            out_path = service.run(video_id, overwrite=args.overwrite, batch_size=args.batch_size)
            logger.info(
                "Preprocessing completed | video_id=%s | out_path=%s",
                video_id,
                out_path,
            )
            print(f"Saved Silver parquet to: {out_path}")
        except Exception as e:
            errors += 1
            logger.warning("Preprocessing failed | video_id=%s", video_id)
            print(f"Failed to preprocess | video_id={video_id}")

    logger.info(
        "Channel preprocessing finished | channel_id=%s | total=%d | processed=%d | errors=%d",
        channel_id,
        summary.video_count,
        summary.video_count-errors,
        errors,
    )         
    print(f"TOTAL | videos={summary.video_count} | errors={errors}")     
    return 1 if errors else 0

def run_channel_stats(args: argparse.Namespace) -> int:
     
    logger.info("Loading latest channel run summary | channel_id=%s", args.channelId)
    channel_id = _load_channel_id_ref_mapping(
        data_root=args.data_root,
        raw_input=args.channelId,
    )
    logger.info(
        "Resolved channel reference | channel_ref=%s | channel_id=%s",
        args.channelId,
        channel_id,
    )
    
    logger.info("Extracting channel load metadata | channel_id=%s", channel_id)
    summary_repo = JSONChannelRunSummaryRepository(data_root=args.data_root)
    summary = summary_repo.load_latest(channel_id=channel_id)

    logger.info("Starting channel token stats computation | channel_id=%s", channel_id)
    stopwords_hash = str(hash_config(sorted(STOPWORDS[args.lang])))
    cfg = BasicStatsConfig(
        top_n_tokens=args.top_n,
        min_token_len=args.min_token_len,
        drop_numeric_tokens=not args.keep_numeric,
        lowercase=not args.no_lowercase,
        drop_stopwords=not args.keep_stopwords,
        stopwords_lang=args.lang,
        stopwords_hash=stopwords_hash,
        normalization=args.stemming_mode,
    )

    silver_repo = ParquetSilverCommentsRepository(args.silver_dir)
    service = ChannelTokenStatsService()
    stats = service.compute_for_channel(
        channel_id=channel_id,
        video_ids=summary.video_ids,
        silver_repo=silver_repo,
        config=cfg,
        created_at_utc=datetime.now(timezone.utc),
    )

    repo = ParquetChannelTokenStatsRepository(data_root=args.data_root)
    repo.save(stats)
    logger.info(
        "Channel token stats completed | channel_id=%s rows=%s total_tokens=%s unique_tokens=%s",
        stats.channel_id,
        stats.row_count,
        stats.total_token_count,
        stats.unique_token_count,
    )

    print(f"channel_id: {stats.channel_id}")
    print(f"videos: {len(stats.video_ids)}")
    print(f"rows: {stats.row_count} | empty_text: {stats.empty_text_count}")
    print(
        f"tokens: total={stats.total_token_count} | unique={stats.unique_token_count}"
    )
    if stats.top_tokens:
        top_preview = ", ".join(f"{t.token}:{t.count}" for t in stats.top_tokens[:10])
        print(f"top_tokens: {top_preview}")
    else:
        print("top_tokens: (none)")

    return 0
         

def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format=LOG_FORMAT)


def _silver_parquet_path(data_root: Path, video_id: str) -> Path:
    return data_root / "silver" / video_id / "comments.parquet"


def _parse_cli_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use ISO format like "
            "2026-01-01 or 2026-01-01T12:00:00"
        ) from e

    # If user passed only a date → assume midnight
    # datetime.fromisoformat already handles this.

    # If timezone missing → assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt

def _scrape_video(
          *,
          video_id: str,
          client: YouTubeApiClient | StubYouTubeClient,
          repo: JSONLCommentsRepository,
          limit: int | None,
          overwrite: bool,
):
     service = ScrapeCommentsService(client=client, repo=repo)
     return service.run(video_id, overwrite=overwrite, limit=limit)

def _save_channel_id_ref_mapping(*, data_root: str, raw_input: str, channel_id: str) -> Path:
     return JSONChannelRefRepository(data_root=Path(data_root)).save(raw_input=raw_input, channel_id=channel_id)

def _load_channel_id_ref_mapping(*, data_root: str, raw_input: str) -> str:
    try: 
        return JSONChannelRefRepository(data_root=Path(data_root)).load(raw_input=raw_input)
    except:
         return raw_input
