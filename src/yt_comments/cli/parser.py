from __future__ import annotations

import argparse

from yt_comments.cli.helpers import _parse_cli_datetime

from yt_comments.cli.commands.channel import (
    run_channel_stats, run_discover_vids, run_distinctive_keywords, run_preprocess_channel,
    run_report_channel, run_scrape_channel, run_tfidf_channel
)
from yt_comments.cli.commands.video import (
    run_corpus, run_preprocess, run_scrape, run_stats, run_tfidf
)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yt-comments")
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Enable verbose (debug) logging"
    )
    
    subparser = parser.add_subparsers(dest="command", required=True)
    
    # SCRAPE
    scrape = subparser.add_parser(
        "scrape", 
        help="Fetch comments from a YouTube video"
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
        help="Output directory for Bronze data (default: data/bronze)"
    )
    scrape.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing Bronze file if it exists",
    )
    scrape.set_defaults(func=run_scrape)
    
    # PREPROCESS
    preprocess = subparser.add_parser(
        "preprocess", 
        help="Convert Bronze JSONL into Silver parquet"
    )
    preprocess.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    preprocess.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Input Bronze directory (default: data/bronze)"
        )
    preprocess.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Output Silver directory (default: data/silver)"
        )
    preprocess.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Number of rows per parquet write batch"
        )
    preprocess.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing Silver file if it exists",
    )
    preprocess.set_defaults(func=run_preprocess)
    
    # STATS
    b_stats = subparser.add_parser(
        "stats", 
        help="Compute basic token statistics (Gold v1) from Silver data"
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
        help="Number of top tokens to return (default: 30)"
    )
    b_stats.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Minimum token length (default: 2)"
    )
    b_stats.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Keep numeric tokens"
    )
    b_stats.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Disable lowercasing before tokenization"
    )
    b_stats.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Keep stopwords"
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
        help="Compute TF-IDF keywords (Gold) from Silver data"
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
        help="Number of top keywords to return (default: 30)"
    )
    tfidf.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Minimum token length (default: 2)"
    )
    tfidf.add_argument(
        "--min-df", 
        type=int, 
        default=2, 
        help="Minimum document frequency (number of documents a token must appear in)"
    )
    tfidf.add_argument(
        "--max-df", 
        type=float, 
        default=0.9, 
        help="Maximum document frequency (fraction of documents; tokens above are dropped)"
    )
    tfidf.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Keep numeric tokens"
    )
    tfidf.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Disable lowercasing before tokenization"
    )
    tfidf.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Keep stopwords"
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
        help="Minimum n-gram size (default: 1)"
    )
    tfidf.add_argument(
        "--ngram-max", 
        type=int, 
        default=1, 
        help="Maximum n-gram size (default: 1, i.e. unigrams only)"
    )
    tfidf.add_argument(
        "--min-ngram-df", 
        type=int, 
        default=2, 
        help="Minimum document frequency for n-grams (default: 2)"
    )
    tfidf.add_argument(
        "--use-corpus",
        action="store_true",
        help="Use global corpus for IDF calculation"
    )
    tfidf.add_argument(
        "--keep-sentiment",
        action="store_true",
        help="Keep sentiment-related tokens in the final output"
    )
    tfidf.add_argument(
        "--stemming-mode",
        default="none", 
        help="Normalize words (e.g., -ing, -ed, -s → base form). Supported: stem_en"
    )
    tfidf.set_defaults(func=run_tfidf)

    # CORPUS
    corpus = subparser.add_parser(
        "corpus", 
        help="Build a global corpus from all Silver comments"
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
        help="Minimum document frequency"
    )
    corpus.add_argument(
        "--max-df", 
        type=float, 
        default=0.9, 
        help="Maximum document frequency (fraction)"
    )
    corpus.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Keep numeric tokens"
    )
    corpus.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Disable lowercasing before tokenization"
    )
    corpus.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Keep stopwords"
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
        help="Minimum n-gram size (default: 1)"
    )
    corpus.add_argument(
        "--ngram-max", 
        type=int, 
        default=1, 
        help="Maximum n-gram size (default: 1, i.e. unigrams only)"
    )
    corpus.add_argument(
        "--min-ngram-df", 
        type=int, 
        default=2, 
        help="Minimum document frequency for n-grams (default: 2)"
    )
    corpus.set_defaults(func=run_corpus)
    
    # DISCOVER_VIDEOS
    discover_vids = subparser.add_parser(
        "discover-videos", 
        help="List videos available on a channel"
    )
    discover_vids.add_argument(
        "channelId", 
        help="YouTube channel reference (ID, @handle, or URL)"
    )
    discover_vids.add_argument(
        "--limit", 
        type=int, 
        default=100, 
        help="Maximum number of videos to return"
    )
    discover_vids.add_argument(
        "--published-after", 
        type=_parse_cli_datetime, 
        help="Include only videos published after this date"
    )
    discover_vids.add_argument(
        "--published-before", 
        type=_parse_cli_datetime, 
        help="Include only videos published before this date"
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
        help="Fetch comments for videos from a channel"
    )
    scrape_channel.add_argument(
        "channelId", 
        help="YouTube channel reference (ID, @handle, or URL)"
    )
    scrape_channel.add_argument(
        "--video-limit", 
        type=int, 
        default=100, 
        help="Maximum number of videos to process"
    )
    scrape_channel.add_argument(
        "--comments-limit", 
        type=int, 
        default=5000, 
        help="Maximum number of comments per video"
    )
    scrape_channel.add_argument(
        "--published-after", 
        type=_parse_cli_datetime, 
        help="Include only videos published after this date"
    )
    scrape_channel.add_argument(
        "--published-before", 
        type=_parse_cli_datetime, 
        help="Include only videos published before this date"
    )
    scrape_channel.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    scrape_channel.add_argument(
        "--bronze-dir", 
        default="data/bronze", 
        help="Output directory for Bronze data (default: data/bronze)"
    )
    scrape_channel.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing Bronze files if they exist",
    )
    scrape_channel.set_defaults(func=run_scrape_channel)

    # PREPROCESS-CHANNEL
    preprocess_channel = subparser.add_parser(
        "preprocess-channel", 
        help="Convert Bronze JSONL comments for a channel into Silver parquet"
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
        help="Input Bronze directory (default: data/bronze)"
    )
    preprocess_channel.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Output Silver directory (default: data/silver)"
    )
    preprocess_channel.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Number of rows per parquet write batch"
    )
    preprocess_channel.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing Silver files if they exist",
    )
    preprocess_channel.set_defaults(func=run_preprocess_channel)


    # CHANNEL STATS
    c_stats = subparser.add_parser(
        "stats-channel", 
        help="Compute channel-level token statistics (Gold v4) from Silver data"
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
        help="Silver data directory (default: data/silver)"
    )
    c_stats.add_argument(
        "--top-n", 
        type=int, 
        default=30, 
        help="Number of top tokens to return (default: 30)"
    )
    c_stats.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Minimum token length (default: 2)"
    )
    c_stats.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Keep numeric tokens"
    )
    c_stats.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Disable lowercasing before tokenization"
    )
    c_stats.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Keep stopwords"
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
        help="Normalize words (e.g., -ing, -ed, -s → base form). Supported: stem_en"
    )
    c_stats.set_defaults(func=run_channel_stats)


    # TFIDF-CHANNEL
    tfidf_channel = subparser.add_parser(
        "tfidf-channel", 
        help="Compute TF-IDF keywords for a channel from Silver data"
    )
    tfidf_channel.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    tfidf_channel.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    tfidf_channel.add_argument(
        "--silver-dir", 
        default="data/silver", 
        help="Silver data directory (default: data/silver)"
    )
    tfidf_channel.add_argument(
        "--top-k", 
        type=int, 
        default=30, 
        help="Number of top keywords to return (default: 30)"
    )
    tfidf_channel.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Minimum token length (default: 2)"
    )
    tfidf_channel.add_argument(
        "--min-df", 
        type=int, 
        default=2, 
        help="Minimum document frequency (number of documents a token must appear in)"
    )
    tfidf_channel.add_argument(
        "--max-df", 
        type=float, 
        default=0.9, 
        help="Maximum document frequency (fraction of documents; tokens above are dropped)"
    )
    tfidf_channel.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Keep numeric tokens"
    )
    tfidf_channel.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Disable lowercasing before tokenization"
    )
    tfidf_channel.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Keep stopwords"
    )
    tfidf_channel.add_argument(
        "--lang", 
        default="en",
        help="Stopwords language (default: en)"
    )
    tfidf_channel.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        help="Arrow batch size (default: 5000)"
    )
    tfidf_channel.add_argument(
        "--ngram-min", 
        type=int, 
        default=1, 
        help="Minimum n-gram size (default: 1)"
    )
    tfidf_channel.add_argument(
        "--ngram-max", 
        type=int, 
        default=1, 
        help="Maximum n-gram size (default: 1, i.e. unigrams only)"
    )
    tfidf_channel.add_argument(
        "--min-ngram-df", 
        type=int, 
        default=2, 
        help="Minimum document frequency for n-grams (default: 2)"
    )
    tfidf_channel.add_argument(
        "--use-corpus",
        action="store_true",
        help="Use a global corpus for IDF calculation"
    )
    tfidf_channel.add_argument(
        "--keep-sentiment",
        action="store_true",
        help="Keep sentiment-related tokens in the final output"
    )
    tfidf_channel.add_argument(
        "--stemming-mode",
        default="none", 
        help="Normalize words (e.g., -ing, -ed, -s → base form). Supported: stem_en"
    )
    tfidf_channel.set_defaults(func=run_tfidf_channel)


    # DISTINCTIVE-KEYWORDS
    distinctive_kws = subparser.add_parser(
        "distinctive-keywords", 
        help="Compute keywords that are distinctive for a video relative to its channel"
    )
    distinctive_kws.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    distinctive_kws.add_argument(
        "video", 
        help="YouTube video URL or 11-character video ID"
    )
    distinctive_kws.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    distinctive_kws.set_defaults(func=run_distinctive_keywords)


    # REPORT-CHANNEL
    report_channel = subparser.add_parser(
        "report-channel", 
        help="Print a compact report based on existing Gold channel artifacts"
    )
    report_channel.add_argument(
        "channelId", 
        help="YouTube channel reference (channel ID, @handle, or URL)"
    )
    report_channel.add_argument(
        "--data-root", 
        default="data", 
        help="Project data directory (default: data)"
    )
    report_channel.add_argument(
        "--video",
        help="Optional video (URL or ID) to include distinctive keywords"
    )
    report_channel.add_argument(
        "--top-k", 
        type=int, 
        default=30, 
        help="Number of top keywords to display (default: 30)"
    )
    report_channel.set_defaults(func=run_report_channel)
    
    return parser