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
        help="Compute Gold TF-IDF from Silver"
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

    # TFIDF-CHANNEL
    tfidf_channel = subparser.add_parser(
        "tfidf-channel", 
        help="Compute TF-IDF from channel Silver files"
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
        help="Silver output directory (default: 'data/silver)"
    )
    tfidf_channel.add_argument(
        "--top-k", 
        type=int, 
        default=30, 
        help="Top K keywords (default: 30)"
    )
    tfidf_channel.add_argument(
        "--min-token-len", 
        type=int, 
        default=2, 
        help="Min token length (default: 2)"
    )
    tfidf_channel.add_argument(
        "--min-df", 
        type=int, 
        default=2, 
        help="Min token df, i.e., in how many documents token must appear (default: 2)"
    )
    tfidf_channel.add_argument(
        "--max-df", 
        type=float, 
        default=0.9, 
        help="Max token df fraction, i.e., drop tokens appearing in more documents in percents (default: 0.9)"
    )
    tfidf_channel.add_argument(
        "--keep-numeric", 
        action="store_true", 
        help="Do not drop numeric tokens"
    )
    tfidf_channel.add_argument(
        "--no-lowercase", 
        action="store_true", 
        help="Do not lowercase before tokenization"
    )
    tfidf_channel.add_argument(
        "--keep-stopwords", 
        action="store_true", 
        help="Do not drop stopwords"
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
        help="Min ngram to be extracted (default: 1)"
    )
    tfidf_channel.add_argument(
        "--ngram-max", 
        type=int, 
        default=1, 
        help="Max ngram to be extracted (default: 1 - meaning extraction of unigrams only)"
    )
    tfidf_channel.add_argument(
        "--min-ngram-df", 
        type=int, 
        default=2, 
        help="Min df of ngram, i.e., in how many documents ngram must appear (default: 2)"
    )
    tfidf_channel.add_argument(
        "--use-corpus",
        action="store_true",
        help="Use global corpus across all silver comments (default: False)"
    )
    tfidf_channel.add_argument(
        "--keep-sentiment",
        action="store_true",
        help="Keep sentiment words in final result (default: False)"
    )
    tfidf_channel.add_argument(
        "--stemming-mode",
        default="none", 
        help="Normalize words so related forms (-ing, -ed, -s) collapse into one base form (default: none)\nCurrent supported mods: stem_en"
    )
    tfidf_channel.set_defaults(func=run_tfidf_channel)
    
    # DISTINCTIVE-KEYWORDS
    distinctive_kws = subparser.add_parser(
        "distinctive-keywords", 
        help="Compute distinctive keywords for video from channel"
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
        help="Print compact report from existing Gold channel artifacts"
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
        help="Optional YouTube video URL or 11-character video ID for distinctive keywords section"
    )
    report_channel.add_argument(
        "--top-k", 
        type=int, 
        default=30, 
        help="Top K keywords (default: 30)"
    )
    report_channel.set_defaults(func=run_report_channel)
    
    return parser