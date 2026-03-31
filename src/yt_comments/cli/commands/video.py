from __future__ import annotations

import argparse
import os

from datetime import datetime, timezone
from pathlib import Path

from yt_comments.analysis.features import hash_config
from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.basic_stats.service import BasicStatsService
from yt_comments.analysis.corpus.service import CorpusService
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.analysis.tfidf.service import TfidfService

from yt_comments.cli.helpers import _scrape_video, _silver_parquet_path, logger

from yt_comments.ingestion.video_id_extractor import extract_video_id
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient

from yt_comments.nlp.stopwords import STOPWORDS

from yt_comments.preprocessing.preprocess_service import PreprocessCommentsService
from yt_comments.preprocessing.text_preprocessor import TextPreprocessor

from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.gold_basic_stats_parquet_repository import ParquetBasicStatsRepository
from yt_comments.storage.gold_corpus_df_parquet_repository import ParquetCorpusDfRepository
from yt_comments.storage.gold_tfidf_keywords_parquet_repository import ParquetTfidfKeywordsRepository
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository



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