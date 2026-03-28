from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import math

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.features import hash_config
from yt_comments.analysis.channel_tfidf.service import ChannelTfidfService
from yt_comments.analysis.corpus.models import CorpusDfTable, CorpusTokenStat
from yt_comments.analysis.keyword_quality import KEYWORD_QUALITY_VERSION
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository


def _write_silver_comments(silver_dir: Path, texts: list[str | None], preprocess_version: str) -> None:
    path = silver_dir / "comments.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version] * len(texts))
        }
    )
    pq.write_table(table, path)
    
def test_channel_tfidf_service_computes_keywords_without_global_corpus(tmp_path: Path):
    channel_id = "chan123"
    video_ids = ("v1", "v2", "v3")
    silver_dir = tmp_path / "data" / "silver" 
    
    _write_silver_comments(
        silver_dir / video_ids[0],
        [
            "i love that cat",
            "this cat is amazing",
            "amazing winter love it"
        ],
        preprocess_version="v1"
    )
    
    _write_silver_comments(
        silver_dir / video_ids[1],
        [
            "beautiful cat",
            "",
        ],
        preprocess_version="v1"
    )
    
    _write_silver_comments(
        silver_dir / video_ids[2],
        [
            "love that cat",
            "amazing",
        ],
        preprocess_version="v1"
    )

    cfg = TfidfConfig(
        top_k=30,
        min_token_len=2,
        drop_numeric_tokens=True,
        lowercase=True,
        drop_stopwords=True,
        stopwords_lang="en",
        min_df=1,
        max_df=1.0,
        tf_mode="norm",
        idf_mode="smooth_log_plus1_ln",
    )
    silver_repo = ParquetSilverCommentsRepository(silver_dir)
    
    service = ChannelTfidfService()
    result = service.compute_for_channel(
        channel_id=channel_id,
        video_ids=video_ids,
        config=cfg,
        silver_repo=silver_repo,
        global_corpus=None,
        created_at_utc=None,
        unfilter_sentiment=False,
        batch_size=3   
    )
    
    assert result.channel_id == "chan123"
    assert result.video_ids == video_ids
    assert result.preprocess_version == "v1"
    assert result.row_count == 7
    assert result.empty_text_count == 1
    assert result.doc_count_non_empty == 6
    assert result.vocab_size == 5
    
    # Expected tokens after stopwords removal:
    # C1: ["love", "cat"]
    # C2: ["cat", "amazing"]
    # C3: ["amazing", "winter", "love"]
    # C4: ["beautiful", "cat"]
    # C6: ["love", "cat"] 
    # C7: ["amazing"] 
    
    got = {kw.token: kw for kw in result.keywords}
    assert set(got) == {"cat", "amazing", "love", "winter", "beautiful"}

    # df checks:
    assert got["cat"].df == 4
    assert got["amazing"].df == 3
    assert got["love"].df == 3
    assert got["winter"].df == 1
    assert got["beautiful"].df == 1
    
    # idf checks:
    N = 6 # empty comm doesn't count for N
    idf_df_4 = math.log((1 + N) / (1 + 4)) + 1
    idf_df_3 = math.log((1 + N) / (1 + 3)) + 1
    idf_df_1 = math.log((1 + N) / (1 + 1)) + 1
    
    assert math.isclose(got["cat"].idf, idf_df_4, rel_tol=1e-9) # isclose used due to floating-point rounding
    assert math.isclose(got["love"].idf, idf_df_3, rel_tol=1e-9) 
    assert math.isclose(got["amazing"].idf, idf_df_3, rel_tol=1e-9)
    assert math.isclose(got["winter"].idf, idf_df_1, rel_tol=1e-9)
    assert math.isclose(got["beautiful"].idf, idf_df_1, rel_tol=1e-9)
    
    # avg_tf checks:
    # amazing: (1 / 2 + 1 / 3 + 1) / 6 = 11 / 36
    # cat: (1 / 2 + 1 / 2 + 1 / 2 + 1 / 2) / 6 = 2 / 6
    # love: (1 / 2 + 1 / 3 + 1 / 2) / 6 = 4 / 18
    # winter: (1 / 3) / 6 = 1 / 18
    # beautiful: (1 / 2) / 6 = 1 / 12
    assert math.isclose(got["amazing"].avg_tf, 11 / 36, rel_tol=1e-9)
    assert math.isclose(got["cat"].avg_tf, 2 / 6, rel_tol=1e-9)
    assert math.isclose(got["love"].avg_tf, 4 / 18, rel_tol=1e-9)
    assert math.isclose(got["winter"].avg_tf, 1 / 18, rel_tol=1e-9)
    assert math.isclose(got["beautiful"].avg_tf, 1 / 12, rel_tol=1e-9)
    
    # ranking:
    # amazing highest
    # cat
    # love
    # beautiful and winter have equal score and df, but alphabetically beautiful is higher (asc order)
    assert tuple(kw.token for kw in result.keywords) == ("amazing", "cat", "love", "beautiful", "winter")
    

def test_channel_tfidf_service_with_global_corpus(tmp_path: Path):
    channel_id = "chan123"
    video_ids = ("v1", "v2")
    silver_dir = tmp_path / "data" / "silver"

    _write_silver_comments(
        silver_dir / video_ids[0],
        [
            "cat cat",
            "dog",
        ],
        preprocess_version="v1",
    )

    _write_silver_comments(
        silver_dir / video_ids[1],
        [
            "cat",
            "bird",
        ],
        preprocess_version="v1",
    )

    # global corpus (intentionally different DF)
    corpus = CorpusDfTable(
        artifact_version="v3",
        preprocess_version="v1",
        config_hash=hash_config({
            "config": asdict(TfidfConfig(
                top_k=10,
                min_token_len=2,
                drop_numeric_tokens=True,
                lowercase=True,
                drop_stopwords=True,
                stopwords_lang="en",
                min_df=1,
                max_df=1.0,
                tf_mode="norm",
                idf_mode="smooth_log_plus1_ln",
            )),
            "keywords_version": KEYWORD_QUALITY_VERSION,
        }),
        video_count=100,  
        tokens=(
            CorpusTokenStat(token="cat", df_videos=50),
            CorpusTokenStat(token="dog", df_videos=10),
            CorpusTokenStat(token="bird", df_videos=5),
        ),
    )

    cfg = TfidfConfig(
        top_k=10,
        min_token_len=2,
        drop_numeric_tokens=True,
        lowercase=True,
        drop_stopwords=True,
        stopwords_lang="en",
        min_df=1,
        max_df=1.0,
        tf_mode="norm",
        idf_mode="smooth_log_plus1_ln",
    )

    silver_repo = ParquetSilverCommentsRepository(silver_dir)

    service = ChannelTfidfService()
    result = service.compute_for_channel(
        channel_id=channel_id,
        video_ids=video_ids,
        config=cfg,
        silver_repo=silver_repo,
        global_corpus=corpus,
        unfilter_sentiment=False,
    )

    # Expected tokens after stopwords removal:
    # C1: ["cat", "cat"]
    # C2: ["dog"]
    # C3: ["cat"]
    # C4: ["bird"]

    got = {kw.token: kw for kw in result.keywords}
    assert set(got) == {"cat", "dog", "bird"}

    # DF should come from GLOBAL corpus, not local
    assert got["cat"].df == 50
    assert got["dog"].df == 10
    assert got["bird"].df == 5

    # IDF should use corpus N (100), not local (4)
    idf_cat = math.log((1 + 100) / (1 + 50)) + 1
    idf_dog = math.log((1 + 100) / (1 + 10)) + 1
    idf_bird = math.log((1 + 100) / (1 + 5)) + 1

    assert math.isclose(got["cat"].idf, idf_cat, rel_tol=1e-9)
    assert math.isclose(got["dog"].idf, idf_dog, rel_tol=1e-9)
    assert math.isclose(got["bird"].idf, idf_bird, rel_tol=1e-9)

    tokens_sorted = tuple(kw.token for kw in result.keywords)

    assert tokens_sorted[0] in {"bird", "dog"}  # rarer tokens win