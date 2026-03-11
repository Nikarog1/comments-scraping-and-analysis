from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import math

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.analysis.tfidf.service import TfidfService




def _write_silver_comments(path: Path, texts: list[str | None], preprocess_version) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version] * len(texts))
        }
    )
    pq.write_table(table, path)
    
def test_tfidf_service_computes_expected_keywords(tmp_path: Path) -> None:
    video_id = "vid1"
    silver_path = tmp_path / "data" / "silver" / video_id / "comments.parquet"
    _write_silver_comments(
        silver_path,
        [
            "i love that cat",
            "this cat is amazing",
            "amazing winter love it"
        ],
        preprocess_version="v1"
    )
    
    service = TfidfService()
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
    result = service.compute_for_video(
        video_id=video_id,
        silver_parquet_path=str(silver_path),
        config=cfg,
        created_at_utc=datetime(2026,1,1,12,0,0, tzinfo=timezone.utc),
        batch_size=2,
        unfilter_sentiment=False
    )
    
    assert result.video_id == video_id
    assert result.row_count == 3
    assert result.empty_text_count == 0
    assert result.doc_count_non_empty == 3
    assert result.vocab_size == 4
    
    # Expected tokens after stopwords removal:
    # C1: ["love", "cat"]
    # C2: ["cat", "amazing"]
    # C3: ["amazing", "winter", "love"]
    
    got = {kw.token: kw for kw in result.keywords}
    assert set(got) == {"love", "cat", "amazing", "winter"}
    
    # df checks:
    assert got["cat"].df == 2
    assert got["love"].df == 2
    assert got["amazing"].df == 2
    assert got["winter"].df == 1
    
    # idf checks:
    N = 3
    idf_df_2 = math.log((1 + N) / (1 + 2)) + 1
    idf_df_1 = math.log((1 + N) / (1 + 1)) + 1
    
    assert math.isclose(got["cat"].idf, idf_df_2, rel_tol=1e-9) # isclose used due to floating-point rounding
    assert math.isclose(got["love"].idf, idf_df_2, rel_tol=1e-9) 
    assert math.isclose(got["amazing"].idf, idf_df_2, rel_tol=1e-9)
    assert math.isclose(got["winter"].idf, idf_df_1, rel_tol=1e-9)
    
    # avg_tf checks:
    # cat: (1 / 2 + 1 / 2) / 3 = 1 / 3
    # love: (1 / 2 + 1 / 3) / 3 = 5 / 18
    # amazing: (1 / 2 + 1 / 3) / 3 = 5 / 18
    # winter: (1 / 3) / 3 = 1 / 9
    assert math.isclose(got["cat"].avg_tf, 1 / 3, rel_tol=1e-9)
    assert math.isclose(got["love"].avg_tf, 5 / 18, rel_tol=1e-9)
    assert math.isclose(got["amazing"].avg_tf, 5 / 18, rel_tol=1e-9)
    assert math.isclose(got["winter"].avg_tf, 1 / 9, rel_tol=1e-9)
    
    # ranking:
    # cat highest
    # amazing and love have equal score and df, but alphabetically amazing is higher (asc order)
    # video last
    assert tuple(kw.token for kw in result.keywords) == ("cat", "amazing", "love", "winter")
    
    def test_tfidf_service_applies_min_df_filter(tmp_path: Path) -> None:
        video_id = "vid1"
        silver_path = tmp_path / "data" / "silver" / video_id / "comments.parquet"
        _write_silver_comments(
            silver_path,
            [
                "i love that cat",
                "this cat is amazing",
                "amazing winter love it"
            ],
            preprocess_version="v1"
        )
        
        service = TfidfService()
        cfg = TfidfConfig(
            top_k=10,
            min_token_len=2,
            drop_numeric_tokens=True,
            lowercase=True,
            drop_stopwords=True,
            stopwords_lang="en",
            min_df=2,
            max_df=1.0,
            tf_mode="norm",
            idf_mode="smooth_log_plus1_ln",
        )
        result = service.compute_for_video(
            video_id=video_id,
            silver_parquet_path=str(silver_path),
            config=cfg,
            created_at_utc=datetime(2026,1,1,12,0,0, tzinfo=timezone.utc),
            batch_size=2,
            unfilter_sentiment=False,
        )
        
        assert result.vocab_size == 3
        assert result.min_df_abs == 2
        assert tuple(kw.token for kw in result.keywords) == ("cat", "amazing", "love")
        
    def test_tfidf_service_handles_empty_corpus_after_filtering(tmp_path: Path) -> None:
        video_id = "vid1"
        silver_path = tmp_path / "data" / "silver" / video_id / "comments.parquet"
        _write_silver_comments(
            silver_path,
            [
                "",
                None,
                "1 22 333"
            ],
            preprocess_version="v1"
        )
        
        service = TfidfService()
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
        result = service.compute_for_video(
            video_id=video_id,
            silver_parquet_path=str(silver_path),
            config=cfg,
            created_at_utc=datetime(2026,1,1,12,0,0, tzinfo=timezone.utc),
            batch_size=2,
            unfilter_sentiment=False,
        )
        
        assert result.row_count == 3
        assert result.empty_text_count == 3
        assert result.doc_count_non_empty == 0
        assert result.vocab_size == 0
        assert result.keywords == ()
        
