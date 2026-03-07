from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.tfidf.service import TfidfService
from yt_comments.analysis.tfidf.models import TfidfConfig


def _write_silver(tmp_path, comments):
    table = pa.table({"text_clean": comments})
    path = tmp_path / "silver.parquet"
    pq.write_table(table, path)
    return str(path)

def test_tfidf_with_bigrams(tmp_path: Path):
    comments = [
        "amazing cat",
        "funny dog",
        "cool vid"
    ]
    silver_path = _write_silver(tmp_path, comments)
    
    cfg = TfidfConfig(
        top_k=10,
        min_token_len=2,
        drop_numeric_tokens=True,
        lowercase=True,
        drop_stopwords=True,
        lang="en",
        min_df=1,
        max_df=1.0,
        tf_mode="norm",
        idf_mode="smooth_log_plus1_ln",
        ngram_range=(1,2),
    )
    service = TfidfService(preprocess_version="v1")
    
    result = service.compute_for_video(
        video_id="id1",
        silver_parquet_path=silver_path,
        config=cfg,
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        batch_size=2
    )
    
    tokens = {kw.token for kw in result.keywords}
    
    assert "amazing cat" in tokens
    assert "funny dog" in tokens
    assert "vid" not in tokens