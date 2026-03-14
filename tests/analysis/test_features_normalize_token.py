from dataclasses import asdict

from yt_comments.analysis.features import build_document_features, hash_config, normalize_token, tokenize
from yt_comments.analysis.tfidf.models import TfidfConfig


def test_normalize_token_en():
    mode = "stem_en"
    
    assert normalize_token("relaxing", mode=mode) == "relax"
    assert normalize_token("relaxed", mode=mode) == "relax"
    assert normalize_token("relax", mode=mode) == "relax"
    
def test_normalize_token_none():
    mode = "none"
    
    assert normalize_token("relaxing", mode=mode) == "relaxing"
    
def test_tokenize_with_stemming():
    
    config = TfidfConfig(
        lowercase=True,
        drop_stopwords=False,
        drop_numeric_tokens=True,
        min_token_len=2,
        stopwords_lang="en",
        normalization="stem_en",
        ngram_range=(1,1),
        top_k=10,
    )
    
    tokens = list(tokenize("Relaxing relaxed Relax", config))
    
    assert tokens == ["relax", "relax", "relax"] 
    
def test_ngrams_use_normalized_tokens():
    
    config = TfidfConfig(
        lowercase=True,
        drop_stopwords=False,
        drop_numeric_tokens=True,
        min_token_len=2,
        stopwords_lang="en",
        normalization="stem_en",
        ngram_range=(1,2),
        top_k=10,
    )
    
    features = build_document_features("Relaxing sound Relaxing", config=config)
    
    assert "relax sound" in features
    
def test_config_hash_changes_with_normalization():
    
    cfg1 = TfidfConfig(normalization="none")
    cfg2 = TfidfConfig(normalization="stem_en")
    
    h1 = hash_config({"config": asdict(cfg1)})
    h2 = hash_config({"config": asdict(cfg2)})

    assert h1 != h2
    