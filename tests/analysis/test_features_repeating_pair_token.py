from yt_comments.analysis.features import is_repeating_pair_token, tokenize
from yt_comments.analysis.tfidf.models import TfidfConfig


def test_detection_repeating_pair_tokens():
    
    assert is_repeating_pair_token("lololo") == True
    assert is_repeating_pair_token("xdxdxd") == True
    
    assert is_repeating_pair_token("cat") == False
    assert is_repeating_pair_token("banana") == False
    assert is_repeating_pair_token("lololol") == False
    
def test_tokenize_drop_repeating_pair_tokens():
    
    config = TfidfConfig(
        drop_stopwords=True,
    )
    
    tokens = list(tokenize("hahaha good lololo content", config=config))
    
    assert "lololo" not in tokens
    assert "hahaha" not in tokens
    assert "good" in tokens
    assert "content" in tokens
    