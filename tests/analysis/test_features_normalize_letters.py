from yt_comments.analysis.features import normalize_repeating_letters, tokenize
from yt_comments.analysis.tfidf.models import TfidfConfig


def test_normalize_repeating_letters():
    w_norm1 = normalize_repeating_letters("cooooooooool")
    w_norm2 = normalize_repeating_letters("hiiiiii")

    assert w_norm1 == "cool"
    assert w_norm2 == "hii"
    
def test_tokenize_with_normalize_repeating_letters():
    
    config = TfidfConfig(
        drop_stopwords=True
    )
    
    tokens = list(tokenize("Hiiiiiiii gooooood content", config=config))
    
    assert "hii" not in tokens # hii is present in stopwords
    assert "good" in tokens
    

    
