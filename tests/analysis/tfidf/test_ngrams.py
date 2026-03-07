import pytest

from yt_comments.analysis.tfidf.service import TfidfService


def test_generate_unigrams_only():
    tokens = ["love", "beautiful", "cat"]
    
    out = list(
        TfidfService._generate_ngrams(tokens, (1, 1))
    )
    
    assert out == tokens
    
def test_generate_bigrams_only():
    tokens = ["love", "beautiful", "cat"]
    
    out = list(
        TfidfService._generate_ngrams(tokens, (2, 2))
    )
    
    assert out == ["love beautiful", "beautiful cat"]
    
def test_generate_unigrams_and_bigrams():
    tokens = ["love", "beautiful", "cat"]
    
    out = list(
        TfidfService._generate_ngrams(tokens, (1, 2))
    )
    
    assert out == ["love", "beautiful", "cat", "love beautiful", "beautiful cat"]
    
def test_short_documents():
    tokens = ["love"]
    
    out = list(
        TfidfService._generate_ngrams(tokens, (2, 2))
    )
    
    assert out == []
    
def test_invalid_range():
    tokens = ["love", "beautiful", "cat"]

    with pytest.raises(ValueError):
        list(TfidfService._generate_ngrams(tokens, (2, 1)))
