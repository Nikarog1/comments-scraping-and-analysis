import hashlib
import json
import re

from dataclasses import asdict
from typing import Iterable

from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.nlp.stopwords import get_stopwords


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")

def hash_config(config: BasicStatsConfig | TfidfConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8") # dicts are not hashable, so need to convert to json string
    return hashlib.sha256(payload).hexdigest()[:16]

def build_document_features(text: str, config: TfidfConfig) -> list[str]:
    """
    Build the per-document feature list consumed by the accumulator.

    Frozen v2.1 rule:
    - tokenize first
    - drop stopwords first
    - then generate n-grams from the filtered token stream

    This keeps vocabulary growth under better control and preserves
    deterministic streaming behavior.
    """
    tokens = list(tokenize(text, config))
    return list(generate_ngrams(tokens, config.ngram_range))


def generate_ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> Iterable[str]: 
    min_n, max_n = ngram_range

    if min_n < 1:
        raise ValueError("ngram_range min must be >= 1")
    if max_n < min_n:
        raise ValueError("ngram_range max must be >= 1")
    
    token_count = len(tokens)
    if token_count == 0:
        pass
    
    for n in range(min_n, max_n + 1): # crucial part; (1, 1) - produces unigrams, (1, 2) - both uni- and bigrams, (2, 2) - bigram only
        if token_count < n:
            continue
        
        for i in range(token_count - n + 1):
            yield " ".join(tokens[i : i + n]) # dequeue can be used here, but slice was kept for simplification


def tokenize(text: str, config: BasicStatsConfig | TfidfConfig) -> Iterable[str]:
    if config.lowercase:
        text = text.lower()
        
    stopwords = get_stopwords(config.stopwords_lang) if config.drop_stopwords else frozenset()
    
    for m in _TOKEN_RE.finditer(text): # finditer used since it returns a generator in comparison to findall
        tok = m.group(0) # returns the matched string
        
        if len(tok) < config.min_token_len:
            continue
        if config.drop_numeric_tokens and tok.isdigit():
            continue
        if stopwords and tok in stopwords:
            continue
        
        yield tok