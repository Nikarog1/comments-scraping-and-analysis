import hashlib
import json
import re

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast, Iterable

from nltk.stem import SnowballStemmer

import pyarrow.parquet as pq

from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.nlp.stopwords import get_stopwords


_STEMMER = SnowballStemmer("english")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")

def hash_config(config: Any) -> str:
    if is_dataclass(config):
        payload_obj = asdict(cast(Any, config)) # is_dataclass returns true for both data class instance and class
    else: 
        payload_obj = config
    payload = json.dumps(
        payload_obj, 
        sort_keys=True, 
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8") # dicts are not hashable, so need to convert to json string
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
        
def read_preprocess_version(silver_parquet_path: Path | str) -> str:
    """Read preprocess_version from a Silver comments parquet file."""
    table = pq.read_table(silver_parquet_path, columns=["preprocess_version"])

    values = table.column("preprocess_version").to_pylist()
    versions = {v for v in values if v is not None}

    if not versions:
        raise ValueError(f"Missing preprocess_version in Silver file: {silver_parquet_path}")

    if len(versions) != 1:
        raise ValueError(
            f"Multiple preprocess_version values found in Silver file: {silver_parquet_path}"
        )

    return next(iter(versions))

def normalize_token(token: str, *, mode: str) -> str:
    if mode == "none":
        return token 
    if mode == "stem_en":
        return _STEMMER.stem(token)
    
    raise ValueError(f"Unsupported normalization mode: {mode}")