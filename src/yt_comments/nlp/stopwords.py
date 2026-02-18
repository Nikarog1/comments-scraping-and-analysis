from __future__ import annotations



EN_STOPWORDS = frozenset(
    {
        "a","an","and","are","as","at","be","but","by",
        "for","from","has","have","he","her","hers","him","his",
        "i","if","in","into","is","it","its","me","my","not","of",
        "on","or","our","ours","she","so","that","the","their","theirs",
        "them","then","there","these","they","this","those","to","too",
        "up","us","was","we","were","what","when","where","which","who",
        "why","will","with","you","your","yours"
    }
)

def get_stopwords(lang: str) -> frozenset[str]:
    if lang == "en":
        return EN_STOPWORDS
    raise ValueError(f"Unsupported stopwords_lang: {lang!r}")