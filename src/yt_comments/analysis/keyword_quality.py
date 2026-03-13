from yt_comments.analysis.tfidf.models import TfidfKeyword
from yt_comments.nlp.stopwords import SENTIMENT_EN_VOCABULARY


KEYWORD_QUALITY_VERSION = "v1"

def filter_keywords(keywords: list[TfidfKeyword]) -> list[TfidfKeyword]:
    out: list[TfidfKeyword] = []
    
    for keyword in keywords:
        is_ngram = " " in keyword.token
        if not is_ngram and keyword.token in SENTIMENT_EN_VOCABULARY:
            continue
        out.append(keyword)
        
    return out
            

    