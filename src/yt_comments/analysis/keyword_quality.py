from yt_comments.analysis.tfidf.models import TfidfKeyword
from yt_comments.nlp.stopwords import SENTIMENT_EN_VOCABULARY


KEYWORD_QUALITY_VERSION = "v1" # sentiment keywords list to unfilter, if anything is changed / added the version here should be changed too

def filter_keywords(keywords: list[TfidfKeyword]) -> list[TfidfKeyword]:
    out: list[TfidfKeyword] = []
    
    for keyword in keywords:
        tokens = keyword.token.split()
        if any(tok in SENTIMENT_EN_VOCABULARY for tok in tokens):
            continue
        out.append(keyword)
        
    return out
            

    