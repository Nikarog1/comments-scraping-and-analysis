from __future__ import annotations


# Linguistic stopwords: grammar/function words
EN_STOPWORDS = frozenset(
    {
        "a", "am", "an","and","are","as","at","be","but","by",
        "for","from", "just", "has","have","he","her","hers","him","his",
        "i","if","in","into","is","it","its","me","my","not","of",
        "on","or","our","ours","she","so","that","the","their","theirs",
        "them","then","there","these","they","this","those","to","too",
        "up","us","was","we","were","what","when","where","which","who",
        "why","will","with","you","your","yours", "do", "does", "did", 
        "can", "could", "would", "should", "than", "very", "all", "any", 
        "each", "few", "more", "most", "other", "some", "such", "same", 
        "only", "just", "it's", "you're", "i'm", "that's"
    }
)

# Domain stopwords: platform-specific filler words
YOUTUBE_EN_STOPWORDS = frozenset(
    {
        "hi", "hello", "yes", "yeah", "yep", "no", "nah", "nope", "actually", 
        "think", "one", "video", "vid", "videos", "bro", "lol", "hey", "omg",  
        "ok", "okay", "pls", "please", "wow", "damn", "uh", "huh", "channel", 
        "upload", "post", "comment", "comments", "sub", "subscribe", "much",
        "now"
    }
)

# Sentiment vocabulary: not filtered out
SENTIMENT_EN_VOCABULARY = frozenset(
    {
        "amazing", "great", "awesome", "perfect", "beautiful", "favorite",
        "favourite", "nice", "cool", "sweet", "love", "best", "top", "sweet"
    }
)

# Combined
STOPWORDS = {
    "en": EN_STOPWORDS | YOUTUBE_EN_STOPWORDS
}

def get_stopwords(lang: str) -> frozenset[str]:
    try:
        return STOPWORDS[lang]
    except KeyError:
        raise ValueError(f"Unsupported stopwords_lang: {lang!r}")