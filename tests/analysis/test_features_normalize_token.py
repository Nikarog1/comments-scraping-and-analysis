from yt_comments.analysis.features import normalize_token


def test_normalize_token_en():
    mode = "stem_en"
    
    assert normalize_token("relaxing", mode=mode) == "relax"
    assert normalize_token("relaxed", mode=mode) == "relax"
    assert normalize_token("relax", mode=mode) == "relax"