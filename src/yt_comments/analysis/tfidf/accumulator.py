from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class TfidfAccumulator:
    """
    Streaming TF-IDF statistics accumulator.

    Maintains corpus statistics required for TF-IDF computation.
    """
    
    row_count: int = 0
    empty_text_count: int = 0
    doc_count_non_empty: int = 0
    df: dict[str, int] = field(default_factory=lambda: defaultdict(int)) # default_factory has to be "zero-argument callable" that's why is wrapped into labmda func
    sum_tf_norm: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    def add_document(self, tokens: list[str]) -> None:
        """
        Add one document (comment) to the corpus statistics.

        tokens must already be tokenized and filtered.
        """
        
        self.row_count += 1
        
        if not tokens:
            self.empty_text_count += 1
            return 
        
        self.doc_count_non_empty += 1
        
        doc_len = len(tokens)
        counts = Counter(tokens)       
            
        for token, count in counts.items():
            # check performance, might create a local vars for self.df and self.sum_tf_norm before for-loop
            self.df[token] += 1 # repeated tokens in same document count only once
            self.sum_tf_norm[token] += count / doc_len
    