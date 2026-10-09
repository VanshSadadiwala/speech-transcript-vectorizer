"""Feature models: Bag-of-Words + PMI collocation features.

Classes
-------
CollocationExtractor      finds statistically strong bigrams / trigrams using
                          nltk.collocations (PMI association measure)
AugmentedBoWVectorizer    CountVectorizer BoW matrix  +  one extra column per
                          selected collocation  ->  scipy sparse matrix

Formula (pointwise mutual information):
    PMI(x, y) = log2( P(x, y) / (P(x) * P(y)) )
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.collocations import (BigramAssocMeasures, BigramCollocationFinder,
                               TrigramAssocMeasures, TrigramCollocationFinder)
from sklearn.feature_extraction.text import CountVectorizer


def _identity(x):
    return x


def count_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Count adjacent n-grams inside one document."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


class CollocationExtractor:
    """Extract collocations with PMI.

    Parameters
    ----------
    min_freq       n-grams seen fewer times than this are ignored (PMI is
                   unreliable for rare pairs, so this filter is essential)
    pmi_threshold  keep n-grams with PMI >= threshold
    ngram_sizes    (2,) bigrams, (3,) trigrams, (2, 3) both
    max_collocations  optional cap, keeps the highest-PMI ones
    """

    def __init__(self, min_freq: int = 2, pmi_threshold: float = 3.0,
                 ngram_sizes: Tuple[int, ...] = (2,), max_collocations: int | None = None):
        self.min_freq = min_freq
        self.pmi_threshold = pmi_threshold
        self.ngram_sizes = tuple(ngram_sizes)
        self.max_collocations = max_collocations
        self.scored_: pd.DataFrame = pd.DataFrame(columns=["phrase", "n", "count", "pmi"])
        self.selected_: List[Tuple[str, ...]] = []

    # ---- fitting ----
    def fit(self, token_docs: Iterable[Sequence[str]]) -> "CollocationExtractor":
        docs = [list(d) for d in token_docs]
        rows = []
        for n in self.ngram_sizes:
            if n == 2:
                finder = BigramCollocationFinder.from_documents(docs)
                measures = BigramAssocMeasures.pmi
            elif n == 3:
                finder = TrigramCollocationFinder.from_documents(docs)
                measures = TrigramAssocMeasures.pmi
            else:
                raise ValueError("ngram_sizes may only contain 2 and/or 3")
            finder.apply_freq_filter(self.min_freq)
            for gram, score in finder.score_ngrams(measures):
                rows.append(("_".join(gram), n, finder.ngram_fd[gram], float(score)))
        df = pd.DataFrame(rows, columns=["phrase", "n", "count", "pmi"])
        self.scored_ = df.sort_values(["pmi", "count"], ascending=False).reset_index(drop=True)
        self._select()
        return self

    def _select(self) -> None:
        df = self.scored_
        df = df[df["pmi"] >= self.pmi_threshold]
        if self.max_collocations:
            df = df.head(self.max_collocations)
        self.selected_df_ = df.reset_index(drop=True)
        self.selected_ = [tuple(p.split("_")) for p in self.selected_df_["phrase"]]

    def set_threshold(self, pmi_threshold: float) -> "CollocationExtractor":
        """Re-select with a new threshold without re-scoring (cheap sweeps)."""
        self.pmi_threshold = pmi_threshold
        self._select()
        return self

    # ---- transform ----
    @property
    def feature_names_(self) -> List[str]:
        return ["_".join(g) for g in self.selected_]

    def transform(self, token_docs: Iterable[Sequence[str]]) -> sp.csr_matrix:
        index: Dict[Tuple[str, ...], int] = {g: i for i, g in enumerate(self.selected_)}
        sizes = sorted({len(g) for g in self.selected_})
        rows, cols, vals = [], [], []
        n_docs = 0
        for r, doc in enumerate(token_docs):
            n_docs += 1
            for n in sizes:
                for gram, c in count_ngrams(doc, n).items():
                    j = index.get(gram)
                    if j is not None:
                        rows.append(r); cols.append(j); vals.append(c)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, len(index)), dtype=np.int64)


class AugmentedBoWVectorizer:
    """BoW (term-frequency) matrix, optionally augmented with PMI collocation columns.

    Final matrix = [ BoW columns | collocation columns ]
    """

    def __init__(self, min_df: int = 1, max_features: int | None = None,
                 use_collocations: bool = True, min_freq: int = 2,
                 pmi_threshold: float = 3.0, ngram_sizes: Tuple[int, ...] = (2,),
                 max_collocations: int | None = None, binary: bool = False):
        self.min_df = min_df
        self.max_features = max_features
        self.use_collocations = use_collocations
        self.min_freq = min_freq
        self.pmi_threshold = pmi_threshold
        self.ngram_sizes = ngram_sizes
        self.max_collocations = max_collocations
        self.binary = binary

    def fit(self, token_docs: Iterable[Sequence[str]]) -> "AugmentedBoWVectorizer":
        docs = [list(d) for d in token_docs]
        self.cv_ = CountVectorizer(analyzer=_identity, min_df=self.min_df,
                                   max_features=self.max_features, binary=self.binary)
        self.cv_.fit(docs)
        self.raw_vocab_size_ = len({w for d in docs for w in d})
        self.collocations_ = None
        if self.use_collocations:
            self.collocations_ = CollocationExtractor(
                self.min_freq, self.pmi_threshold, self.ngram_sizes, self.max_collocations).fit(docs)
        self.n_docs_fit_ = len(docs)
        return self

    def transform(self, token_docs: Iterable[Sequence[str]]) -> sp.csr_matrix:
        docs = [list(d) for d in token_docs]
        X = self.cv_.transform(docs).astype(np.int64)
        if self.collocations_ is not None and self.collocations_.selected_:
            C = self.collocations_.transform(docs)
            if self.binary:
                C.data[:] = 1
            X = sp.hstack([X, C], format="csr")
        return X

    def fit_transform(self, token_docs):
        docs = [list(d) for d in token_docs]
        return self.fit(docs).transform(docs)

    @property
    def bow_feature_names_(self) -> List[str]:
        return list(self.cv_.get_feature_names_out())

    @property
    def collocation_feature_names_(self) -> List[str]:
        return self.collocations_.feature_names_ if self.collocations_ is not None else []

    @property
    def feature_names_(self) -> List[str]:
        return self.bow_feature_names_ + self.collocation_feature_names_

    def to_dataframe(self, X: sp.spmatrix, index=None) -> pd.DataFrame:
        return pd.DataFrame(X.toarray(), columns=self.feature_names_, index=index)

    def stats(self, X: sp.spmatrix) -> dict:
        n_rows, n_cols = X.shape
        n_bow = len(self.bow_feature_names_)
        n_col = len(self.collocation_feature_names_)
        total = n_rows * n_cols
        return {
            "documents": n_rows,
            "vocab_size_before_pruning": self.raw_vocab_size_,
            "bow_features": n_bow,
            "collocation_features": n_col,
            "total_features": n_cols,
            "non_zero_entries": int(X.nnz),
            "sparsity_pct": round(100 * (1 - X.nnz / total), 2) if total else 0.0,
            "avg_nonzero_per_doc": round(X.nnz / max(n_rows, 1), 2),
        }
