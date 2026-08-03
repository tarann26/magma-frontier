"""The shared representation every attack and the value estimator read.

Everything here is fitted on the training rows only. A vocabulary, an IDF weighting, a
scaler or an SVD basis fitted over held-out rows leaks information across the split and
inflates every downstream number.

The n-gram block is TF-IDF weighted because raw rates over-weight tools that appear in
nearly every session and separate nobody, while rare sequences carry the signal. The
scalar block is standardized separately since the two live on incomparable scales.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler

_NGRAM_PREFIXES = ("uni:", "bi:", "tri:")


@dataclass(frozen=True, slots=True)
class Representation:
    Z: np.ndarray
    n_components: int
    explained_variance: float


def _blocks(fs) -> tuple[np.ndarray, np.ndarray]:
    """Split the feature matrix into (scalar block, n-gram block) by column name."""
    is_ngram = np.array(
        [name.startswith(_NGRAM_PREFIXES) for name in fs.feature_names], dtype=bool
    )
    return fs.X[:, ~is_ngram], fs.X[:, is_ngram]


def fit_transform(fs, train_idx: np.ndarray, n_components: int = 128,
                  seed: int = 0) -> Representation:
    """Fit TF-IDF, scaling and SVD on `train_idx` rows; transform all rows."""
    train_idx = np.asarray(train_idx, dtype=int)
    # Two rows minimum: with one, sklearn's explained_variance_ratio_ divides zero
    # variance by zero variance and returns NaN without raising, which would put a
    # silently meaningless diagnostic into a published table.
    if train_idx.size < 2:
        raise ValueError(
            f"fit_transform() needs at least 2 training rows, got {train_idx.size}"
        )

    scalars, ngrams = _blocks(fs)

    scaler = StandardScaler().fit(scalars[train_idx])
    scaled = scaler.transform(scalars)

    if ngrams.shape[1] > 0:
        tfidf = TfidfTransformer().fit(ngrams[train_idx])
        weighted = tfidf.transform(ngrams).toarray()
        dense = np.hstack([scaled, weighted])
    else:
        dense = scaled

    # SVD cannot produce more components than the smaller of the fitted rows and columns,
    # and TruncatedSVD requires strictly fewer components than features.
    usable = min(n_components, dense.shape[1] - 1, train_idx.size - 1)
    usable = max(1, usable)

    svd = TruncatedSVD(n_components=usable, random_state=seed).fit(dense[train_idx])
    Z = svd.transform(dense)

    return Representation(
        Z=Z,
        n_components=usable,
        explained_variance=float(svd.explained_variance_ratio_.sum()),
    )
