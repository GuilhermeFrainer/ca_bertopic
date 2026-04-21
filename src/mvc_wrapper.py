from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class MVCWrapper(BaseEstimator, ClusterMixin):
    metadata: np.ndarray
    labels_: Optional[np.ndarray]

    def __init__(self, model, metadata: np.ndarray):
        self.model = model
        self.metadata = metadata
        self.labels_ = None

    def fit(self, X, y=None):
        if not len(X) == len(self.metadata):
            raise ValueError(
                f"Shape mismatch between text data and metadata: {len(X)} vs {len(self.metadata)}"
            )

        Xs = [X, self.metadata]
        self.model.fit(Xs)

        self.labels_ = self.model.labels_
        return self

    def predict(self, X):
        if not len(X) == len(self.metadata):
            raise ValueError(
                f"Metadata and textual embeddings must have the same length. Found {len(X)} and {len(self.metadata)}"
            )
        Xs = [X, self.metadata]
        return self.model.predict(Xs)


class AlignedUMAPWrapper(BaseEstimator):
    relations: dict
    metadata: np.ndarray
    training_embeddings: Optional[np.ndarray]

    def __init__(self, model, metadata: np.ndarray):
        self.model = model
        self.metadata = metadata
        self.training_embeddings = None

    def fit(self, X, y=None):
        # Relations mapping needed for AlignUMAP
        # In our case, it's just the identity function,
        # since metadata row i refers to data row i
        relation_dict = {i: i for i in range(len(X))}
        # Only one dict needed, since it maps rows of data "slice"
        # i to rows in slice i + 1
        relations = [relation_dict]
        Xs = [X, self.metadata]

        self.model.fit(Xs, relations=relations)

        # As AlignedUMAP does not have a 'transform' method,
        # we need to store the embeddings generated during the
        # 'fit' call
        # In a way, this mimicks the 'fit_transform' method,
        # and shouldn't be an issue if we are only generating
        # embeddings for data during training
        # We take only the embeddings at index 0, which seem to refer
        # to X.
        # I don't know why we should do this in particular, but it was
        # suggested by Gemini and it's experimental. Also, I don't know
        # what the alternative would be.
        self.training_embeddings = self.model.embeddings_[0]

        return self

    def transform(self, X):
        # Return the pre-calculated embeddings
        # BERTopic calls fit(X).transform(X) internally.
        # We simply return what we calculated in fit().
        if self.training_embeddings is not None:
            return self.training_embeddings
        else:
            raise ValueError("Model has not been fitted yet.")
