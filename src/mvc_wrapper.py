from sklearn.base import ClusterMixin, BaseEstimator
from typing import Optional
import numpy as np


class MVCWrapper(BaseEstimator, ClusterMixin):
    metadata: np.ndarray
    labels_: Optional[np.ndarray]


    def __init__(self, model, metadata: np.ndarray):
        self.model = model
        self.metadata = metadata
        self.labels_ = None


    def fit(self, X, y=None):
        if not len(X) == len(self.metadata):
             raise ValueError(f"Shape mismatch between text data and metadata: {len(X)} vs {len(self.metadata)}")

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
    
