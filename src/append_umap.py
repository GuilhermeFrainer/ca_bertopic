from typing import Optional

import numpy as np
import polars as pl
from umap import UMAP


class AppendUMAP(UMAP):
    """
    A custom UMAP class that extends the standard UMAP functionality to concatenate
    user-specified dimensions (metadata) to the input embeddings BEFORE running
    dimensionality reduction.

    Args:
        metadata (np.ndarray):
            An array containing metadata dimensions to be concatenated
            to the input document embeddings.
        **kwargs:
            Any additional keyword arguments to be passed to the
            underlying umap.UMAP constructor, such as n_components,
            n_neighbors, min_dist, etc.
    """

    def __init__(self, metadata: Optional[np.ndarray] = None, **kwargs):
        # Call the constructor of the parent UMAP class, passing all other kwargs.
        super().__init__(**kwargs)

        # Store the metadata for later use in concatenation.
        self.metadata = metadata

    def _concatenate_metadata(self, X: np.ndarray) -> np.ndarray:
        """
        Concatenates the metadata to the input document embeddings.
        """
        if self.metadata is not None:
            if X.shape[0] != self.metadata.shape[0]:
                raise ValueError(
                    f"Shape mismatch: X has {X.shape[0]} samples, "
                    f"but metadata has {self.metadata.shape[0]} samples."
                )
            return np.hstack((X, self.metadata))
        return X

    def fit(self, X, y=None):
        """
        Fits the UMAP model on the concatenated embeddings and metadata.
        """
        X_combined = self._concatenate_metadata(X)
        super().fit(X_combined, y)
        return self

    def transform(self, X) -> np.ndarray:
        """
        Transforms the data into the embedding space after concatenating metadata.
        """
        X_combined = self._concatenate_metadata(X)
        return super().transform(X_combined)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """
        Fits the data and transforms it into the embedding space on the
        concatenated input.
        """
        X_combined = self._concatenate_metadata(X)
        return super().fit_transform(X_combined, y)

    @staticmethod
    def shape_dims(df: pl.DataFrame) -> np.ndarray:
        """
        Extracts and stacks specified columns into a 2D NumPy feature matrix.

        This method converts each selected column into a vertical vector and
        horizontally stacks them, creating a format suitable for dimensionality
        reduction algorithms (like UMAP).

        Parameters
        ----------
        df : pl.DataFrame
            The source Polars DataFrame containing the data.
            Dataframe must contain only the desired metadata.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_rows, n_cols) containing the stacked features.
        """
        import polars.selectors as cs

        # Skips categorical variables
        df_clean = df.select(~cs.by_dtype(pl.String))

        if len(df.columns) != len(df_clean.columns):
            print(
                f"Warning: Dropped {len(df.columns) - len(df_clean.columns)} non-numeric columns."
            )

        # Casts DateTimes to Float64
        df_clean = df_clean.with_columns(
            cs.by_dtype(pl.Datetime, pl.Date).cast(pl.Int64).cast(pl.Float64)
        )

        cols = df_clean.columns
        reoriented_dims = tuple(df_clean[c].to_numpy().reshape(-1, 1) for c in cols)
        return np.hstack(reoriented_dims)
