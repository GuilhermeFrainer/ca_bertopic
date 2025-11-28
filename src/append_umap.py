from umap import UMAP
import polars as pl
import numpy as np

from typing import Optional


class AppendUMAP(UMAP):
    """
    A custom UMAP class that extends the standard UMAP functionality to include
    user-specified dimensions in the final output. This is useful for
    applications like BERTopic where you may want to include metadata
    alongside the reduced embeddings.

    Args:
        additional_dimensions (np.ndarray):
            An array containing extra dimensions
            to be appended to the output embedding.
        **kwargs: 
            Any additional keyword arguments to be passed to the
            underlying umap.UMAP constructor, such as n_components,
            n_neighbors, min_dist, etc.
    """
    def __init__(self, additional_dimensions: Optional[np.ndarray] = None, **kwargs):
        
        # Call the constructor of the parent UMAP class, passing all other kwargs.
        super().__init__(**kwargs)
        
        # Store the custom parameters for later use in fit_transform.
        self.additional_dimensions = additional_dimensions


    def fit_transform(self, X, y=None) -> np.ndarray: # type: ignore
        """
        Fits the data and transforms it into the embedding space, then appends
        the additional dimensions to the output.

        Args:
            X (np.ndarray):
                The data to be embedded.
            y (np.ndarray, optional):
                A target array used for supervised UMAP,
                    passed to the parent's fit_transform method.
        
        Returns:
            np.ndarray:
                The transformed data with the additional dimensions appended.
        """
        # First, perform the standard UMAP fit and transform.
        embedding = super().fit_transform(X, y)

        # Check if additional dimensions were provided and have the correct shape.
        if self.additional_dimensions is not None:
            if self.additional_dimensions.shape[0] != embedding.shape[0]: # type: ignore
                raise ValueError("The 'additional_dimensions' must have the same number of samples as the data.")
            
            # Concatenate the additional dimensions to the embedding.
            embedding = np.hstack((embedding, self.additional_dimensions)) # type: ignore
        return embedding # type: ignore


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
            print(f"Warning: Dropped {len(df.columns) - len(df_clean.columns)} non-numeric columns.")
        
        # Casts DateTimes to Float64
        df_clean = df_clean.with_columns(
            cs.by_dtype(pl.Datetime, pl.Date).cast(pl.Int64).cast(pl.Float64)
        )
        
        cols = df_clean.columns
        reoriented_dims = tuple(df_clean[c].to_numpy().reshape(-1,1) for c in cols)
        return np.hstack(reoriented_dims)

