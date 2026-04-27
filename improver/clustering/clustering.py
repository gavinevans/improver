# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform clustering on DataFrames using scikit-learn or kmedoids."""

from typing import Any

import numpy as np
import pandas as pd

from improver import BasePlugin


class _RandomStateWithReplacement:
    """Wrapper around numpy RandomState that forces replace=True in choice method.

    This wrapper is needed to work around kmedoids library calling
    random_state.choice() with replace=False hardcoded. When kmedoids needs
    to select more initial medoids than samples exist, this wrapper allows
    sampling with replacement so the operation succeeds.
    """

    def __init__(self, random_state: int | np.random.RandomState | None = None):
        """Initialise the wrapper with a RandomState.

        Args:
            random_state: Seed (int), numpy RandomState object, or None.
        """
        if isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        else:
            # Create a new RandomState from seed (int or None)
            self._random_state = np.random.RandomState(random_state)

    def choice(self, a: int, size: int, replace: bool = False, p=None):
        """Override choice to force replace=True when needed.

        Args:
            a: Population size or array to choose from.
            size: Number of samples to draw.
            replace: Requested replacement setting (ignored, always set to True
                if size > a when a is an integer).
            p: Probabilities for each element.

        Returns:
            Samples drawn with replacement if needed (size > a).
        """
        # If size > population and a is an integer (not an array),
        # force replace=True to allow sampling more items than population
        if isinstance(a, int) and size > a:
            return self._random_state.choice(a, size=size, replace=True, p=p)
        return self._random_state.choice(a, size=size, replace=replace, p=p)

    def __getattr__(self, name):
        """Delegate all other method calls to the underlying RandomState."""
        return getattr(self._random_state, name)


class FitClustering(BasePlugin):
    """Class to perform clustering on DataFrames using scikit-learn or kmedoids.

    This plugin provides a unified interface for applying various clustering algorithms
    to pandas DataFrames. It supports clustering methods from scikit-learn's cluster
    module as well as the KMedoids algorithm from the kmedoids package.
    The plugin automatically selects the appropriate package based on the specified
    clustering method:
    - "KMedoids": Uses the kmedoids package
    - All other methods: Uses sklearn.cluster
    """

    def __init__(self, clustering_method: str, **kwargs: Any) -> None:
        """Initialise the clustering plugin.

        Args:
            clustering_method: The name of the clustering method to use.
                Must be either "KMedoids" (from kmedoids package) or a valid
                clustering class name from sklearn.cluster (e.g., "KMeans",
                "DBSCAN", "AgglomerativeClustering").
            **kwargs: Additional keyword arguments to pass to the clustering
                algorithm. These are method-specific parameters. Common examples:
                - n_clusters (int): Number of clusters (for KMeans,
                AgglomerativeClustering)
                - random_state (int): Random seed for reproducibility
                Refer to the scikit-learn or kmedoids documentation for the complete
                list of parameters for each clustering method.
        Raises:
            ValueError: If the specified clustering method is not found in
                sklearn.cluster or kmedoids packages.
        """
        self.clustering_method = clustering_method
        self.kwargs = kwargs

    def process(self, df: pd.DataFrame) -> Any:
        """Apply the clustering method to the DataFrame. Fits the specified clustering
        algorithm to the input DataFrame and returns the fitted clustering model.

        Args:
            df: The input DataFrame to cluster. Each row represents
                a sample and each column represents a feature. The DataFrame should
                contain numeric data suitable for the chosen clustering algorithm.
        Returns:
            A fitted clustering model object from either sklearn.cluster or kmedoids.
            The returned object will have at minimum a `labels_` attribute containing
            the cluster assignment for each sample. Additional attributes depend on
            the specific clustering method used (e.g., `cluster_centers_` for KMeans,
            `core_sample_indices_` for DBSCAN).
        Raises:
            ValueError: If the specified clustering method is not found in
                sklearn.cluster or is not "KMedoids".
        """
        # Use kmedoids directly if requested
        if self.clustering_method == "KMedoids":
            import kmedoids

            # Set default metric to euclidean if not specified
            kwargs = self.kwargs.copy()
            if "metric" not in kwargs:
                kwargs["metric"] = "euclidean"

            # Wrap random_state to force replace=True in choice method if provided
            if "random_state" in kwargs:
                kwargs["random_state"] = _RandomStateWithReplacement(
                    kwargs["random_state"]
                )

            clustering_class = getattr(kmedoids, self.clustering_method)
            # Convert DataFrame to numpy array for kmedoids
            return clustering_class(**kwargs).fit(df.values)

        # Otherwise, use sklearn
        from sklearn import cluster

        if hasattr(cluster, self.clustering_method):
            clustering_class = getattr(cluster, self.clustering_method)
            return clustering_class(**self.kwargs).fit(df)
        else:
            msg = (
                f"The clustering method '{self.clustering_method}' is not supported. "
                "Please check sklearn.cluster documentation for available methods."
            )
            raise ValueError(msg)
