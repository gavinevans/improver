# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform quantile regression using random forests."""

from typing import Optional

import iris
import joblib
import numpy as np
from iris.cube import Cube, CubeList
from quantile_forest import RandomForestQuantileRegressor

from improver.constants import DAYS_IN_YEAR, HOURS_IN_DAY


def prep_feature(
    self, cube: Cube, feature: str, additional_feature_cubes: Optional[CubeList] = None
):
    """Prepare the feature values for the quantile regression random forest model.

    Args:
        cube:
            Cube containing the forecast data.
        feature:
            Name of the feature to be extracted.
        additional_feature_cubes:
            List of additional feature cubes.
    Returns:
        feature_values (numpy.ndarray):
            Flattened array of feature values.
    """
    # Need to revisit how the feature values get broadcast to be the desired shape.
    # The current approach is assuming a 3D cube with a fixed order to the dimensions.

    additional_feature_cube_names = []
    if additional_feature_cubes is not None:
        additional_feature_cube_names = [c.name() for c in additional_feature_cubes]
    collapsed_cube = cube.collapsed(["realization"], iris.analysis.MEAN)

    if "mean" in self.features:
        feature_values = cube.collapsed(
            ["realization"], iris.analysis.MEAN
        ).data.flatten()
    elif "std" in self.features:
        feature_values = cube.collapsed(
            ["realization"], iris.analysis.STD_DEV
        ).data.flatten()
    elif set(["latitude", "longitude", "altitude"]).intersection(self.features):
        coord_multidim = cube.coord(feature).points[np.newaxis, :, np.newaxis]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature == "forecast_period":
        coord_multidim = cube.coord("forecast_period").points[:, np.newaxis, np.newaxis]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
    elif feature in ["day_of_year", "day_of_year_sin", "day_of_year_cos"]:
        time_coord = cube.coord("time").copy()
        day_of_year = np.array([c.point.strftime("%j") for c in time_coord.cells()])
        coord_multidim = day_of_year[np.newaxis, np.newaxis, :]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape).flatten()
        if feature == "day_of_year_sin":
            feature_values = np.sin(2 * np.pi * feature_values / HOURS_IN_DAY)
        elif feature == "day_of_year_cos":
            feature_values = np.cos(2 * np.pi * feature_values / HOURS_IN_DAY)
    elif feature in ["hour_of_day", "hour_of_day_sin", "hour_of_day_cos"]:
        hour_of_day = np.zeros(cube.coord("time").shape)
        for i in range(cube.coord("time").shape[0]):
            for j in range(cube.coord("time").shape[1]):
                hour_of_day[i, j] = cube.coord("time")[i][j].cell(0).point.hour
        hour_of_day = np.array(hour_of_day)[:, np.newaxis, :]
        feature_values = np.broadcast_to(hour_of_day, collapsed_cube.shape).flatten()
        if feature == "day_of_year_sin":
            feature_values = np.sin(2 * np.pi * feature_values / (DAYS_IN_YEAR + 1))
        elif feature == "day_of_year_cos":
            feature_values = np.cos(2 * np.pi * feature_values / (DAYS_IN_YEAR + 1))
    elif feature in additional_feature_cube_names:
        feature_cube = additional_feature_cubes.extract(iris.Constraint(name=feature))
        coord_multidim = feature_cube.data[np.newaxis, :, np.newaxis]
        feature_values = np.broadcast_to(coord_multidim, collapsed_cube.shape)
    return feature_values


class TrainQuantileRegressionRandomForests:
    """Plugin to perform quantile regression using random forests."""

    def __init__(
        self,
        features,
        n_estimators=100,
        max_depth=None,
        random_state=None,
        transformation=None,
        pre_transform_addition=0,
        compression=5,
        **kwargs,
    ):
        """Initialise the plugin.

        Args:
            features (list):
                List of features to be used in the quantile regression.
            n_estimators (int):
                Number of trees in the forest.
            max_depth (int):
                Maximum depth of the tree.
            random_state (int):
                Random seed for reproducibility.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
            compression (int):
                Compression level for saving the model.
            kwargs:
                Additional keyword arguments for the quantile regression model.
        """
        self.features = features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.transformation = transformation
        if self.transformation not in ["log", "log10", "sqrt", "cbrt"]:
            msg = (
                "Currently the only supported transformations are log, log10, sqrt "
                f"and cbrt. The transformation supplied was {self.transformation}."
            )
            raise ValueError(msg)
        self.pre_transform_addition = pre_transform_addition
        self.compression = compression
        self.kwargs = kwargs

    def fit_qrf(
        self, forecast_features: np.ndarray, target: np.ndarray
    ) -> RandomForestQuantileRegressor:
        """Fit the quantile regression model.
        Args:
            forecast_features (numpy.ndarray):
                Array of forecast features.
            target (numpy.ndarray):
                Array of target values.
        Returns:
            qrf_model (RandomForestQuantileRegressor):
                Fitted quantile regression model.
        """
        qrf_model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state**self.kwargs,
        )
        qrf_model.fit(forecast_features, target)
        return qrf_model

    def process(
        self,
        forecast_cube: Cube,
        truth_cube: Cube,
        additional_feature_cubes: Optional[CubeList] = None,
    ):
        """Train a quantile regression random forests model.

        Args:
            forecast_cube:
                Cube containing the forecasts.
            truth_cube:
                Cube containing the truths.
            additional_feature_cubes:
                List of additional feature cubes.
        """
        if self.transformation:
            forecast_cube.data = getattr(np, self.transformation)(
                forecast_cube.data + self.pre_transform_addition
            )
            truth_cube.data = getattr(np, self.transformation)(
                truth_cube.data + self.pre_transform_addition
            )

        feature_values = []
        for feature in self.features:
            feature_values.append(
                prep_feature(forecast_cube, feature, additional_feature_cubes)
            )

        feature_values = np.array(feature_values).T
        target_values = truth_cube.data.flatten()
        # Fit the quantile regression model
        qrf_model = self.fit_qrf(feature_values, target_values)

        # Need to do something to avoid the output from this plugin being saved
        # as a netCDF file. This is a pickle-based format.
        output_path = None
        joblib.dump(qrf_model, output_path, compress=self.compression)


class ApplyQuantileRegressionRandomForests:
    """"""

    def __init__(
        self, features, quantiles, transformation=None, pre_transform_addition=0
    ):
        """Initialise the plugin."""
        self.features = features
        self.quantiles = quantiles
        self.transformation = transformation
        if self.transformation not in ["log", "log10", "sqrt", "cbrt"]:
            msg = (
                "Currently the only supported transformations are log, log10, sqrt "
                f"and cbrt. The transformation supplied was {self.transformation}."
            )
            raise ValueError(msg)
        self.pre_transform_addition = pre_transform_addition

    def process(
        self, forecast_cube, additional_feature_cubes: Optional[CubeList] = None
    ):
        if self.transformation:
            if self.transformation == "log":
                forecast_cube.data = (
                    np.exp(forecast_cube.data) - self.pre_transform_addition
                )
            elif self.transformation == "log10":
                forecast_cube.data = (
                    10 ** (forecast_cube.data) - self.pre_transform_addition
                )
            elif self.transformation == "sqrt":
                forecast_cube.data = forecast_cube.data**2 - self.pre_transform_addition
            elif self.transformation == "cbrt":
                forecast_cube.data = forecast_cube.data**3 - self.pre_transform_addition
            else:
                raise ValueError(
                    f"Transformation {self.transformation} not available from numpy. "
                    "Currently only transformations directly available from numpy are "
                    "supported e.g. log, log10, sqrt, cbrt."
                )

        feature_values = []
        for feature in self.features:
            feature_values.append(
                prep_feature(forecast_cube, feature, additional_feature_cubes)
            )
        feature_values = np.array(feature_values).T

        # We don't usually load stuff within plugins, so might need to think about this.
        model_path = None
        qrf_model = joblib.load(model_path)
        calibrated_forecast = qrf_model.predict(
            feature_values, quantiles=self.quantiles
        )

        calibrated_forecast_cube = calibrated_forecast.copy(data=calibrated_forecast.T)
        return calibrated_forecast_cube
