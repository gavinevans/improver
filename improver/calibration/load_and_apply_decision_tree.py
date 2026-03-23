#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load and apply a trained decision tree model."""

import warnings
from typing import Any, Optional

import iris
import numpy as np
import pandas as pd
from iris.cube import Cube, CubeList
from iris.pandas import as_data_frame

from improver import PostProcessingPlugin
from improver.calibration import add_static_feature_from_cube_to_df, add_warning_comment
from improver.calibration.decision_tree import (
    ApplyDecisionTreeModel,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.utilities.cube_checker import assert_spatial_coords_match
from improver.utilities.temporal import datetime_to_iris_time

iris.FUTURE.pandas_ndim = True


class PrepareAndApplyDecisionTree(PostProcessingPlugin):
    """Prepare the input forecast for application of a trained decision tree model
    and apply the model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_cf_name: str,
        method: str = "qrf",
        unique_site_id_keys: list[str] = ["wmo_id"],
        cycletime: Optional[str] = None,
        forecast_period: Optional[int] = None,
    ):
        """Initialise the plugin.

        Args:
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the input cube(s) supplied, and the values are
                a list. This list can contain both computed features, such as the mean
                or standard deviation (std), or static features, such as the altitude.
                The computed features will be computed using the cube defined in the
                dictionary key. If the key is the feature itself e.g. a distance to
                water cube, then the value should state "static". This will ensure
                the cube's data is used as the feature.
                The config will have the structure:
                "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g:
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            target_cf_name (str):
                A string containing the CF name of diagnostic to be calibrated. This
                will be used to separate it from the rest of the dynamic predictors,
                if present.
            method (str):
                Name of the decision tree method that was used to train the model.
                Supported values are ``"qrf"``, ``"lightgbm"``, and ``"xgboost"``.
                Defaults to ``"qrf"``.
            unique_site_id_keys (list):
                The names of the coordinates that uniquely identify each site,
                e.g. "wmo_id" or ["latitude", "longitude"].
            cycletime (str):
                The cycle time of the forecast to be calibrated in the format
                YYYYMMDDTHHMMZ. If not provided, the cycle time found in the first
                forecast cube will be used.
            forecast_period (int):
                The forecast period of the forecast to be calibrated in seconds. If not
                provided, the forecast period found in the first forecast cube
                will be used.
        """
        self.feature_config = feature_config
        self.target_cf_name = target_cf_name
        self.method = method
        self.unique_site_id_keys = unique_site_id_keys
        self.cycletime = cycletime
        self.forecast_period = forecast_period

    def _get_inputs(
        self,
        cube_inputs: iris.cube.CubeList,
        model: Optional[Any] = None,
    ) -> tuple[CubeList, Cube]:
        """Split the forecast to be calibrated from the other features. Handle
        the case where the model is not provided, for example, if the input
        data required to train the model isn't yet available. In this case,
        the uncalibrated forecast is returned with a warning comment added.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.
            model: The trained model to be applied to the forecast. If None,
                the input forecast will be returned unchanged with a warning comment
                added.

        Returns:
            CubeList of the features cubes and the forecast cube.

        Raises:
            ValueError: If the target forecast is not provided.
            ValueError: If the number of cubes provided does not match the number of
                features expected.
            ValueError: If the input cubes contain a mix of realization and percentile
                coordinates.
        """
        # Extract all additional cubes which are associated with a feature in the
        # feature_config.
        forecast_constraint = iris.Constraint(name=self.target_cf_name)
        forecast_cube = cube_inputs.extract(forecast_constraint)

        if forecast_cube:
            (forecast_cube,) = forecast_cube
        else:
            msg = (
                "No target forecast provided. An input cube representing the target "
                "must be provided, even if the target will not be used as a feature. "
                f"The target is '{self.target_cf_name}'."
            )
            raise ValueError(msg)

        if not model:
            # If no model is provided, return the input forecast with a warning.
            forecast_cube = add_warning_comment(forecast_cube)
            return None, forecast_cube

        if len(cube_inputs) < len(self.feature_config.keys()):
            msg = (
                "The number of cubes loaded is fewer than the number of features "
                "expected. The number of cubes loaded was: "
                f"{len(cube_inputs)}. The number of features expected was: "
                f"{len(self.feature_config.keys())}."
            )
            raise ValueError(msg)

        # If target diagnostic not a feature in the training then remove.
        if self.target_cf_name not in self.feature_config.keys():
            cube_inputs.remove(forecast_cube)

        representations = []
        for cube in cube_inputs:
            for coord in ["percentile", "realization"]:
                if cube.coords(coord):
                    representations.append(coord)
                    break
        if len(set(representations)) > 1:
            msg = (
                "The input cubes contain a mix of realization and percentile "
                "coordinates. All input cubes must use the same representation."
            )
            raise ValueError(msg)

        return cube_inputs, forecast_cube

    @staticmethod
    def _compute_quantile_list(forecast_cube: Cube, coord: str) -> list[float]:
        """Compute the list of quantiles e.g. 0.25, 0.5, 0.75 that will be produced
        from a specified coordinate on the forecast cube.

        Args:
            forecast_cube: Forecast to be calibrated.
            coord: Coordinate name. The length of the coordinate will be used to
                determine the number of quantiles to compute.

        Returns:
            List of quantiles (e.g. 0.25, 0.5, 0.75) computed from the forecast cube.
        """
        n_percentiles = len(forecast_cube.coord(coord).points)
        quantiles = (np.array(choose_set_of_percentiles(n_percentiles)) / 100).tolist()
        return quantiles

    def _update_forecast_reference_time_and_period(
        self, cube_inputs: CubeList
    ) -> CubeList:
        """Update the forecast_reference_time and forecast_period coordinates
        on the input cubes to match those provided, if they are provided. The rebadging
        of the forecast_period introduces a slight discrepancy between the forecasts
        used for training and application of the decision tree model. However, as any forecast
        period rebadging is expected to be small (e.g. a few hours), this is not
        expected to be a significant issue.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.
        Returns:
            CubeList of the input cubes with updated forecast_reference_time and
            forecast_period coordinates, if they were provided.
        """
        if self.cycletime:
            cycletime = datetime_to_iris_time(
                pd.to_datetime(self.cycletime, format="%Y%m%dT%H%MZ")
            )
        else:
            (cycletime,) = cube_inputs[0].coord("forecast_reference_time").points

        if self.forecast_period:
            forecast_period = self.forecast_period
        else:
            (forecast_period,) = cube_inputs[0].coord("forecast_period").points

        # Update the forecast_reference_time and forecast_period to match those
        # provided, if they are provided.
        for cube in cube_inputs:
            for coord_name in ["blend_time", "forecast_reference_time"]:
                if coord_name in [coord.name() for coord in cube.coords()]:
                    cube.coord(coord_name).points = cycletime
            if "forecast_period" in [coord.name() for coord in cube.coords()]:
                cube.coord("forecast_period").points = forecast_period
                if cube.coord("forecast_period").bounds is not None:
                    diff = (
                        cube.coord("forecast_period").bounds[0][1]
                        - cube.coord("forecast_period").bounds[0][0]
                    )
                    cube.coord("forecast_period").bounds = np.array(
                        [[forecast_period - diff, forecast_period]]
                    )
        return cube_inputs

    def _cube_to_dataframe(self, cube_inputs: CubeList) -> pd.DataFrame:
        """Convert cube inputs to a pandas DataFrame.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.
        Returns:
            DataFrame containing the data from the cubes, with auxiliary coordinates
            included as columns.
        """
        # Convert the first cube to a DataFrame.
        df = as_data_frame(cube_inputs[0], add_aux_coords=True).reset_index()

        possible_columns = [
            *self.unique_site_id_keys,
            "time",
            "forecast_reference_time",
            "forecast_period",
            "percentile",
            "realization",
        ]

        # Iteratively convert remaining cubes to DataFrame and merge.
        for cube in cube_inputs[1:]:
            df = add_static_feature_from_cube_to_df(
                df,
                cube,
                cube.name(),
                possible_columns,
            )

        for column in ["forecast_reference_time", "time"]:
            df[column] = df[column].apply(lambda x: x._to_real_datetime())
        return df

    def process(
        self,
        cube_inputs: CubeList,
        model_descriptors: Optional[tuple[Any, str, float]] = None,
    ) -> Cube:
        """Load and apply the trained decision tree model. The model is used to
        calibrate the forecast provided. The calibrated forecast is written to a cube.
        If no model is provided the input forecast is returned unchanged.

        Args:
            cube_inputs: List of cubes containing the features and the forecast to be
                calibrated.
            model_descriptors: The trained model to be applied to the forecast
                and the transformation and pre-transform addition applied during
                training. The descriptors expected are a tuple of:
                (model, transformation, pre_transform_addition).

        Returns:
            iris.cube.Cube:
                The calibrated forecast cube.
        """
        if model_descriptors is None:
            # If no descriptors are provided, return the input forecast with a warning.
            # Descriptors expected: (model, transformation, pre_transform_addition)
            model_descriptors = (None, None, 0)
        model, transformation, pre_transform_addition = model_descriptors

        cube_inputs, forecast_cube = self._get_inputs(cube_inputs, model=model)

        if cube_inputs:
            assert_spatial_coords_match(cube_inputs)

        if not model:
            msg = (
                "Unable to apply the decision tree model. "
                "No trained model has been provided. "
                "Returning the input forecast without calibration."
            )
            warnings.warn(msg)
            return forecast_cube

        template_forecast_cube = forecast_cube.copy()
        if forecast_cube.coords("realization"):
            quantile_list = self._compute_quantile_list(
                forecast_cube.copy(), "realization"
            )
        elif forecast_cube.coords("percentile"):
            quantile_list = (forecast_cube.coord("percentile").points / 100.0).tolist()

        cube_inputs = self._update_forecast_reference_time_and_period(cube_inputs)

        df = self._cube_to_dataframe(cube_inputs)

        calibrated_forecast = ApplyDecisionTreeModel(
            target_name=self.target_cf_name,
            feature_config=self.feature_config,
            quantiles=quantile_list,
            method=self.method,
            transformation=transformation,
            pre_transform_addition=pre_transform_addition,
            unique_site_id_keys=self.unique_site_id_keys,
        )(model, df)
        calibrated_forecast_cube = template_forecast_cube.copy(
            data=np.broadcast_to(calibrated_forecast.T, template_forecast_cube.shape)
        )

        return calibrated_forecast_cube
