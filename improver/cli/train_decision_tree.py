#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to train a decision tree-based model for post-processing."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *file_paths: cli.inputpath,
    feature_config: cli.inputjson,
    parquet_diagnostic_names: cli.comma_separated_list,
    cf_names: cli.comma_separated_list,
    forecast_periods: str,
    cycletime: str,
    training_length: int,
    experiments: cli.comma_separated_list,
    method: str = "qrf",
    model_kwargs: cli.inputjson = None,
    transformation: str = None,
    pre_transform_addition: float = 0,
    unique_site_id_keys: cli.comma_separated_list = "wmo_id",
):
    """Training a decision tree model for post-processing.

    Loads in arguments for training a decision tree-based model which can later
    be applied to calibrate the forecast. Two sources of input data must be
    provided: historical forecasts and historical truth data (to use in
    calibration). The model is output as a pickle file.

    Args:
        file_paths (cli.inputpaths):
            A list of input paths (in any order) containing:
            - The path to a Parquet file containing the truths to be used
            for calibration. The expected columns within the
            Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
            longitude and altitude.
            - The path to a Parquet file containing the forecasts to be used
            for calibration. The expected columns within the Parquet file are:
            forecast, blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period, height,
            cf_name, units. Please note that the presence of a forecast_period
            column is used to separate the forecast parquet file from the truth
            parquet file.
            - Optionally, paths to NetCDF files containing additional predictors.
        feature_config (dict):
            Feature configuration defining the features to be used for the
            decision tree model. The configuration is a dictionary of strings,
            where the keys are the names of the input cube(s) supplied, and the
            values are a list. This list can contain both computed features, such
            as the mean or standard deviation (std), or static features, such as
            the altitude. The computed features will be computed using the cube
            defined in the dictionary key. If the key is the feature itself e.g.
            a distance to water cube, then the value should state "static". This
            will ensure the cube's data is used as the feature.
            The config will have the structure:
            "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g:
            {
            "air_temperature": ["mean", "std", "altitude"],
            "visibility_at_screen_level": ["mean", "std"]
            "distance_to_water": ["static"],
            }
        parquet_diagnostic_names (list of str):
            A list containing the diagnostic names that will be used for filtering
            the forecast and truth DataFrames read in from the parquet files. The
            target diagnostic name is expected to be the first item in the list.
            These names could be different from the CF name e.g.
            'temperature_at_screen_level'. This is expected to be the same length
            as the cf_names and experiments lists.
        cf_names (list of str):
            A list containing the CF names of the diagnostics. The CF names should
            match the order of the parquet_diagnostic_names. The target diagnostic to be
            calibrated is expected to be the first item in the list. These names
            could be different from the diagnostic name used to identify in the
            parquet files. For example, the diagnostic name could be
            'temperature_at_screen_level' and the corresponding CF name could be
            'air_temperature'. This is expected to be the same length as the
            parquet_diagnostic_names and experiments lists.
        forecast_periods (str):
            Range of forecast periods to be calibrated in hours in the form:
            "start:end:interval" e.g. "6:18:6" or a single forecast period e.g. "6".
            The end value is exclusive, so "6:18:6" will calibrate the 6 and 12 hours.
        cycletime (str):
            Cycletime of a format similar to 20170109T0000Z used to filter the
            correct blendtimes from the dataframe on load.
        training_length (int):
            The length of the training period in days.
        experiments (list of str):
            The names of the experiment (step) that calibration is
            applied to. This is used to filter the forecast DataFrame on load.
            This is expected to be the same length as the parquet_diagnostic_names
            and cf_names lists.
        method (str):
            The decision tree method to use for training. Supported values are
            ``"qrf"`` (default), ``"lightgbm"``, and ``"xgboost"``.
        model_kwargs (dict):
            A JSON dictionary of keyword arguments to pass to the model
            constructor selected by ``method``. For ``"qrf"`` these correspond
            to ``RandomForestQuantileRegressor`` parameters (e.g.
            ``{"n_estimators": 100, "max_depth": 5, "random_state": 0}``).
            For ``"lightgbm"`` these correspond to ``lgb.LGBMRegressor``
            parameters. For ``"xgboost"`` these correspond to
            ``xgb.XGBRegressor`` parameters.
        transformation (str):
            Transformation to be applied to the data before fitting.
        pre_transform_addition (float):
            Value to be added before transformation.
        unique_site_id_keys (str):
            The names of the coordinates that uniquely identify each site,
            e.g. "wmo_id" or "latitude,longitude".
    Returns:
        A trained decision tree model with associated transformation and
        pre-transformation addition that will be stored as a pickle file.
    """

    from improver.calibration.load_and_train_decision_tree import (
        LoadForTrainDecisionTree,
        PrepareAndTrainDecisionTree,
    )

    forecast_df, truth_df, cube_inputs = LoadForTrainDecisionTree(
        experiments=experiments,
        feature_config=feature_config,
        parquet_diagnostic_names=parquet_diagnostic_names,
        cf_names=cf_names,
        forecast_periods=forecast_periods,
        cycletime=cycletime,
        training_length=training_length,
        unique_site_id_keys=unique_site_id_keys,
    )(file_paths)
    if forecast_df is None or truth_df is None or cube_inputs is None:
        return None

    result = PrepareAndTrainDecisionTree(
        feature_config=feature_config,
        target_cf_name=cf_names[0],
        method=method,
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
        unique_site_id_keys=unique_site_id_keys,
        **(model_kwargs or {}),
    )(forecast_df, truth_df, cube_inputs)
    if result == (None, None, None):
        return None

    return result
