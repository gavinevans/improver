# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""init for calibration that contains functionality to split forecast and truth
inputs, and functionality to convert a pandas DataFrame in the expected format
into an iris cube.

.. Further information is available in:
.. include:: extended_documentation/calibration/calibration_data_ingestion.rst

"""

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.utilities.cube_manipulation import MergeCubes

FORECAST_DATAFRAME_COLUMNS = [
    "altitude",
    "blend_time",
    "cf_name",
    "diagnostic",
    "forecast",
    "forecast_period",
    "forecast_reference_time",
    "height",
    "latitude",
    "longitude",
    "percentile",
    "period",
    "time",
    "units",
    "wmo_id",
]

TRUTH_DATAFRAME_COLUMNS = [
    "altitude",
    "diagnostic",
    "latitude",
    "longitude",
    "ob_value",
    "time",
    "wmo_id",
]


def split_forecasts_and_truth(
    cubes: List[Cube], truth_attribute: str
) -> Tuple[Cube, Cube, Optional[Cube]]:
    """
    A common utility for splitting the various inputs cubes required for
    calibration CLIs. These are generally the forecast cubes, historic truths,
    and in some instances a land-sea mask is also required.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            These include the historical forecasts, in the format supported by
            the calibration CLIs, and the truth cubes.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        - A cube containing all the historic forecasts.
        - A cube containing all the truth data.
        - If found within the input cubes list a land-sea mask will be
          returned, else None is returned.

    Raises:
        ValueError:
            An unexpected number of distinct cube names were passed in.
        IOError:
            More than one cube was identified as a land-sea mask.
        IOError:
            Missing truth or historical forecast in input cubes.
    """
    grouped_cubes = {}
    for cube in cubes:
        try:
            cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
        except ValueError:
            cube_name = cube.name()
        grouped_cubes.setdefault(cube_name, []).append(cube)
    if len(grouped_cubes) == 1:
        # Only one group - all forecast/truth cubes
        land_sea_mask = None
        diag_name = list(grouped_cubes.keys())[0]
    elif len(grouped_cubes) == 2:
        # Two groups - the one with exactly one cube matching a name should
        # be the land_sea_mask, since we require more than 2 cubes in
        # the forecast/truth group
        grouped_cubes = OrderedDict(
            sorted(grouped_cubes.items(), key=lambda kv: len(kv[1]))
        )
        # landsea name should be the key with the lowest number of cubes (1)
        landsea_name, diag_name = list(grouped_cubes.keys())
        land_sea_mask = grouped_cubes[landsea_name][0]
        if len(grouped_cubes[landsea_name]) != 1:
            raise IOError("Expected one cube for land-sea mask.")
    else:
        raise ValueError("Must have cubes with 1 or 2 distinct names.")

    # split non-land_sea_mask cubes on forecast vs truth
    truth_key, truth_value = truth_attribute.split("=")
    input_cubes = grouped_cubes[diag_name]
    grouped_cubes = {"truth": [], "historical forecast": []}
    for cube in input_cubes:
        if cube.attributes.get(truth_key) == truth_value:
            grouped_cubes["truth"].append(cube)
        else:
            grouped_cubes["historical forecast"].append(cube)

    missing_inputs = " and ".join(k for k, v in grouped_cubes.items() if not v)
    if missing_inputs:
        raise IOError(f"Missing {missing_inputs} input.")

    truth = MergeCubes()(grouped_cubes["truth"])
    forecast = MergeCubes()(grouped_cubes["historical forecast"])

    return forecast, truth, land_sea_mask


def _dataframe_column_check(df: DataFrame, compulsory_columns: Sequence) -> None:
    """Check that the compulsory columns are present on the DataFrame.
    Any other columns within the DataFrame are ignored.

    Args:
        df:
            Dataframe expected to contain the compulsory columns.
        compulsory_columns:
            The names of the compulsory columns.

    Raises:
        ValueError: Raise an error if a compulsory column is missing.
    """
    if not set(compulsory_columns).issubset(df.columns):
        diff = set(compulsory_columns).difference(df.columns)
        msg = (
            "The following compulsory column(s) are missing from the "
            f"dataframe: {diff}"
        )
        raise ValueError(msg)


def _preprocess_temporal_columns(df: DataFrame) -> DataFrame:
    """Pre-process the columns with temporal dtype to convert
    from numpy datetime objects to pandas datetime objects.
    Casting the dtype of the columns to object type results
    in columns of dtype "object" with the contents of the
    columns being pandas datetime objects, rather than numpy
    datetime objects.

    Args:
        df:
            A DataFrame with temporal columns with numpy
            datetime dtypes.

    Returns:
        A DataFrame without numpy datetime dtypes. The
        content of the columns with temporal dtypes are
        accessible as pandas datetime objects.
    """
    for col in df.select_dtypes(include="datetime64[ns]"):
        df[col] = df[col].dt.tz_localize("UTC").astype("O")
    for col in df.select_dtypes(include="timedelta64[ns]"):
        df[col] = df[col].astype("O")
    return df


def _unique_check(df: DataFrame, column: str) -> None:
    """Check whether the values in the column are unique.

    Args:
        df:
            The DataFrame to be checked.
        column:
            Name of a column in the DataFrame.

    Raises:
        ValueError: Only one unique value within the specifed column
            is expected.
    """
    if df[column].nunique(dropna=False) > 1:
        msg = (
            f"Multiple values provided for the {column}: "
            f"{df[column].unique()}. "
            f"Only one value for the {column} is expected."
        )
        raise ValueError(msg)


def _define_time_coord(
    adates: List[pd.Timestamp],
    all_time_bounds: Optional[Sequence[pd.Timestamp]] = None,
) -> DimCoord:
    """Define a time coordinate. The coordinate will have bounds,
    if bounds are provided.

    Args:
        adate:
            The point for the time coordinate.
        time_bounds:
            The values defining the bounds for the time coordinate.

    Returns:
        A time coordinate. This coordinate will have bounds, if bounds
        are provided.
    """
    if all_time_bounds:
        timestamp_time_bounds = []
        for a_time_bounds_pair in all_time_bounds:
            timestamp_time_bounds.append(
                [
                    np.array(t.timestamp(), dtype=TIME_COORDS["time"].dtype)
                    for t in a_time_bounds_pair
                ]
            )

    return DimCoord(
        np.array(
            [adate.timestamp() for adate in adates], dtype=TIME_COORDS["time"].dtype
        ),
        "time",
        bounds=all_time_bounds if all_time_bounds is None else timestamp_time_bounds,
        units=TIME_COORDS["time"].units,
    )


def _define_height_coord(height) -> AuxCoord:
    """Define a height coordinate. A unit of metres is assumed.

    Args:
        height:
            The value for the height coordinate in metres.

    Returns:
        The height coordinate.
    """
    return AuxCoord(np.array(height, dtype=np.float32), "height", units="m",)


def _training_dates_for_calibration(
    cycletime: str, forecast_period: int, training_length: int
) -> DatetimeIndex:
    """Compute the date range required for extracting the required training
    dataset. The final validity time within the training dataset is
    at least one day prior to the cycletime. The final validity time
    within the training dataset is additionally offset by the number
    of days within the forecast period to ensure that the dates defined
    by the training dataset are in the past relative to the cycletime.
    For example, for a cycletime of 20170720T0000Z with a forecast period
    of T+30 and a training length of 3 days, the validity time is
    20170721T0600Z. Subtracting one day gives 20170720T0600Z. Note that
    this is in the future relative to the cycletime and we want the
    training dates to be in the past relative to the cycletime.
    Subtracting the forecast period rounded down to the nearest day for
    T+30 gives 1 day. Subtracting this additional day gives 20170719T0600Z.
    This is the final validity time within the training period. We then
    compute the validity times for a 3 day training period using 20170719T0600Z
    as the final validity time giving 20170719T0600Z, 20170718T0600Z
    and 20170717T0600Z.

    Args:
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
            The training dates will always be in the past, relative
            to the cycletime.
        forecast_period:
            Forecast period in hours as an integer.
        training_length:
            Training length in days as an integer.

    Returns:
        Datetimes defining the training dataset. The number of datetimes
        is equal to the training length.
    """
    forecast_period = pd.Timedelta(int(forecast_period), unit="hours")
    validity_time = pd.Timestamp(cycletime) + forecast_period
    return pd.date_range(
        end=validity_time - pd.Timedelta(1, unit="days") - forecast_period.floor("D"),
        periods=int(training_length),
        freq="D",
        tz="UTC",
    )


def _prepare_dataframes(
    forecast_df: DataFrame, truth_df: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """Prepare dataframes for conversion to cubes by: 1) checking
    that the expected columns are present, 2) finding the sites
    common to both the forecast and truth dataframes and 3)
    replacing and supplementing the truth dataframe with
    information from the forecast dataframe. Note that this third
    step will also ensure that a row containing a NaN for the
    ob_value is inserted for any missing observations.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units. Any other columns are ignored.
        truth_df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
            Any other columns are ignored.

    Returns:
        A sanitised version of the forecasts and truth dataframes that
        are ready for conversion to cubes.
    """
    _dataframe_column_check(forecast_df, FORECAST_DATAFRAME_COLUMNS)
    _dataframe_column_check(truth_df, TRUTH_DATAFRAME_COLUMNS)

    # Find the common set of WMO IDs.
    common_wmo_ids = set(forecast_df["wmo_id"]).intersection(truth_df["wmo_id"])
    forecast_df = forecast_df[forecast_df["wmo_id"].isin(common_wmo_ids)]
    truth_df = truth_df[truth_df["wmo_id"].isin(common_wmo_ids)]

    truth_df = truth_df.drop(columns=["altitude", "latitude", "longitude"])
    # Identify columns to copy onto the truth_df from the forecast_df
    forecast_subset = forecast_df[
        [
            "wmo_id",
            "latitude",
            "longitude",
            "altitude",
            "period",
            "height",
            "cf_name",
            "units",
            "time",
            "diagnostic",
        ]
    ].drop_duplicates()
    # Use "outer" to fill in any missing observations in the truth dataframe.
    truth_df = truth_df.merge(
        forecast_subset, on=["wmo_id", "time", "diagnostic"], how="outer"
    )
    return forecast_df, truth_df


def reshape_forecast_column(
    df: DataFrame, column: str, training_dates: DatetimeIndex
) -> np.ndarray:
    """Reshape a forecast column into the shape required for the output cube.

    Args:
        df: The forecast DataFrame.
        column: The name of a column in the forecast DataFrame.
        training_dates: The validity times within the training period.

    Returns:
        A column from the forecast DataFrame that has been reshaped into an
        appropriately shaped numpy array.

    """
    return np.transpose(
        np.reshape(
            df[column].values,
            (len(training_dates), df["percentile"].nunique(), df["wmo_id"].nunique(),),
        ),
        axes=(1, 0, 2),
    )


def reshape_truth_column(
    df: DataFrame, column: str, training_dates: DatetimeIndex
) -> np.ndarray:
    """Reshape a forecast column into the shape required for the output cube.

    Args:
        df: The truth DataFrame
        column: The name of a column in the forecast DataFrame.
        training_dates: The validity times within the training period.

    Returns:
        A column from the truth DataFrame that has been reshaped into an
        appropriately shaped numpy array.
    """
    return np.reshape(
        df[column].values, (len(training_dates), df["wmo_id"].nunique(),),
    )


def forecast_dataframe_to_cube(
    df: DataFrame, training_dates: DatetimeIndex, forecast_period: int
) -> Cube:
    """Convert a forecast DataFrame into an iris Cube. The percentiles
    within the forecast DataFrame are rebadged as realizations.

    Args:
        df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units. Any other columns are ignored.
        training_dates:
            Datetimes spanning the training period.
        forecast_period:
            Forecast period in hours as an integer.

    Returns:
        Cube containing the forecasts from the training period.
    """
    df = _preprocess_temporal_columns(df)

    fp_point = pd.Timedelta(int(forecast_period), unit="hours")

    # The following columns are expected to contain one unique value
    # per column.
    for col in ["period", "height", "cf_name", "units", "diagnostic"]:
        _unique_check(df, col)

    time_df = df.loc[
        (df["time"].isin(training_dates)) & (df["forecast_period"] == fp_point)
    ]

    if time_df.empty:
        return

    if time_df["period"].isna().all():
        fp_bounds = None
    else:
        period = time_df["period"].values[0]
        fp_bounds = [fp_point - period, fp_point]

    if time_df["period"].isna().all():
        time_bounds = None
    else:
        period = time_df["period"].values[0]
        time_bounds = []
        for adate in training_dates:
            time_bounds.append([adate - period, adate])

    time_coord = _define_time_coord(training_dates, time_bounds)
    height_coord = _define_height_coord(time_df["height"].values[0])

    fp_coord = AuxCoord(
        np.array(fp_point.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype),
        "forecast_period",
        bounds=fp_bounds
        if fp_bounds is None
        else [
            np.array(f.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype)
            for f in fp_bounds
        ],
        units=TIME_COORDS["forecast_period"].units,
    )
    frt_coord = AuxCoord(
        np.array(
            [t.timestamp() for t in time_df["forecast_reference_time"].unique()],
            dtype=TIME_COORDS["forecast_reference_time"].dtype,
        ),
        "forecast_reference_time",
        units=TIME_COORDS["forecast_reference_time"].units,
    )

    perc_coord = DimCoord(
        np.array(sorted(df["percentile"].unique()), dtype=np.float32),
        long_name="percentile",
        units="%",
    )

    cube = build_spotdata_cube(
        reshape_forecast_column(time_df, "forecast", training_dates).astype(np.float32),
        time_df["cf_name"].values[0],
        time_df["units"].values[0],
        reshape_forecast_column(time_df, "altitude", training_dates).astype(np.float32)[
            0, 0, :
        ],
        reshape_forecast_column(time_df, "latitude", training_dates).astype(np.float32)[
            0, 0, :
        ],
        reshape_forecast_column(time_df, "longitude", training_dates).astype(
            np.float32
        )[0, 0, :],
        reshape_forecast_column(time_df, "wmo_id", training_dates).astype("U5")[
            0, 0, :
        ],
        additional_dims=[perc_coord, time_coord],
        scalar_coords=[fp_coord, height_coord],
    )
    cube.add_aux_coord(frt_coord, data_dims=1)

    return RebadgePercentilesAsRealizations()(cube)


def truth_dataframe_to_cube(df: DataFrame, training_dates: DatetimeIndex,) -> Cube:
    """Convert a truth DataFrame into an iris Cube.

    Args:
        df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
            Any other columns are ignored.
        training_dates:
            Datetimes spanning the training period.

    Returns:
        Cube containing the truths from the training period.
    """
    df = _preprocess_temporal_columns(df)

    time_df = df.loc[df["time"].isin(training_dates)]

    if time_df.empty:
        return

    # The following columns are expected to contain one unique value
    # per column.
    _unique_check(time_df, "diagnostic")

    if time_df["period"].isna().all():
        time_bounds = None
    else:
        period = time_df["period"].values[0]
        time_bounds = []
        for adate in training_dates:
            time_bounds.append([adate - period, adate])

    time_coord = _define_time_coord(training_dates, time_bounds)
    height_coord = _define_height_coord(time_df["height"].values[0])

    cube = build_spotdata_cube(
        reshape_truth_column(time_df, "ob_value", training_dates).astype(np.float32),
        time_df["cf_name"].values[0],
        time_df["units"].values[0],
        reshape_truth_column(time_df, "altitude", training_dates).astype(np.float32)[
            0, :
        ],
        reshape_truth_column(time_df, "latitude", training_dates).astype(np.float32)[
            0, :
        ],
        reshape_truth_column(time_df, "longitude", training_dates).astype(np.float32)[
            0, :
        ],
        reshape_truth_column(time_df, "wmo_id", training_dates).astype("U5")[0, :],
        additional_dims=[time_coord],
        scalar_coords=[height_coord],
    )
    return cube


def forecast_and_truth_dataframes_to_cubes(
    forecast_df: DataFrame,
    truth_df: DataFrame,
    cycletime: str,
    forecast_period: int,
    training_length: int,
) -> Tuple[Cube, Cube]:
    """Convert a forecast DataFrame into an iris Cube and a
    truth DataFrame into an iris Cube.

    Args:
        forecast_df:
            DataFrame expected to contain the following columns: forecast,
            blend_time, forecast_period, forecast_reference_time, time,
            wmo_id, percentile, diagnostic, latitude, longitude, period,
            height, cf_name, units. Any other columns are ignored.
        truth_df:
            DataFrame expected to contain the following columns: ob_value,
            time, wmo_id, diagnostic, latitude, longitude and altitude.
            Any other columns are ignored.
        cycletime:
            Cycletime of a format similar to 20170109T0000Z.
        forecast_period:
            Forecast period in hours as an integer.
        training_length:
            Training length in days as an integer.

    Returns:
        Forecasts and truths for the training period in Cube format.
    """
    training_dates = _training_dates_for_calibration(
        cycletime, forecast_period, training_length
    )

    forecast_df, truth_df = _prepare_dataframes(forecast_df, truth_df)

    forecast_cube = forecast_dataframe_to_cube(
        forecast_df, training_dates, forecast_period
    )
    truth_cube = truth_dataframe_to_cube(truth_df, training_dates)
    return forecast_cube, truth_cube
