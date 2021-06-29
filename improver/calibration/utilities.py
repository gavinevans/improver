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
"""
This module defines all the utilities used by the "plugins"
specific for ensemble calibration.

"""
import importlib
from typing import Optional, Set, Tuple, Union

import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver.utilities.temporal import iris_time_to_datetime


def convert_cube_data_to_2d(
    forecast: Cube, coord: str = "realization", transpose: bool = True
) -> ndarray:
    """
    Function to convert data from a N-dimensional cube into a 2d
    numpy array. The result can be transposed, if required.

    Args:
        forecast:
            N-dimensional cube to be reshaped.
        coord:
            This dimension is retained as the second dimension by default,
            and the leading dimension if "transpose" is set to False.
        transpose:
            If True, the resulting flattened data is transposed.
            This will transpose a 2d array of the format [coord, :]
            to [:, coord].  If coord is not a dimension on the input cube,
            the resulting array will be 2d with items of length 1.

    Returns:
        Reshaped 2d array.
    """
    forecast_data = []
    if np.ma.is_masked(forecast.data):
        forecast.data = np.ma.filled(forecast.data, np.nan)

    for coord_slice in forecast.slices_over(coord):
        forecast_data.append(coord_slice.data.flatten())
    if transpose:
        forecast_data = np.asarray(forecast_data).T
    return np.array(forecast_data)


def flatten_ignoring_masked_data(
    data_array: Union[MaskedArray, ndarray], preserve_leading_dimension: bool = False
) -> ndarray:
    """
    Flatten an array, selecting only valid data if the array is masked. There
    is also the option to reshape the resulting array so it has the same
    leading dimension as the input array, but the other dimensions of the
    array are flattened. It is assumed that each of the slices
    along the leading dimension are masked in the same way. This functionality
    is used in EstimateCoefficientsForEnsembleCalibration when realizations
    are used as predictors.

    Args:
        data_array:
            An array or masked array to be flattened. If it is masked and the
            leading dimension is preserved the mask must be the same for every
            slice along the leading dimension.
        preserve_leading_dimension:
            Default False.
            If True the flattened array is reshaped so it has the same leading
            dimension as the input array. If False the returned array is 1D.

    Returns:
        A flattened array containing only valid data. Either 1D or, if
        preserving the leading dimension 2D. In the latter case the
        leading dimension is the same as the input data_array.

    Raises:
        ValueError: If preserving the leading dimension and the mask on the
                    input array is not the same for every slice along the
                    leading dimension.
    """
    if np.ma.is_masked(data_array):
        # If we have multiple 2D x-y slices check that the mask is the same for
        # each slice along the leading dimension.
        if data_array.ndim > 2:
            first_slice_mask = data_array[0].mask

            for i in range(1, data_array.shape[0]):
                if not np.all(first_slice_mask == data_array[i].mask):
                    msg = (
                        "The mask on the input array is not the same for "
                        "every slice along the leading dimension."
                    )
                    raise ValueError(msg)
        # If the mask is ok, select the unmasked data, flattening it at
        # the same time.
        result = data_array[~data_array.mask]
    else:
        result = data_array.flatten()
    if preserve_leading_dimension:
        # Reshape back to give the same leading dimension in the array. The 2nd
        # dimension is inferred through the use of -1.
        final_shape = (data_array.shape[0], -1)
        result = result.reshape(final_shape)
    return result


def check_predictor(predictor: str) -> None:
    """
    Check the predictor at the start of the process methods in relevant
    ensemble calibration plugins, to avoid having to check and raise an error
    later.

    Args:
        predictor:
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.

    Raises:
        ValueError: If the predictor is not valid.
    """
    if predictor.lower() not in ["mean", "realizations"]:
        msg = (
            "The requested value for the predictor {} is not an accepted "
            "value. Accepted values are 'mean' or 'realizations'"
        ).format(predictor.lower())
        raise ValueError(msg)


def filter_non_matching_cubes(
    historic_forecast: Cube, truth: Cube, additional_fields: Optional[CubeList] = None
) -> Tuple[Cube, Cube, Optional[CubeList]]:
    """
    Provide filtering for the historic forecast and truth to make sure
    that these contain matching validity times. This ensures that any
    mismatch between the historic forecasts and truth is dealt with.

    Args:
        historic_forecast:
            Cube of historic forecasts that potentially contains
            a mismatch compared to the truth.
        truth:
            Cube of truth that potentially contains a mismatch
            compared to the historic forecasts.
        additional_fields:


    Returns:
        - Cube of historic forecasts where any mismatches with
          the truth cube have been removed.
        - Cube of truths where any mismatches with
          the historic_forecasts cube have been removed.

    Raises:
        ValueError: The filtering has found no matches in validity time
            between the historic forecasts and the truths.
    """
    matching_historic_forecasts = iris.cube.CubeList([])
    matching_truths = iris.cube.CubeList([])
    matching_additional_fields = iris.cube.CubeList([])
    for hf_slice in historic_forecast.slices_over("time"):
        if hf_slice.coord("time").has_bounds():
            point = iris_time_to_datetime(
                hf_slice.coord("time"), point_or_bound="point"
            )
            (bounds,) = iris_time_to_datetime(
                hf_slice.coord("time"), point_or_bound="bound"
            )
            coord_values = {
                "time": lambda cell: point[0] == cell.point
                and bounds[0] == cell.bound[0]
                and bounds[1] == cell.bound[1]
            }
        else:
            coord_values = {
                "time": iris_time_to_datetime(
                    hf_slice.coord("time"), point_or_bound="point"
                )
            }

        constr = iris.Constraint(coord_values=coord_values)
        truth_slice = truth.extract(constr)

        if truth_slice and not additional_fields:
            matching_historic_forecasts.append(hf_slice)
            matching_truths.append(truth_slice)

        if additional_fields:
            constr = iris.Constraint(coord_values=coord_values)
            af_slices = [
                af_cube.extract(constr)
                for af_cube in additional_fields
                if af_cube.extract(constr) is not None
            ]
            if af_slices:
                matching_historic_forecasts.append(hf_slice)
                matching_truths.append(truth_slice)
                matching_additional_fields.extend(af_slices)

    if additional_fields:
        matching_additional_fields.extend(
            [af_cube for af_cube in additional_fields if not af_cube.coords("time")]
        )

    if not matching_historic_forecasts and not matching_truths:
        msg = (
            "The filtering has found no matches in validity time "
            "between the historic forecasts and the truths."
        )
        raise ValueError(msg)

    matching_additional_fields = (
        matching_additional_fields.merge() if matching_additional_fields else None
    )

    return (
        matching_historic_forecasts.merge_cube(),
        matching_truths.merge_cube(),
        matching_additional_fields,
    )


def create_unified_frt_coord(forecast_reference_time: DimCoord) -> DimCoord:
    """
    Constructs a single forecast reference time coordinate from a multi-valued
    coordinate. The new coordinate records the maximum range of bounds of
    the input forecast reference times, with the point value set to the latest
    of those in the inputs.

    Args:
        forecast_reference_time:
            The forecast_reference_time coordinate to be used in the
            coordinate creation.

    Returns:
        A dimension coordinate containing the forecast reference time
        coordinate with suitable bounds. The coordinate point is that
        of the latest contributing forecast.
    """
    frt_point = forecast_reference_time.points.max()
    frt_bounds_min = forecast_reference_time.points.min()
    frt_bounds_max = frt_point
    if forecast_reference_time.has_bounds():
        frt_bounds_min = min(frt_bounds_min, forecast_reference_time.bounds.min())
        frt_bounds_max = max(frt_bounds_max, forecast_reference_time.bounds.max())
    frt_bounds = (frt_bounds_min, frt_bounds_max)
    return forecast_reference_time[0].copy(points=frt_point, bounds=frt_bounds)


def merge_land_and_sea(calibrated_land_only: Cube, uncalibrated: Cube) -> None:
    """
    Merge data that has been calibrated over the land with uncalibrated data.
    Calibrated data will have masked data over the sea which will need to be
    filled with the uncalibrated data.

    Args:
        calibrated_land_only:
            A cube that has been calibrated over the land, with sea points
            masked out. Either realizations, probabilities or percentiles.
            Data is modified in place.
        uncalibrated:
            A cube of uncalibrated data with valid data over the sea. Either
            realizations, probabilities or percentiles. Dimension coordinates
            must be the same as the calibrated_land_only cube.

    Raises:
        ValueError: If input cubes do not have the same input dimensions.
    """
    # Check dimensions the same on both cubes.
    if calibrated_land_only.dim_coords != uncalibrated.dim_coords:
        message = "Input cubes do not have the same dimension coordinates"
        raise ValueError(message)
    # Merge data if calibrated_land_only data is masked.
    if np.ma.is_masked(calibrated_land_only.data):
        new_data = calibrated_land_only.data.data
        mask = calibrated_land_only.data.mask
        new_data[mask] = uncalibrated.data[mask]
        calibrated_land_only.data = new_data


def forecast_coords_match(first_cube: Cube, second_cube: Cube) -> None:
    """
    Determine if two cubes have equivalent forecast_periods and that the hours
    of the forecast_reference_time coordinates match. Only the point of the
    forecast reference time coordinate is checked to ensure that a calibration
    / coefficient cube matches the forecast cube, as appropriate.

    Args:
        first_cube:
            First cube to compare.
        second_cube:
            Second cube to compare.

    Raises:
        ValueError: The two cubes are not equivalent.
    """
    mismatches = []
    if (
        first_cube.coord("forecast_period").points
        != second_cube.coord("forecast_period").points
    ):
        mismatches.append("forecast_period")

    if get_frt_hours(first_cube.coord("forecast_reference_time")) != get_frt_hours(
        second_cube.coord("forecast_reference_time")
    ):
        mismatches.append("forecast_reference_time hours")
    if mismatches:
        msg = "The following coordinates of the two cubes do not match: {}"
        raise ValueError(msg.format(", ".join(mismatches)))


def get_frt_hours(forecast_reference_time: DimCoord) -> Set[int]:
    """
    Returns a set of integer representations of the hour of the
    forecast reference time.

    Args:
        forecast_reference_time:
            The forecast_reference_time coordinate to extract the hours from.

    Returns:
        A set of integer representations of the forecast reference time
        hours.
    """
    frt_hours = []
    for frt in forecast_reference_time.cells():
        frt_hours.append(np.int32(frt.point.hour))
    return set(frt_hours)


def check_forecast_consistency(forecasts: Cube) -> None:
    """
    Checks that the forecast cubes have a consistent forecast reference time
    hour and a consistent forecast period.

    Args:
        forecasts:

    Raises:
        ValueError: Forecast cubes have differing forecast reference time hours
        ValueError: Forecast cubes have differing forecast periods
    """
    frt_hours = get_frt_hours(forecasts.coord("forecast_reference_time"))

    if len(frt_hours) != 1:
        msg = (
            "Forecasts have been provided with differing hours for the "
            "forecast reference time {}"
        )
        raise ValueError(msg.format(frt_hours))
    if len(forecasts.coord("forecast_period").points) != 1:
        msg = "Forecasts have been provided with differing forecast periods {}"
        raise ValueError(msg.format(forecasts.coord("forecast_period").points))


def statsmodels_available() -> bool:
    """True if statsmodels library is importable.

    Returns:
        If True, statsmodels is available, otherwise, False.
    """
    if importlib.util.find_spec("statsmodels"):
        return True
    return False


def standardise_forecast_and_truths(
    historic_forecasts, truths, global_standardise=False, using_forecasts=False
):
    """Standardise the forecast and truths by subtracting the mean and dividing
    by the standard deviation.

    Args:
        forecast (iris.cube.Cube)
        truth (iris.cube.Cube)
        global_standardise
        using_forecasts
            Standardise the truths using the forecast mean and
            standard deviation

    Returns:
        Tuple:
    """
    if global_standardise:
        hf_coords = historic_forecasts.coords(dim_coords=True)
        truth_coords = truths.coords(dim_coords=True)
    else:
        hf_coords = ["realization", "time"]
        truth_coords = "time"

    # standardise ensemble members using the mean and standard deviation of the ensemble mean
    forecast_mean = historic_forecasts.collapsed(hf_coords, iris.analysis.MEAN)
    forecast_mean.rename("fbar")
    forecast_sd = historic_forecasts.collapsed(hf_coords, iris.analysis.STD_DEV)
    forecast_sd.rename("fsig")

    std_forecast = (historic_forecasts - forecast_mean) / forecast_sd
    std_forecast.rename(historic_forecasts.name())

    if using_forecasts:
        std_truth = (truths - forecast_mean) / forecast_sd
        std_truth.rename(truths.name())
        # Replace masked values created by dividing existing NaN truth values
        # by a number (truth_sd). Otherwise, the "flatten_ignoring_masked_data"
        # call in compute_initial_guess ignores these masked truths and ends up
        # with a different number of points between the truth and the forecast.
        std_truth.data = std_truth.data.filled(np.nan)
        return std_forecast, std_truth, forecast_mean, forecast_sd, None, None

    # If using the truth to standardise
    if not truths.coords("time", dim_coords=True):
        # Handle a training dataset with a single timestep.
        msg = ("The truths cube provided does not have a time dimension. "
            "Multiple times are required to collapse over the time dimension "
            "to calculate the climatological mean or standard deviation.")
        raise ValueError(msg)
        #truth_mean = truths.copy(np.full(truths.shape, np.nanmean(truths.data)))
        #truth_sd = truths.copy(np.full(truths.shape, np.nanstd(truths.data)))

    # Use nanmean and nanstd as observations can sometimes be missing i.e. nan.
    from iris.analysis import WeightedAggregator

    nanmean = WeightedAggregator("mean", np.nanmean)
    nanstd = WeightedAggregator("standard_deviation", np.nanstd)

    truth_mean = truths.collapsed(truth_coords, nanmean)
    truth_mean.rename("ybar")

    truth_sd = truths.collapsed(truth_coords, nanstd)
    truth_sd.rename("ysig")

    std_truth = (truths - truth_mean) / truth_sd
    std_truth.rename(truths.name())
    # Replace masked values created by dividing existing NaN truth values
    # by a number (truth_sd). Otherwise, the "flatten_ignoring_masked_data"
    # call in compute_initial_guess ignores these masked truths and ends up
    # with a different number of points between the truth and the forecast.
    std_truth.data = std_truth.data.filled(np.nan)

    # This check has been commented out because there are instances where a
    # particular site has only one observation from a 30 day training dataset.
    # This leads to a standard deviation of 0 for that site. A standard
    # deviation of 0 will result in std_truth being NaN for the site anyway.
    # if np.any(np.isclose(truth_sd.data, 0)):
    #     msg = ("Standardised truths cannot be calculated if the truth "
    #            "standard deviation is zero. Increasing the training dataset "
    #            "length may solve this issue.")
    #     raise ValueError(msg)

    # # Ensure that masked values in the truth created by the standardisation
    # # are also masked in the forecast.
    # (rdim,) = std_forecast.coord_dims("realization")
    # expanded_truth = np.repeat(
    #     np.expand_dims(std_truth.data.mask, axis=rdim),
    #     len(std_forecast.coord("realization").points),
    #     axis=rdim,
    # )
    # std_forecast.data = np.ma.masked_where(expanded_truth, std_forecast.data)
    return std_forecast, std_truth, forecast_mean, forecast_sd, truth_mean, truth_sd


def standardise_forecasts(historic_forecasts, hf_coords=["realization", "time"]):
    """Standardise the forecast by subtracting the mean and dividing
    by the standard deviation.

    Args:
        forecast (iris.cube.Cube)
        truth (iris.cube.Cube)

    Returns:
        Tuple:
    """
    # standardise ensemble members using the mean and standard deviation of the ensemble mean
    forecast_mean = historic_forecasts.collapsed(hf_coords, iris.analysis.MEAN)
    forecast_mean.rename("fbar")
    forecast_sd = historic_forecasts.collapsed(hf_coords, iris.analysis.STD_DEV)
    forecast_sd.rename("fsig")
    std_forecast = (historic_forecasts - forecast_mean) / forecast_sd
    std_forecast.rename(historic_forecasts.name())
    return std_forecast, forecast_mean, forecast_sd


def standardise_truths(truths, truth_coords=["time"]):
    """Standardise the truths by subtracting the mean and dividing by the
    standard deviation.

    Args:
        truths (iris.cube.Cube)

    Returns:
        Tuple:
    """
    # Use nanmean and nanstd as observations can sometimes be missing i.e. nan.
    from iris.analysis import WeightedAggregator

    nanmean = WeightedAggregator("mean", np.nanmean)
    nanstd = WeightedAggregator("standard_deviation", np.nanstd)

    truth_mean = truths.collapsed(truth_coords, nanmean)
    truth_mean.rename("ybar")

    truth_sd = truths.collapsed(truth_coords, nanstd)
    truth_sd.rename("ysig")

    std_truth = (truths - truth_mean) / truth_sd
    std_truth.rename(truths.name())
    std_truth.data = std_truth.data.filled(np.nan)
    return std_truth, truth_mean, truth_sd
