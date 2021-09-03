#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Script to apply coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

import warnings

import pandas as pd
import numpy as np
import iris
from iris.exceptions import CoordinateNotFoundError
from scipy.stats import boxcox, yeojohnson
from improver.argparser import ArgParser
from improver.ensemble_calibration.ensemble_calibration import (
    ApplyCoefficientsFromEnsembleCalibration, apply_coefficients_from_regimes,
    LinearMixedModelCoefficientApplication)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    EnsembleReordering,
    GeneratePercentilesFromMeanAndVariance,
    GeneratePercentilesFromProbabilities,
    GenerateProbabilitiesFromMeanAndVariance,
    RebadgePercentilesAsRealizations,
    ResamplePercentiles)
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.metadata.amend import amend_metadata
from improver.ensemble_calibration.ensemble_calibration_utilities import get_regime_probabilities

def correct_metadata(cube):
    attributes = {"um_version": "delete"}
    coordinates = {"forecast_period": {"units": "seconds"},
                   "time": {"units": "seconds since 1970-01-01 00:00:00"},
                   "forecast_reference_time":
                       {"units": "seconds since 1970-01-01 00:00:00"}}
    cube = amend_metadata(cube, attributes=attributes,
                          coordinates=coordinates)
    cube.coord(axis="x").points = \
        cube.coord(axis="x").points.astype(np.float32)
    coordi_list = ["forecast_reference_time", "time"]
    for coordi in coordi_list:
        cube.coord(coordi).points = \
            cube.coord(coordi).points.astype(np.int64)
    cube.coord("height").points = \
        cube.coord("height").points.astype(np.float32)
    cube.coord("forecast_period").points = \
        cube.coord("forecast_period").points.astype(np.int32)
    cube.coord("realization").points = \
        cube.coord("realization").points.astype(np.int32)
    return cube


def main(argv=None):
    """Load in arguments for applying coefficients for Ensemble Model Output
       Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
       Regression (NGR). The coefficients are applied to the forecast
       that is supplied, so as to calibrate the forecast. The calibrated
       forecast is written to a netCDF file. If no coefficients are supplied
       the input forecast is returned unchanged.
    """
    parser = ArgParser(
        description='Apply coefficients for Ensemble Model Output '
                    'Statistics (EMOS), otherwise known as Non-homogeneous '
                    'Gaussian Regression (NGR). The supported input formats '
                    'are realizations, probabilities and percentiles. '
                    'The forecast will be converted to realizations before '
                    'applying the coefficients and then converted back to '
                    'match the input format.')
    # Filepaths for the forecast, EMOS coefficients and the output.
    parser.add_argument(
        'forecast_filepath', metavar='FORECAST_FILEPATH',
        help='A path to an input NetCDF file containing the forecast to be '
             'calibrated. The input format could be either realizations, '
             'probabilities or percentiles.')
    parser.add_argument(
        'coefficients_filepath',
        metavar='COEFFICIENTS_FILEPATH', nargs='?',
        help='(Optional) A path to an input NetCDF file containing the '
             'coefficients used for calibration. If this file is not '
             'provided the input forecast is returned unchanged.')
    parser.add_argument(
        'output_filepath', metavar='OUTPUT_FILEPATH',
        help='The output path for the processed NetCDF')
    parser.add_argument(
        '--distribution', metavar='DISTRIBUTION',
        default="gaussian",
        choices=['gaussian', 'truncated_gaussian', 'gaussian_lmm', 'logistic', 't', 'skew_normal',
                 'gamma', 'log_normal', 'truncated_logistic', 'generalized_logistic', 'normal_mixture_model',
                 'trunc_normal_mixture_model'],
        help='The distribution that will be used for '
             'calibration. This will be dependent upon the '
             'input phenomenon. This has to be supported by '
             'the minimisation functions in '
             'ContinuousRankedProbabilityScoreMinimisers.')
    # Optional arguments.
    parser.add_argument(
        '--num_realizations', metavar='NUMBER_OF_REALIZATIONS',
        default=None, type=np.int32,
        help='Optional argument to specify the number of '
             'ensemble realizations to produce. '
             'If the current forecast is input as probabilities or '
             'percentiles then this argument is used to create the requested '
             'number of realizations. In addition, this argument is used to '
             'construct the requested number of realizations from the mean '
             'and variance output after applying the EMOS coefficients.'
             'Default will be the number of realizations in the raw input '
             'file, if realizations are provided as input, otherwise if the '
             'input format is probabilities or percentiles, then an error '
             'will be raised if no value is provided.')
    parser.add_argument(
        '--random_ordering', default=False,
        action='store_true',
        help='Option to reorder the post-processed forecasts randomly. If not '
             'set, the ordering of the raw ensemble is used. This option is '
             'only valid when the input format is realizations.')
    parser.add_argument(
        '--random_seed', metavar='RANDOM_SEED', default=None,
        help='Option to specify a value for the random seed for testing '
             'purposes, otherwise, the default random seed behaviour is '
             'utilised. The random seed is used in the generation of the '
             'random numbers used for either the random_ordering option to '
             'order the input percentiles randomly, rather than use the '
             'ordering from the raw ensemble, or for splitting tied values '
             'within the raw ensemble, so that the values from the input '
             'percentiles can be ordered to match the raw ensemble.')
    parser.add_argument(
        '--ecc_bounds_warning', default=False,
        action='store_true',
        help='If True, where the percentiles exceed the ECC bounds range, '
             'raise a warning rather than an exception. This occurs when the '
             'current forecast is in the form of probabilities and is '
             'converted to percentiles, as part of converting the input '
             'probabilities into realizations.')
    parser.add_argument(
        '--predictor_of_mean', metavar='PREDICTOR_OF_MEAN',
        choices=['mean', 'realizations'], default='mean',
        help='String to specify the predictor used to calibrate the forecast '
             'mean. Currently the ensemble mean ("mean") and the ensemble '
             'realizations ("realizations") are supported as options. '
             'Default: "mean".')
    parser.add_argument(
        '--regime_filepath', default=None,
        help='Path to the input csv file containing historic '
             'regime occurrences.')
    parser.add_argument(
        '--val_time', default=False,
        action='store_true',
        help='Logical that is true if regime-dependent'
             'post-processing is to use the regime predicted'
             'by the ensemble members at the forecast '
             'validation time, and false if the regime at'
             'the initialisation time is to be used.')
    parser.add_argument(
        '--sitespecific', default=False,
        action='store_true',
        help='Logical that specifies whether or not EMOS '
             'coefficients are to be estimated for each '
             'grid point separately.')
    parser.add_argument(
        '--standardise', default=False,
        action='store_true',
        help='Logical that specifies whether forecasts and'
            'observations are to be standardised using local'
            'means and standard deviations prior to '
            'post-processing.')
    parser.add_argument(
        '--log_transform', default=False,
        action='store_true',
        help='Logical that specifies whether forecasts and'
             'observations are to be transformed back from' 
             'the log scale after coefficients are applied.')
    parser.add_argument(
        '--sqrt_transform', default=False,
        action='store_true',
        help='Logical that specifies whether forecasts and'
             'observations are to be transformed back from' 
             'the square-root scale after coefficients are applied.')
    parser.add_argument(
        '--boxcox_transform', default=False,
        action='store_true',
        help='Logical that specifies whether forecasts and'
             'observations are to be transformed back from' 
             'the box-cox transformation after coefficients are applied.')
    parser.add_argument(
        '--yeojohnson_transform', default=False,
        action='store_true',
        help='Logical that specifies whether forecasts and'
             'observations are to be transformed back from' 
             'the yeo-johnson transformation after coefficients are applied.')
    parser.add_argument(
        '--yeojohnson_transform_standardised', default=False,
        action='store_true',
        help='Logical that specifies whether forecasts and'
             'observations are to be transformed back from' 
             'the yeo-johnson transformation after coefficients are applied,'
             'after standardising the forecasts using local data.')

    args = parser.parse_args(args=argv)

    # Load Cubes
    current_forecast = load_cube(args.forecast_filepath)
    current_forecast = correct_metadata(current_forecast)

    coeffs = load_cube(args.coefficients_filepath)

    reg_df = None
    if args.regime_filepath:
        reg_df = pd.read_csv(args.regime_filepath, sep=" ", header=None)
        reg_df.columns = ["year", "month", "day", "hour", "T0", "T24", "T48",
                          "T72", "T96", "T120", "T144", "End"]

    if args.log_transform:
        print("Performing log transformation")
        current_forecast.data[current_forecast.data == 0] = 0.00001
        current_forecast.data = np.log(current_forecast.data)
    elif args.sqrt_transform:
        print("Performing square-root transformation")
        current_forecast.data = np.sqrt(current_forecast.data)
    elif args.boxcox_transform: # will only work for one set of parameters estimated over all data
        print("Performing Box-Cox transformation")
        bc_lambda = coeffs.data[coeffs.coord("coefficient_name").points == "lam_obs"].data[0]
        truth_mean = coeffs.data[coeffs.coord("coefficient_name").points == "mean_obs"].data[0]
        truth_sd = coeffs.data[coeffs.coord("coefficient_name").points == "sd_obs"].data[0]
        truth_std_min = coeffs.data[coeffs.coord("coefficient_name").points == "std_min_obs"].data[0]
        print("Lambda is ", bc_lambda)
        current_forecast.data = ((current_forecast.data - truth_mean) / truth_sd) - truth_std_min
        if (current_forecast.data < 0).any():
            shift = np.floor(np.min(current_forecast.data))
            current_forecast.data -= shift
            truth_std_min += shift
            coeffs.data[coeffs.coord("coefficient_name").points == "std_min_obs"].data[0] = truth_std_min
        current_forecast.data[current_forecast.data == 0] = 0.00001
        bc_data = boxcox(current_forecast.data.flatten(), lmbda=bc_lambda)
        current_forecast.data = np.reshape(bc_data, current_forecast.data.shape)
    elif args.yeojohnson_transform:
        print("Performing Yeo-Johnson transformation")
        yj_lambda = coeffs.data[coeffs.coord("coefficient_name").points == "lam_obs"].data[0]
        truth_mean = coeffs.data[coeffs.coord("coefficient_name").points == "mean_obs"].data[0]
        truth_sd = coeffs.data[coeffs.coord("coefficient_name").points == "sd_obs"].data[0]
        print("Lambda is ", yj_lambda)
        current_forecast.data = (current_forecast.data - truth_mean) / truth_sd
        yj_data = yeojohnson(current_forecast.data.flatten(), lmbda=yj_lambda)
        current_forecast.data = np.reshape(yj_data, current_forecast.data.shape)
    elif args.yeojohnson_transform_standardised:
        print("Performing Yeo-Johnson standardised transformation")
        yj_lambda = coeffs[coeffs.coord("coefficient_name").points == "lam_obs"][0].data[0, 0]
        fcst_mean = coeffs[coeffs.coord("coefficient_name").points == "fbar"][0]
        fcst_sd = coeffs[coeffs.coord("coefficient_name").points == "fsig"][0]

        current_forecast.data = (current_forecast.data - fcst_mean.data)/fcst_sd.data
        print("Lambda is ", yj_lambda)
        print("Fcst mean is ", fcst_mean)
        print("Fcst sd is ", fcst_sd)
        print(current_forecast)
        yj_data = yeojohnson(current_forecast.data.flatten(), lmbda=yj_lambda)
        current_forecast.data = np.reshape(yj_data, current_forecast.data.shape)

    # Process Cube
    result = process(current_forecast, coeffs, args.distribution,
                     args.num_realizations, args.random_ordering,
                     args.random_seed, args.ecc_bounds_warning,
                     args.predictor_of_mean, reg_df, args.val_time,
                     args.sitespecific, args.standardise,
                     args.log_transform, args.sqrt_transform,
                     args.boxcox_transform, args.yeojohnson_transform,
                     args.yeojohnson_transform_standardised)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(current_forecast, coeffs, distribution="gaussian",
            num_realizations=None, random_ordering=False,
            random_seed=None, ecc_bounds_warning=False,
            predictor_of_mean='mean', reg_df=None, val_time=False,
            sitespecific=False, standardise=False,
            log_transform=False, sqrt_transform=False, boxcox_transform=False,
            yeojohnson_transform=False, yeojohnson_transform_standardised=False):
    """Applying coefficients for Ensemble Model Output Statistics.

    Load in arguments for applying coefficients for Ensemble Model Output
    Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). The coefficients are applied to the forecast
    that is supplied, so as to calibrate the forecast. The calibrated
    forecast is written to a cube. If no coefficients are provided the input
    forecast is returned unchanged.

    Args:
        current_forecast (iris.cube.Cube):
            A Cube containing the forecast to be calibrated. The input format
            could be either realizations, probabilities or percentiles.
        coeffs (iris.cube.Cube or None):
            A cube containing the coefficients used for calibration or None.
            If none then then current_forecast is returned unchanged.
        distribution (str):
            The distribution that will be used for calibration. This will be
            dependant upon the input phenomenon. Default is gaussian.
        num_realizations (numpy.int32):
            Optional argument to specify the number of ensemble realizations
            to produce. If the current forecast is input as probabilities or
            percentiles then this argument is used to create the requested
            number of realizations. In addition, this argument is used to
            construct the requested number of realizations from the mean and
            variance output after applying the EMOS coefficients.
            Default is None.
        random_ordering (bool):
            Option to reorder the post-processed forecasts randomly. If not
            set, the ordering of the raw ensemble is used. This option is
            only valid when the input format is realizations.
            Default is False.
        random_seed (int):
            Option to specify a value for the random seed for testing
            purposes, otherwise the default random seen behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the random_ordering option to order the input
            percentiles randomly, rather than use the ordering from the raw
            ensemble, or for splitting tied values within the raw ensemble,
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
            Default is None.
        ecc_bounds_warning (bool):
            If True, where the percentiles exceed the ECC bounds range,
            raises a warning rather than an exception. This occurs when the
            current forecasts is in the form of probabilities and is
            converted to percentiles, as part of converting the input
            probabilities into realizations.
            Default is False.
        predictor_of_mean (str):
            String to specify the predictor used to calibrate the forecast
            mean. Currently the ensemble mean "mean" as the ensemble
            realization "realization" are supported as options.
            Default is 'mean'
        distribution (str):
            String to specify the predictive distribution with which to apply
            EMOS. Must be one of "gaussian", "truncated_gaussian" or
            "gaussian_lmm".
        reg_df (pandas.DataFrame):
            A data frame containing a series of past dates and the coinciding
             weather regimes.
        val_time (bool):
            Logical that is true if regime-dependent
            post-processing is to use the regime predicted
            by the ensemble members at the forecast
            validation time, and false if the regime at
            the initialisation time is to be used.
        sitespecific (bool):
            Logical that specifies whether or not EMOS coefficients are to
            be estimated for each grid point separately.
        standardise (bool):
            Logical that specifies whether forecasts and
            observations are to be standardised using local
            means and standard deviations prior to
            post-processing.
        log_transform (bool):
            Logical that specifies whether forecasts and
            observations are to be transformed back from
            the log scale after coefficients are applied.
        sqrt_transform (bool):
            Logical that specifies whether forecasts and
            observations are to be transformed back from
            the square-root scale after coefficients are applied.
        boxcox_transform (bool):
            Logical that specifies whether forecasts and
            observations are to be transformed back from
            the Box-Cox scale after coefficients are applied.
        yeojohnson_transform (bool):
            Logical that specifies whether forecasts and
            observations are to be transformed back from
            the Yeo-Johnson scale after coefficients are applied.
        yeojohnson_transform_standardised (bool):
            Logical that specifies whether forecasts and
            observations are to be transformed back from
            the Yeo-Johnson scale after coefficients are applied,
            with local standardisation.
    Returns:
        result (iris.cube.Cube):
            The calibrated forecast cube.

    Raises:
        ValueError:
            If the current forecast is a coefficients cube.
        ValueError:
            If the coefficients cube does not have the right name of
            "emos_coefficients".
        ValueError:
            If the forecast type is 'percentiles' or 'probabilities' while no
            num_realizations are given.

    """
    if coeffs is None:
        msg = ("There are no coefficients provided for calibration. The "
               "uncalibrated forecast will be returned.")
        warnings.warn(msg)
        return current_forecast

    elif coeffs.name() != 'emos_coefficients':
        msg = ("The current coefficients cube does not have the "
               "name 'emos_coefficients'")
        raise ValueError(msg)

    if current_forecast.name() == 'emos_coefficients':
        msg = "The current forecast cube has the name 'emos_coefficients'"
        raise ValueError(msg)

    original_current_forecast = current_forecast.copy()
    try:
        find_percentile_coordinate(current_forecast)
        input_forecast_type = "percentiles"
    except CoordinateNotFoundError:
        input_forecast_type = "realizations"

    if current_forecast.name().startswith("probability_of"):
        input_forecast_type = "probabilities"
        # If probabilities, convert to percentiles.
        conversion_plugin = GeneratePercentilesFromProbabilities(
            ecc_bounds_warning=ecc_bounds_warning)
    elif input_forecast_type == "percentiles":
        # If percentiles, resample percentiles so that the percentiles are
        # evenly spaced.
        conversion_plugin = ResamplePercentiles(
            ecc_bounds_warning=ecc_bounds_warning)

    # If percentiles, re-sample percentiles and then re-badge.
    # If probabilities, generate percentiles and then re-badge.
    if input_forecast_type in ["percentiles", "probabilities"]:
        if not num_realizations:
            raise ValueError(
                "The current forecast has been provided as {0}. "
                "These {0} need to be converted to realizations "
                "for ensemble calibration. The num_realizations "
                "argument is used to define the number of realizations "
                "to construct from the input {0}, so if the "
                "current forecast is provided as {0} then "
                "num_realizations must be defined.".format(
                    input_forecast_type))
        current_forecast = conversion_plugin.process(
            current_forecast, no_of_percentiles=num_realizations)
        current_forecast = (
            RebadgePercentilesAsRealizations().process(current_forecast))

    # Default number of ensemble realizations is the number in
    # the raw forecast.
    if not num_realizations:
        num_realizations = len(
            current_forecast.coord('realization').points)

    # Apply coefficients as part of Ensemble Model Output Statistics (EMOS).
    if reg_df is not None:
        if distribution == "gaussian_lmm":
            ac = LinearMixedModelCoefficientApplication()
            calibrated_predictor, calibrated_variance = ac.process(
                current_forecast, coeffs,
                reg_df=reg_df, val_time=val_time,
                standardise=standardise)
        else:
            calibrated_predictor, calibrated_variance = \
                apply_coefficients_from_regimes(
                    current_forecast, coeffs,
                    predictor_of_mean_flag=predictor_of_mean,
                    reg_df=reg_df, val_time=val_time,
                    sitespecific=sitespecific,
                    standardise=standardise)
    elif distribution == "trunc_normal_mixture_model" or distribution == "normal_mixture_model":
        n_reg = len(coeffs.coord("regime").points)
        ac = ApplyCoefficientsFromEnsembleCalibration(
            predictor_of_mean_flag=predictor_of_mean)
        calibrated_predictor = []
        calibrated_variance = []
        for i in range(n_reg):
            calibrated_predictor_i, calibrated_variance_i = ac.process(
                current_forecast, coeffs[i], sitespecific=sitespecific,
                standardise=standardise)
            calibrated_predictor.append(calibrated_predictor_i)
            calibrated_variance.append(calibrated_variance_i)
    else:
        if yeojohnson_transform_standardised:
            yj_lambda = coeffs[coeffs.coord("coefficient_name").points == "lam_obs"][0].data[0, 0]
            obs_mean = coeffs[coeffs.coord("coefficient_name").points == "ybar"][0]
            obs_sd = coeffs[coeffs.coord("coefficient_name").points == "ysig"][0]
            coeffs = coeffs[:4, 0, 0]
        ac = ApplyCoefficientsFromEnsembleCalibration(
            predictor_of_mean_flag=predictor_of_mean)
        calibrated_predictor, calibrated_variance = ac.process(
            current_forecast, coeffs, sitespecific=sitespecific,
            standardise=standardise)

    # If input forecast is probabilities, convert output into probabilities.
    # If input forecast is percentiles, convert output into percentiles.
    # If input forecast is realizations, convert output into realizations.
    if input_forecast_type == "probabilities":
        result = GenerateProbabilitiesFromMeanAndVariance().process(
            calibrated_predictor, calibrated_variance,
            original_current_forecast)
    elif input_forecast_type == "percentiles":
        perc_coord = find_percentile_coordinate(original_current_forecast)
        result = GeneratePercentilesFromMeanAndVariance().process(
            calibrated_predictor, calibrated_variance,
            percentiles=perc_coord.points)
    elif input_forecast_type == "realizations":
        # Ensemble Copula Coupling to generate realizations
        # from mean and variance.
        if (distribution == "t") or (distribution == "skew_normal"):
            extra_params = coeffs.data[-1]
        elif distribution == "generalized_logistic":
            if standardise:
                extra_params = coeffs[coeffs.coord("coefficient_name").points == "shape"][0].data[0, 0]**2
            else:
                extra_params = coeffs.data[-1]**2
        elif distribution == "trunc_normal_mixture_model" or distribution == "normal_mixture_model":
            extra_params = get_regime_probabilities(current_forecast)[0]
        else:
            extra_params = None
        percentiles = GeneratePercentilesFromMeanAndVariance().process(
            calibrated_predictor, calibrated_variance,
            no_of_percentiles=num_realizations, distribution=distribution,
            extra_coeff=extra_params)
        result = EnsembleReordering().process(
            percentiles, current_forecast,
            random_ordering=random_ordering, random_seed=random_seed)
        if log_transform:
            result.data = np.exp(result.data)
        elif sqrt_transform:
            result.data = result.data ** 2
        elif boxcox_transform:
            bc_lambda = coeffs.data[coeffs.coord("coefficient_name").points == "lam_obs"].data[0]
            truth_mean = coeffs.data[coeffs.coord("coefficient_name").points == "mean_obs"].data[0]
            truth_sd = coeffs.data[coeffs.coord("coefficient_name").points == "sd_obs"].data[0]
            truth_std_min = coeffs.data[coeffs.coord("coefficient_name").points == "std_min_obs"].data[0]
            print("Lambda is ", bc_lambda)
            if bc_lambda == 0:
                bc_data = np.exp(result.data.flatten())
            else:
                bc_data = (bc_lambda*result.data.flatten() + 1) ** (1/bc_lambda)
            bc_data = (bc_data + truth_std_min)*truth_sd + truth_mean
            result.data = np.reshape(bc_data, result.data.shape)
        elif yeojohnson_transform:
            yj_lambda = coeffs.data[coeffs.coord("coefficient_name").points == "lam_obs"].data[0]
            truth_mean = coeffs.data[coeffs.coord("coefficient_name").points == "mean_obs"].data[0]
            truth_sd = coeffs.data[coeffs.coord("coefficient_name").points == "sd_obs"].data[0]
            print("Lambda is ", yj_lambda)
            result_data = result.data.flatten()
            result_data_neg = result_data[result_data < 0]
            result_data_pos = result_data[result_data >= 0]
            yj_data = result_data.copy()
            yj_data[result_data < 0] = 1 - (1 - (2 - yj_lambda)*result_data_neg) ** (1/(2 - yj_lambda))
            yj_data[result_data >= 0] = ((yj_lambda*result_data_pos + 1) ** (1/yj_lambda)) - 1
            if yj_lambda == 0:
                yj_data[result_data >= 0] = np.exp(result_data_pos) - 1
            elif yj_lambda == 2:
                yj_data[result_data < 0] = 1 - np.exp(-result_data_neg)
            yj_data = yj_data*truth_sd + truth_mean
            result.data = np.reshape(yj_data, result.data.shape)
        elif yeojohnson_transform_standardised:
            print("Lambda is ", yj_lambda)
            result_data = result.data.flatten()
            result_data_neg = result_data[result_data < 0]
            result_data_pos = result_data[result_data >= 0]
            yj_data = result_data.copy()
            yj_data[result_data < 0] = 1 - (1 - (2 - yj_lambda)*result_data_neg) ** (1/(2 - yj_lambda))
            yj_data[result_data >= 0] = ((yj_lambda*result_data_pos + 1) ** (1/yj_lambda)) - 1
            if yj_lambda == 0:
                yj_data[result_data >= 0] = np.exp(result_data_pos) - 1
            elif yj_lambda == 2:
                yj_data[result_data < 0] = 1 - np.exp(-result_data_neg)
            obs_sd_data = np.repeat(obs_sd.data[np.newaxis, :, :], result.data.shape[0]).flatten()
            obs_mean_data = np.repeat(obs_mean.data[np.newaxis, :, :], result.data.shape[0]).flatten()
            print(obs_sd_data.shape)
            yj_data = yj_data*obs_sd_data + obs_mean_data
            result.data = np.reshape(yj_data, result.data.shape)
    return result


if __name__ == "__main__":
    main()
