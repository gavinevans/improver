# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
This module defines all the "plugins" specific for Ensemble Model Output
Statistics (EMOS).

.. Further information is available in:
.. include:: extended_documentation/calibration/ensemble_calibration/
   ensemble_calibration.rst

"""
import os
import warnings
from multiprocessing import Pool

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm

from improver import BasePlugin, PostProcessingPlugin
from improver.calibration.utilities import (
    check_forecast_consistency,
    check_predictor,
    convert_cube_data_to_2d,
    create_unified_frt_coord,
    filter_non_matching_cubes,
    flatten_ignoring_masked_data,
    forecast_coords_match,
    merge_land_and_sea,
)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParametersToPercentiles,
    ConvertLocationAndScaleParametersToProbabilities,
    ConvertProbabilitiesToPercentiles,
    EnsembleReordering,
    RebadgePercentilesAsRealizations,
    ResamplePercentiles,
)
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import collapsed, enforce_coordinate_ordering


class ContinuousRankedProbabilityScoreMinimisers(BasePlugin):
    """
    Minimise the Continuous Ranked Probability Score (CRPS)

    Calculate the optimised coefficients for minimising the CRPS based on
    assuming a particular probability distribution for the phenomenon being
    minimised.

    The number of coefficients that will be optimised depend upon the initial
    guess.

    Minimisation is performed using the Nelder-Mead algorithm for 200
    iterations to limit the computational expense.
    Note that the BFGS algorithm was initially trialled but had a bug
    in comparison to comparative results generated in R.

    """

    # The tolerated percentage change for the final iteration when
    # performing the minimisation.
    TOLERATED_PERCENTAGE_CHANGE = 5

    # An arbitrary value set if an infinite value is detected
    # as part of the minimisation.
    BAD_VALUE = np.float64(999999)

    def __init__(self, tolerance=0.01, max_iterations=1000, each_point=False):
        """
        Initialise class for performing minimisation of the Continuous
        Ranked Probability Score (CRPS).

        Args:
            tolerance (float):
                The tolerance for the Continuous Ranked Probability
                Score (CRPS) calculated by the minimisation. The CRPS is in
                the units of the variable being calibrated. The tolerance is
                therefore representative of how close to the actual value are
                we aiming to forecast for a particular variable. Once multiple
                iterations result in a CRPS equal to the same value within the
                specified tolerance, the minimisation will terminate.
            max_iterations (int):
                The maximum number of iterations allowed until the
                minimisation has converged to a stable solution. If the
                maximum number of iterations is reached, but the minimisation
                has not yet converged to a stable solution, then the available
                solution is used anyway, and a warning is raised. If the
                predictor_of_mean is "realizations", then the number of
                iterations may require increasing, as there will be
                more coefficients to solve for.

        """
        # Dictionary containing the functions that will be minimised,
        # depending upon the distribution requested. The names of these
        # distributions match the names of distributions in scipy.stats.
        self.minimisation_dict = {
            "norm": self.calculate_normal_crps,
            "truncnorm": self.calculate_truncated_normal_crps,
        }
        self.tolerance = tolerance
        # Maximum iterations for minimisation using Nelder-Mead.
        self.max_iterations = max_iterations
        self.each_point = each_point

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = (
            "<ContinuousRankedProbabilityScoreMinimisers: "
            "minimisation_dict: {}; tolerance: {}; max_iterations: {}>"
        )
        print_dict = {}
        for key in self.minimisation_dict:
            print_dict.update({key: self.minimisation_dict[key].__name__})
        return result.format(print_dict, self.tolerance, self.max_iterations)

    def process(
        self,
        initial_guess,
        forecast_predictor,
        truth,
        forecast_var,
        predictor,
        distribution,
    ):
        """
        Function to pass a given function to the scipy minimize
        function to estimate optimised values for the coefficients.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor (iris.cube.Cube):
                Cube containing the fields to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth (iris.cube.Cube):
                Cube containing the field, which will be used as truth.
            forecast_var (iris.cube.Cube):
                Cube containing the field containing the ensemble variance.
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            distribution (str):
                String used to access the appropriate function for use in the
                minimisation within self.minimisation_dict.

        Returns:
            list of float:
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].

        Raises:
            KeyError: If the distribution is not supported.

        Warns:
            Warning: If the minimisation did not converge.

        """

        def calculate_percentage_change_in_last_iteration(allvecs):
            """
            Calculate the percentage change that has occurred within
            the last iteration of the minimisation. If the percentage change
            between the last iteration and the last-but-one iteration exceeds
            the threshold, a warning message is printed.

            Args:
                allvecs (list):
                    List of numpy arrays containing the optimised coefficients,
                    after each iteration.

            Warns:
                Warning: If a satisfactory minimisation has not been achieved.
            """
            last_iteration_percentage_change = (
                np.absolute((allvecs[-1] - allvecs[-2]) / allvecs[-2]) * 100
            )
            if np.any(
                last_iteration_percentage_change > self.TOLERATED_PERCENTAGE_CHANGE
            ):
                np.set_printoptions(suppress=True)
                msg = (
                    "The final iteration resulted in a percentage change "
                    "that is greater than the accepted threshold of 5% "
                    "i.e. {}. "
                    "\nA satisfactory minimisation has not been achieved. "
                    "\nLast iteration: {}, "
                    "\nLast-but-one iteration: {}"
                    "\nAbsolute difference: {}\n"
                ).format(
                    last_iteration_percentage_change,
                    allvecs[-1],
                    allvecs[-2],
                    np.absolute(allvecs[-2] - allvecs[-1]),
                )
                # warnings.warn(msg)

        try:
            minimisation_function = self.minimisation_dict[distribution]
        except KeyError as err:
            msg = (
                "Distribution requested {} is not supported in {}"
                "Error message is {}".format(distribution, self.minimisation_dict, err)
            )
            raise KeyError(msg)

        # Ensure predictor is valid.
        check_predictor(predictor)

        preserve_leading_dimension = False
        if self.each_point:
            preserve_leading_dimension = True

        # Flatten the data arrays and remove any missing data.
        truth_data = flatten_ignoring_masked_data(
            truth.data, preserve_leading_dimension=preserve_leading_dimension
        )
        forecast_var_data = flatten_ignoring_masked_data(
            forecast_var.data, preserve_leading_dimension=preserve_leading_dimension
        )
        if predictor.lower() == "mean":
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data,
                preserve_leading_dimension=preserve_leading_dimension,
            )
        elif predictor.lower() == "realizations":
            enforce_coordinate_ordering(forecast_predictor, "realization")
            # Need to transpose this array so there are columns for each
            # ensemble member rather than rows.
            forecast_predictor_data = flatten_ignoring_masked_data(
                forecast_predictor.data, preserve_leading_dimension=True
            ).T

        # Increased precision is needed for stable coefficient calculation.
        # The resulting coefficients are cast to float32 prior to output.
        initial_guess = np.array(initial_guess, dtype=np.float64)
        forecast_predictor_data = forecast_predictor_data.astype(np.float64)
        forecast_var_data = forecast_var_data.astype(np.float64)
        truth_data = truth_data.astype(np.float64)
        sqrt_pi = np.sqrt(np.pi).astype(np.float64)

        if self.each_point:
            argument_list = []
            for index in range(forecast_predictor_data.shape[1]):
                argument_list.append(
                    (
                        minimisation_function,
                        initial_guess[index],
                        forecast_predictor_data[:, index],
                        truth_data[:, index],
                        forecast_var_data[:, index],
                        sqrt_pi,
                        predictor,
                    )
                )

            with Pool(os.cpu_count()) as pool:
                optimised_coeffs = pool.starmap(self.minimise_caller, argument_list)

            optimised_coeffs = [x.x.astype(np.float32) for x in optimised_coeffs]
            return np.array(np.transpose(optimised_coeffs)).reshape(
                (len(initial_guess[0]),) + forecast_predictor.data.shape[1:]
            )

        else:
            optimised_coeffs = self.minimise_caller(
                minimisation_function,
                initial_guess,
                forecast_predictor_data,
                truth_data,
                forecast_var_data,
                sqrt_pi,
                predictor,
            )

            if not optimised_coeffs.success:
                msg = (
                    "Minimisation did not result in convergence after "
                    "{} iterations. \n{}".format(
                        self.max_iterations, optimised_coeffs.message
                    )
                )
                warnings.warn(msg)
            calculate_percentage_change_in_last_iteration(optimised_coeffs.allvecs)
            return optimised_coeffs.x.astype(np.float32)

    def minimise_caller(
        self,
        minimisation_function,
        initial_guess,
        forecast_predictor_data,
        truth_data,
        forecast_var_data,
        sqrt_pi,
        predictor,
    ):
        optimised_coeffs = minimize(
            minimisation_function,
            initial_guess,
            args=(
                forecast_predictor_data,
                truth_data,
                forecast_var_data,
                sqrt_pi,
                predictor,
            ),
            method="Nelder-Mead",
            tol=self.tolerance,
            options={"maxiter": self.max_iterations, "return_all": True},
        )
        return optimised_coeffs

    def calculate_normal_crps(
        self, initial_guess, forecast_predictor, truth, forecast_var, sqrt_pi, predictor
    ):
        """
        Calculate the CRPS for a normal distribution.

        Scientific Reference:
        Gneiting, T. et al., 2005.
        Calibrated Probabilistic Forecasting Using Ensemble Model Output
        Statistics and Minimum CRPS Estimation.
        Monthly Weather Review, 133(5), pp.1098-1118.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor (numpy.ndarray):
                Data to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth (numpy.ndarray):
                Data to be used as truth.
            forecast_var (numpy.ndarray):
                Ensemble variance data.
            sqrt_pi (numpy.ndarray):
                Square root of Pi
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        Returns:
            float:
                CRPS for the current set of coefficients. This CRPS is a mean
                value across all points.

        """
        if predictor.lower() == "mean":
            a, b, gamma, delta = initial_guess
            a_b = np.array([a, b], dtype=np.float64)
        elif predictor.lower() == "realizations":
            a, b, gamma, delta = (
                initial_guess[0],
                initial_guess[1:-2] ** 2,
                initial_guess[-2],
                initial_guess[-1],
            )
            a_b = np.array([a] + b.tolist(), dtype=np.float64)

        new_col = np.ones(truth.shape, dtype=np.float32)
        all_data = np.column_stack((new_col, forecast_predictor))
        mu = np.dot(all_data, a_b)
        sigma = np.sqrt(gamma ** 2 + delta ** 2 * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        if np.isfinite(np.min(mu / sigma)):
            result = np.nanmean(
                sigma * (xz * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / sqrt_pi)
            )
        else:
            result = self.BAD_VALUE
        return result

    def calculate_truncated_normal_crps(
        self, initial_guess, forecast_predictor, truth, forecast_var, sqrt_pi, predictor
    ):
        """
        Calculate the CRPS for a truncated normal distribution with zero
        as the lower bound.

        Scientific Reference:
        Thorarinsdottir, T.L. & Gneiting, T., 2010.
        Probabilistic forecasts of wind speed: Ensemble model
        output statistics by using heteroscedastic censored regression.
        Journal of the Royal Statistical Society.
        Series A: Statistics in Society, 173(2), pp.371-388.

        Args:
            initial_guess (list):
                List of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            forecast_predictor (numpy.ndarray):
                Data to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            truth (numpy.ndarray):
                Data to be used as truth.
            forecast_var (numpy.ndarray):
                Ensemble variance data.
            sqrt_pi (numpy.ndarray):
                Square root of Pi
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.

        Returns:
            float:
                CRPS for the current set of coefficients. This CRPS is a mean
                value across all points.

        """
        if predictor.lower() == "mean":
            a, b, gamma, delta = initial_guess
            a_b = np.array([a, b], dtype=np.float64)
        elif predictor.lower() == "realizations":
            a, b, gamma, delta = (
                initial_guess[0],
                initial_guess[1:-2] ** 2,
                initial_guess[-2],
                initial_guess[-1],
            )
            a_b = np.array([a] + b.tolist(), dtype=np.float64)

        new_col = np.ones(truth.shape, dtype=np.float32)
        all_data = np.column_stack((new_col, forecast_predictor))
        mu = np.dot(all_data, a_b)
        sigma = np.sqrt(gamma ** 2 + delta ** 2 * forecast_var)
        xz = (truth - mu) / sigma
        normal_cdf = norm.cdf(xz)
        normal_pdf = norm.pdf(xz)
        x0 = mu / sigma
        normal_cdf_0 = norm.cdf(x0)
        normal_cdf_root_two = norm.cdf(np.sqrt(2) * x0)
        if np.isfinite(np.min(mu / sigma)) or (np.min(mu / sigma) >= -3):
            result = np.nanmean(
                (sigma / normal_cdf_0 ** 2)
                * (
                    xz * normal_cdf_0 * (2 * normal_cdf + normal_cdf_0 - 2)
                    + 2 * normal_pdf * normal_cdf_0
                    - normal_cdf_root_two / sqrt_pi
                )
            )
        else:
            result = self.BAD_VALUE
        return result


class Boosting(BasePlugin):

    """Class to implement boosting as in Messner et al., 2017 and
    Messner et al., 2016. This results from this implementation has been
    checked as equal to the R implementation in Messner et al., 2016."""

    def __init__(self, distribution="norm", max_iterations=100, step_size=0.1):
        """Initialise the class."""
        self.distribution = distribution
        self.max_iterations = max_iterations
        self.step_size = step_size
        if self.distribution != "norm":
            msg = (
                "Nonhomogeneous boosting is only supported for a "
                "normal / gaussian distribution."
            )
            raise ValueError(msg)

        if max_iterations > 100:
            msg = "For nonhomogeneous boosting, the max_iterations allowed " "is 100."
            raise ValueError(msg)

    def _standardise_forecasts(self, forecasts):
        """Standardise the forecasts for each predictor by subtracting by the
        mean and dividing by the standard deviation.

        Args:
            forecasts (iris.cube.CubeList):
                CubeList where each cube is a different predictor.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Standardised forecasts, means and standard deviations of each
                predictor.
        """
        standardised_forecasts = []
        means = []
        stds = []
        for forecast in forecasts:
            forecast = forecast.data.data.flatten()
            means.append(np.mean(forecast))
            stds.append(np.std(forecast))
            standardised_forecasts.append((forecast - means[-1]) / stds[-1])
        return np.stack(standardised_forecasts), means, stds

    def _standardise_truth(self, truth):
        """Standardise the truth by subtracting by the mean and dividing
        by the standard deviation.

        Args:
            truth (iris.cube.CubeList):
                Truth cube.

        Returns
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Standardised truth, mean and standard deviation
        """
        return (truth - np.mean(truth)) / np.std(truth), np.mean(truth), np.std(truth)

    def _calculate_location_parameter(self, forecast_predictor, lp_coeffs):
        """Compute the location parameter and sum over each predictor.

        mu = xTbeta

        Args:
            forecast_predictor (numpy.ndarray):
                Array with predictors as rows with each event as a column.
            lp_coeffs (numpy.ndarray):
                Location parameter coefficients.

        Returns:
            numpy.ndarray
        """
        return np.sum(np.atleast_2d(lp_coeffs).T * forecast_predictor, axis=0)

    def _calculate_scale_parameter(self, forecast_var, sp_coeffs):
        """Compute the scale parameter and sum over each predictor.

        sigma = exp(zTgamma)

        Args:
            forecast_var (numpy.ndarray):
                Array with forecast variance as rows with each event as a column.
            sp_coeffs (numpy.ndarray):
                Scale parameter coefficients.

        Returns:
            numpy.ndarray
        """
        return np.exp(np.sum(np.atleast_2d(sp_coeffs).T * forecast_var, axis=0))

    def _compute_location_parameter_partial_derivatives(
        self, truth, location_parameter, scale_parameter
    ):
        """Partial derivative of the negative log-likelihood with respect to
        the location parameter taken from Appendix of Messner et al., 2017.

        Args:
            truth (numpy.ndarray):
                Truth array.
            location_parameter (numpy.ndarray):
                Location parameter array.
            scale_parameter (numpy.ndarray):
                Scale parameter array.

        Returns:
            numpy.ndarray
        """
        return -(truth - location_parameter) / (scale_parameter * scale_parameter)

    def _compute_scale_parameter_partial_derivatives(
        self, truth, location_parameter, scale_parameter
    ):
        """Partial derivative of the negative log-likelihood with respect to
        the scale parameter taken from Appendix of Messner et al., 2017. The
        partial derivative is multiplied by the scale parameter following
        Messner et al., 2016.

        Args:
            truth (numpy.ndarray):
                Truth array.
            location_parameter (numpy.ndarray):
                Location parameter array.
            scale_parameter (numpy.ndarray):
                Scale parameter array.

        Returns:
            numpy.ndarray
        """
        partial_derivative = -(
            -1 / scale_parameter
            + (truth - location_parameter) ** 2 / scale_parameter ** 3
        )
        return partial_derivative * scale_parameter

    def _find_correlation(self, forecast, partial_derivative):
        """Mean of the multiplication of the forecast descriptor by the
        partial derivative of the negative log-likelihood. Larger values
        indicate greater correlation between the forecast descriptor and the
        partial derivative.

        Args:
            forecast (numpy.ndarray):
                Forecast descriptor.
            partial_derivative (numpy.ndarray):
                Partial derivative of the negative log-likelihood.

        Returns:
            Tuple[numpy.ndarray, int]:
                Correlations and index of the maximum correlation.
        """
        means = np.mean(forecast * -partial_derivative, axis=1)
        return means, np.nanargmax(np.abs(means))

    def _update_coefficients(self, coeffs, corrcoef, corrcoef_index):
        """Update the coefficient with the maximum correlation by a specified
        fraction of the correlation.

        Args:
            coeffs (numpy.ndarray):
               Coefficients array.
            corrcoef (numpy.ndarray):
               Correlation.
            corrcoef_index (int):
                Index of the maximum correlation.

        Returns:
            numpy.ndarray
         """
        coeffs[corrcoef_index] = (
            coeffs[corrcoef_index] + self.step_size * corrcoef[corrcoef_index]
        )
        return coeffs

    def _negative_log_likelihood(
        self, truth, forecast_predictor, forecast_var, lp_coeffs, sp_coeffs
    ):
        """Compute the negative log likelihood following Messner et al., 2017.
        A normal distribution is assumed.

        Args:
            truth (numpy.ndarray):
                Truth array.
            forecast_predictor (numpy.ndarray):
                Forecast predictor.
            forecast_var (numpy.ndarray):
                Forecast variance.
            lp_coeffs (numpy.ndarray):
                Location parameter coefficients array.
            sp_coeffs (numpy.ndarray):
                Scale parameter coefficients array.

        Returns:
            numpy.ndarray:
                Sum of the negative log-likelihood.

        """
        location_parameter = self._calculate_location_parameter(
            forecast_predictor, lp_coeffs
        )
        scale_parameter = self._calculate_scale_parameter(forecast_var, sp_coeffs)

        nll = -np.log(
            norm.pdf((truth - location_parameter) / scale_parameter) / scale_parameter
        )
        nll[~np.isfinite(nll)] = np.nan
        sum_nll = np.nansum(nll)
        return sum_nll

    def coeffs_to_json(self, optimised_coeffs_dict):
        import json

        with open("coefficients.json", "w") as fp:
            json.dump(optimised_coeffs_dict, fp)

    def _convert_coefficients(
        self, optimised_coeffs, location_parameter, scale_parameter
    ):
        """Re-scale coefficients from standardised coefficients for
        application. This re-scaling is general for both the location parameter
        and scale parameter coefficients.

        Args:
            optimised_coeffs (numpy.ndarray):
                Standardised coefficients.
            location_parameter (numpy.ndarray):
                Location parameter array.
            scale_parameter (numpy.ndarray):
                Scale parameter array.

        Returns:
            numpy.ndarray
                Partially re-standardised coefficients.
        """
        optimised_coeffs[0] = optimised_coeffs[0] - np.sum(
            optimised_coeffs[1:] * location_parameter / scale_parameter
        )
        optimised_coeffs[1:] = optimised_coeffs[1:] / scale_parameter
        return optimised_coeffs

    def convert_lp_coefficients(
        self,
        optimised_coeffs,
        location_parameter,
        scale_parameter,
        truth_location_parameter,
        truth_scale_parameter,
    ):
        """Re-scale location parameter coefficients from standardised
        coefficients for application.

        Args:
            optimised_coeffs (numpy.ndarray):
                Standardised coefficients.
            location_parameter (numpy.ndarray):
                Location parameter array of the forecast.
            scale_parameter (numpy.ndarray):
                Scale parameter array of the forecast.
            truth_location_parameter (numpy.ndarray):
                Location parameter array of the truth.
            truth_scale_parameter (numpy.ndarray):
                Scale parameter array of the truth.

        Returns:
            numpy.ndarray
                Re-standardised location parameter coefficients.
        """
        optimised_coeffs = self._convert_coefficients(
            optimised_coeffs, location_parameter, scale_parameter
        )
        optimised_coeffs = optimised_coeffs * truth_scale_parameter
        optimised_coeffs[0] = optimised_coeffs[0] + truth_location_parameter
        return optimised_coeffs

    def convert_sp_coefficients(
        self,
        optimised_coeffs,
        location_parameter,
        scale_parameter,
        truth_scale_parameter,
    ):
        """Re-scale scale parameter coefficients from standardised
        coefficients for application.

        Args:
            optimised_coeffs (numpy.ndarray):
                Standardised coefficients.
            location_parameter (numpy.ndarray):
                Location parameter array of the forecast.
            scale_parameter (numpy.ndarray):
                Scale parameter array of the forecast.
            truth_location_parameter (numpy.ndarray):
                Location parameter array of the truth.
            truth_scale_parameter (numpy.ndarray):
                Scale parameter array of the truth.

        Returns:
            numpy.ndarray
                Re-standardised scale parameter coefficients.
        """
        optimised_coeffs = self._convert_coefficients(
            optimised_coeffs, location_parameter, scale_parameter
        )
        optimised_coeffs[0] = optimised_coeffs[0] + np.log(truth_scale_parameter)
        return optimised_coeffs

    def _apply_boosting(
        self, lp_coeffs, sp_coeffs, truth, forecast_predictor, forecast_var
    ):
        """Following Messner et al., 2017's description of Nonhomogeneous
        boosting.

        1. Compute partial derivatives of the negative log-likelihood (NLL) score
        with respect to the location and scale parameters.
        2. Find the predictor that has the maximum correlation with the partial
        derivative of the NLL.
        3. Tentatively update coefficients following 2., so that only the
        coefficient of the predictor with the maximum correction is incremented.
        4. Compare the NLL following the update for the location and scale
        parameter and choose to update either the location or scale parameter
        coefficients.

        Args:
             lp_coeffs (numpy.ndarray):
                Location parameter coefficients.
             sp_coeffs
                Scale parameter coefficients.
            truth (numpy.ndarray):
                Truth array.
            forecast_predictor (numpy.ndarray):
                Forecast predictor.
            forecast_var (numpy.ndarray):
                Forecast variance.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                Location parameter coefficients and scale parameter coefficients.
        """
        location_parameter = self._calculate_location_parameter(
            forecast_predictor, lp_coeffs
        )
        scale_parameter = self._calculate_scale_parameter(forecast_var, sp_coeffs)

        lp_pd = self._compute_location_parameter_partial_derivatives(
            truth, location_parameter, scale_parameter
        )
        sp_pd = self._compute_scale_parameter_partial_derivatives(
            truth, location_parameter, scale_parameter
        )

        corrcoef_lp, corrcoef_lp_index = self._find_correlation(
            forecast_predictor, lp_pd
        )
        corrcoef_sp, corrcoef_sp_index = self._find_correlation(forecast_var, sp_pd)

        new_lp_coeffs = self._update_coefficients(
            lp_coeffs.copy(), corrcoef_lp, corrcoef_lp_index
        )
        new_sp_coeffs = self._update_coefficients(
            sp_coeffs.copy(), corrcoef_sp, corrcoef_sp_index
        )

        if self._negative_log_likelihood(
            truth, forecast_predictor, forecast_var, new_lp_coeffs, sp_coeffs
        ) < self._negative_log_likelihood(
            truth, forecast_predictor, forecast_var, lp_coeffs, new_sp_coeffs
        ):
            lp_coeffs = new_lp_coeffs
        else:
            sp_coeffs = new_sp_coeffs

        return lp_coeffs, sp_coeffs

    def process(self, truth, forecast_predictors, forecast_vars):
        """Produce optimised coefficients using nonhomogeneous boosting.

        Args:
            truth (iris.cube.Cube):
                Truth cube.
            forecast_predictors (iris.cube.CubeList):
                Cubelist containing a separate cube for each predictor.
             forecast_vars (iris.cube.CubeList):
                Cubelist containing a separate cube for each predictor.

        Returns:
            numpy.ndarray:
                Optimised location parameter coefficients and scale parameter
                coefficients.

        """
        # import pandas as pd
        # data = {
        #     "truth": truth.data.flatten(),
        #     "forecast_predictor_0": forecast_predictors[0].data.flatten(),
        #     "forecast_predictor_1": forecast_predictors[1].data.flatten(),
        #     "forecast_predictor_2": forecast_predictors[2].data.flatten(),
        #     "forecast_predictor_3": forecast_predictors[3].data.flatten(),
        #     "forecast_var_0": forecast_vars[0].data.flatten(),
        #     "forecast_var_1": forecast_vars[1].data.flatten(),
        #     "forecast_var_2": forecast_vars[2].data.flatten(),
        #     "forecast_var_3": forecast_vars[3].data.flatten(),
        # }
        # df = pd.DataFrame(data=data)
        # df.to_csv("/home/h06/gevans/impro/improver_ml/predictor_df.csv")
        # Standardise the truth and forecasts.
        # Assume the mean and standard deviation of the forecasts and truths
        # are gaussian and therefore assume that the mean equals the
        # location parameter and the standard deviation equals the scale
        # parameter.
        truth, truth_mean, truth_std = self._standardise_truth(
            truth.data.data.flatten()
        )
        (
            forecast_predictor_data,
            forecast_predictor_mean,
            forecast_predictor_std,
        ) = self._standardise_forecasts(forecast_predictors)
        forecast_var_data, forecast_var_mean, forecast_var_std = self._standardise_forecasts(
            forecast_vars
        )

        # Pre-pend an additional array of zeroes for computing the intercept
        # coefficient.
        forecast_predictor_data = np.insert(
            forecast_predictor_data,
            int(0),
            np.zeros(forecast_predictor_data[0].shape, dtype=np.float32),
            axis=0,
        )
        forecast_var_data = np.insert(
            forecast_var_data,
            int(0),
            np.ones(forecast_var_data[0].shape, dtype=np.float32),
            axis=0,
        )

        optimised_lp_coeffs = np.zeros(forecast_predictor_data.shape[0])
        optimised_sp_coeffs = np.zeros(forecast_var_data.shape[0])
        optimised_coeffs_dict = {"location_parameter": {}, "scale_parameter": {}}
        for index in range(self.max_iterations):
            optimised_lp_coeffs, optimised_sp_coeffs = self._apply_boosting(
                optimised_lp_coeffs,
                optimised_sp_coeffs,
                truth,
                forecast_predictor_data,
                forecast_var_data,
            )
            optimised_coeffs_dict["location_parameter"][
                index
            ] = optimised_lp_coeffs.tolist()
            optimised_coeffs_dict["scale_parameter"][
                index
            ] = optimised_sp_coeffs.tolist()

        self.coeffs_to_json(optimised_coeffs_dict)

        # Re-standardise the coefficients for application.
        optimised_lp_coeffs = self.convert_lp_coefficients(
            optimised_lp_coeffs,
            forecast_predictor_mean,
            forecast_predictor_std,
            truth_mean,
            truth_std,
        )
        optimised_sp_coeffs = self.convert_sp_coefficients(
            optimised_sp_coeffs, forecast_var_mean, forecast_var_std, truth_std
        )

        return optimised_lp_coeffs.astype(np.float32), optimised_sp_coeffs.astype(np.float32)
        #np.concatenate((optimised_lp_coeffs.astype(np.float32), optimised_sp_coeffs.astype(np.float32)))

    def process_with_r(
        self, truth, forecast_predictor, forecast_var,
    ):
        import pandas as pd
        import rpy2.robjects as ro
        from rpy2.robjects import Formula, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        # truth_data, _, _ = self._standardise_truth(truth.data)
        # forecast_predictor_data = self._standardise(forecast_predictor.data.data)
        # forecast_var_data = self._standardise(forecast_var.data.data)

        data = {
            "truth": truth.data.flatten(),
            "forecast_predictor_0": forecast_predictor[0].data.flatten(),
            "forecast_predictor_1": forecast_predictor[1].data.flatten(),
            "forecast_predictor_2": forecast_predictor[2].data.flatten(),
            "forecast_predictor_3": forecast_predictor[3].data.flatten(),
            "forecast_var_0": forecast_var[0].data.flatten(),
            "forecast_var_1": forecast_var[1].data.flatten(),
            "forecast_var_2": forecast_var[2].data.flatten(),
            "forecast_var_3": forecast_var[3].data.flatten(),
        }
        # "forecast_predictor": np.array([fp.data.flatten() for fp in forecast_predictor]).flatten(),
        # "forecast_var": np.array([fv.data.flatten() for fv in forecast_var]).flatten()}
        print("data = ", data)
        df = pd.DataFrame(data=data)

        crch = importr("crch")
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_dataframe = ro.conversion.py2rpy(df)
        df.to_csv("/home/h06/gevans/impro/improver_ml/predictor_df.csv")
        crch_model = crch.crch(
            Formula(
                "truth~forecast_predictor_0+forecast_predictor_1+forecast_predictor_2+forecast_predictor_3|forecast_var_0+forecast_var_1+forecast_var_2+forecast_var_3"
            ),
            data=r_dataframe,
            dist=self.distribution,
            link_scale="log",
            method="boosting",
            maxit=self.max_iterations,
            mstop="aic",
        )
        coefficients = crch_model.rx2("coefficients")
        print("coefficients = ", crch_model.rx2("coefficients"))
        coeff_dict = dict(zip(coefficients.names, np.array(coefficients)))
        print("coeff_dict = ", coeff_dict)
        optimised_coeffs = np.array(
            list(coeff_dict["location"][:2])
            + coeff_dict["scale"][0]
            + coeff_dict["scale"][2]
        )

        return optimised_coeffs.astype(np.float32)


class EstimateCoefficientsForEnsembleCalibration(BasePlugin):
    """
    Class focussing on estimating the optimised coefficients for ensemble
    calibration.
    """

    # Logical flag for whether initial guess estimates for the coefficients
    # will be estimated using linear regression i.e.
    # ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = True, or whether default
    # values will be used instead i.e.
    # ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = False.
    ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = True

    def __init__(
        self,
        distribution,
        each_point=False,
        minimise_each_point=False,
        desired_units=None,
        predictor="mean",
        tolerance=0.01,
        max_iterations=1000,
        boosting=False,
        number_of_predictors=None
    ):
        """
        Create an ensemble calibration plugin that, for Nonhomogeneous Gaussian
        Regression, calculates coefficients based on historical forecasts and
        applies the coefficients to the current forecast.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            distribution (str):
                Name of distribution. Assume that a calibrated version of the
                current forecast could be represented using this distribution.
            each_point (bool):
                If True, coefficients are calculated independently for each
                point within the input cube by creating an initial guess and
                minimising each grid point independently. Please note this
                option is memory intensive and is unsuitable for gridded input,
                please consider using the minimise_each_point option.
            minimise_each_point (bool):
                If True, coefficients are calculated independently for each
                point within the input cube by minimising each grid point
                independently. Each point uses the default initial guess.
            desired_units (str or cf_units.Unit):
                The unit that you would like the calibration to be undertaken
                in. The current forecast, historical forecast and truth will be
                converted as required.
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            tolerance (float):
                The tolerance for the Continuous Ranked Probability
                Score (CRPS) calculated by the minimisation. The CRPS is in
                the units of the variable being calibrated. The tolerance is
                therefore representative of how close to the actual value are
                we aiming to forecast for a particular variable. Once multiple
                iterations result in a CRPS equal to the same value within the
                specified tolerance, the minimisation will terminate.
            max_iterations (int):
                The maximum number of iterations allowed until the
                minimisation has converged to a stable solution. If the
                maximum number of iterations is reached, but the minimisation
                has not yet converged to a stable solution, then the available
                solution is used anyway, and a warning is raised. If the
                predictor_of_mean is "realizations", then the number of
                iterations may require increasing, as there will be
                more coefficients to solve for.
            boosting (bool):
                If True, enable nonhomogeneous boosting following
                Messner et al., 2017 allowing multiple predictors to be provided.
                If False, coefficients are estimated using EMOS.
            number_of_predictors (Optional[int]):
                Number of predictors for boosting. An error is raised if this
                argument is specified without boosting enabled.

        """
        self.distribution = distribution
        self.each_point = each_point
        self.minimise_each_point = minimise_each_point
        self._validate_distribution()
        self.desired_units = desired_units
        # Ensure predictor is valid.
        check_predictor(predictor)
        self.predictor = predictor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.boosting = boosting
        self.number_of_predictors = number_of_predictors
        if boosting:
            self.minimiser = Boosting(
                distribution=self.distribution, max_iterations=self.max_iterations
            )
        else:
            self.minimiser = ContinuousRankedProbabilityScoreMinimisers(
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
                each_point=self.each_point | self.minimise_each_point,
            )

        # Setting default values for coeff_names.
        if boosting:
            # self.coeff_names = [f"{i}_{j}" for i in ["beta", "gamma"]
            #                     for j in range(self.number_of_predictors+1)]
            self.coeff_names = ["beta", "gamma"]
        else:
            self.coeff_names = ["alpha", "beta", "gamma", "delta"]

        if boosting and predictor == "realization":
            msg = ("Using Nonhomogeneous Gaussian Boosting with a realization "
                   "predictor is not yet supported.")
            raise ValueError(msg)

    def _validate_distribution(self):
        """Validate that the distribution supplied has a corresponding method
        for minimising the Continuous Ranked Probability Score.

        Raises:
            ValueError: If the distribution requested is not supported.

        """
        valid_distributions = (
            ContinuousRankedProbabilityScoreMinimisers().minimisation_dict.keys()
        )
        if self.distribution not in valid_distributions:
            msg = (
                "Given distribution {} not available. Available "
                "distributions are {}".format(self.distribution, valid_distributions)
            )
            raise ValueError(msg)

    def _get_statsmodels_availability(self):
        """Import the statsmodels module, if available.

        Returns:
            bool:
                True if the statsmodels module is available. Otherwise, False.

        Warns:
            ImportWarning: If the statsmodels module cannot be imported.
        """
        import importlib

        try:
            importlib.import_module("statsmodels")
        except (ModuleNotFoundError, ImportError):
            sm = None
            if self.predictor.lower() == "realizations":
                msg = (
                    "The statsmodels module cannot be imported. "
                    "Will not be able to calculate an initial guess from "
                    "the individual ensemble realizations. "
                    "A default initial guess will be used without "
                    "estimating coefficients from a linear model."
                )
                warnings.warn(msg, ImportWarning)
        else:
            import statsmodels.api as sm

        return sm

    def _set_attributes(self, historic_forecasts):
        """Set attributes for use on the EMOS coefficients cube.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            dict:
                Attributes for an EMOS coefficients cube including
                "diagnostic standard name", "distribution", "shape_parameters"
                and an updated title.
        """
        attributes = generate_mandatory_attributes([historic_forecasts])
        attributes["diagnostic_standard_name"] = historic_forecasts.name()
        attributes["distribution"] = self.distribution
        if self.distribution == "truncnorm":
            # For the CRPS minimisation, the truncnorm distribution is
            # truncated at zero.
            attributes["shape_parameters"] = np.array([0, np.inf], dtype=np.float32)
        attributes["title"] = "Ensemble Model Output Statistics coefficients"
        return attributes

    @staticmethod
    def _create_temporal_coordinates(historic_forecasts):
        """Create forecast reference time and forecast period coordinates
        for the EMOS coefficients cube.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            list of tuples:
                List of tuples of the temporal coordinates and the associated
                dimension. This format is suitable for use by iris.cube.Cube.
        """
        # Create forecast reference time coordinate.
        frt_coord = create_unified_frt_coord(
            historic_forecasts.coord("forecast_reference_time")
        )

        fp_coord = historic_forecasts.coord("forecast_period")

        if fp_coord.shape[0] != 1:
            msg = (
                "The forecast period must be the same for all historic forecasts. "
                "Forecast periods found: {}".format(fp_coord.points)
            )
            raise ValueError(msg)

        return [(frt_coord, None), (fp_coord, None)]

    @staticmethod
    def _create_spatial_coordinates(historic_forecasts):
        """Create spatial coordinates for the EMOS coefficients cube.

        Args:
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.

        Returns:
            list of tuples:
                List of tuples of the spatial coordinates and the associated
                dimension. This format is suitable for use by iris.cube.Cube.
        """
        spatial_coords_and_dims = []
        for axis in ["x", "y"]:
            spatial_coords_and_dims.append(
                (historic_forecasts.coord(axis=axis).collapsed(), None)
            )
        return spatial_coords_and_dims

    def _create_cubelist(
        self, optimised_coeffs, historic_forecasts, aux_coords_and_dims, attributes
    ):
        """Create a cubelist by combining the optimised coefficients and the
        appropriate metadata. The units of the alpha and gamma coefficients
        match the units of the historic forecast. If the predictor is the
        realizations, then the beta coefficient cube contains a realization
        coordinate.

        Args:
            optimised_coeffs (numpy.ndarray)
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.
            aux_coords_and_dims (list of tuples):
                List of tuples of the format [(coord, dim), (coord, dim)]
            attributes (dict):
                Attributes for an EMOS coefficients cube including
                "diagnostic standard name" and an updated title.

        Returns:
            cubelist (iris.cube.CubeList):
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.
        """
        cubelist = iris.cube.CubeList([])
        for optimised_coeff, coeff_name in zip(optimised_coeffs, self.coeff_names):
            coeff_units = "1"
            if coeff_name in ["alpha", "gamma"]:
                coeff_units = historic_forecasts.units
            dim_coords_and_dims = []
            if self.predictor.lower() == "realizations" and coeff_name == "beta":
                dim_coords_and_dims = [
                    (historic_forecasts.coord("realization").copy(), 0)
                ]
            cube = iris.cube.Cube(
                optimised_coeff,
                long_name=f"emos_coefficient_{coeff_name}",
                units=coeff_units,
                dim_coords_and_dims=dim_coords_and_dims,
                aux_coords_and_dims=aux_coords_and_dims,
                attributes=attributes,
            )
            cubelist.append(cube)
        return cubelist

    def _create_cubelist_from_boosting(
        self, optimised_coeffs, aux_coords_and_dims, attributes, predictor_names
    ):
        """Create a cubelist by combining the optimised coefficients and the
        appropriate metadata. A predictor_index coordinate and predictor_names
        coordinate are created for storing the location parameter (beta) and
        scale parameter (gamma) coefficients.

        Args:
            optimised_coeffs (numpy.ndarray)
            aux_coords_and_dims (list of tuples):
                List of tuples of the format [(coord, dim), (coord, dim)]
            attributes (dict):
                Attributes for an EMOS coefficients cube including
                "diagnostic standard name" and an updated title.
            predictor_names (list):
                Names of predictors used in Nonhomogeneous Boosting.

        Returns:
            cubelist (iris.cube.CubeList):
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate Nonhomogeneous boosting coefficient
                i.e. beta and gamma.
        """
        dim_coords_and_dims = [
            (iris.coords.DimCoord(list(range(len(predictor_names))),
                                  long_name="predictor_index"), 0)
        ]
        aux_coords_and_dims.extend(
            [(iris.coords.AuxCoord(predictor_names, long_name="predictor_name"), 0)]
        )
        cubelist = iris.cube.CubeList([])
        for optimised_coeff, coeff_name in zip(optimised_coeffs, self.coeff_names):
            cube = iris.cube.Cube(
                optimised_coeff,
                long_name=f"ngb_coefficient_{coeff_name}",
                units="1",
                dim_coords_and_dims=dim_coords_and_dims,
                aux_coords_and_dims=aux_coords_and_dims,
                attributes=attributes,
            )
            cubelist.append(cube)
        return cubelist

    def create_coefficients_cubelist(self, optimised_coeffs, historic_forecasts,
                                     predictor_names=None):
        """Create a cubelist for storing the coefficients computed using EMOS.

        .. See the documentation for examples of these cubes.
        .. include:: extended_documentation/calibration/
           ensemble_calibration/create_coefficients_cube.rst

        Args:
            optimised_coeffs (list or numpy.ndarray):
                Array or list of optimised coefficients.
                Order of coefficients is [alpha, beta, gamma, delta].
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.
            predictor_names (list):
                Names of predictors used in Nonhomogeneous Boosting.

        Returns:
            iris.cube.CubeList:
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.

        Raises:
            ValueError: If the number of coefficients in the optimised_coeffs
                does not match the expected number.
        """
        if self.predictor.lower() == "realizations":
            optimised_coeffs = [
                optimised_coeffs[0],
                optimised_coeffs[1:-2],
                optimised_coeffs[-2],
                optimised_coeffs[-1],
            ]

        if len(optimised_coeffs) != len(self.coeff_names):
            msg = (
                "The number of coefficients in {} must equal the "
                "number of coefficient names {}.".format(
                    optimised_coeffs, self.coeff_names
                )
            )
            raise ValueError(msg)

        aux_coords_and_dims = self._create_temporal_coordinates(historic_forecasts)
        aux_coords_and_dims.extend(self._create_spatial_coordinates(historic_forecasts))
        attributes = self._set_attributes(historic_forecasts)

        if self.boosting:
            print("aux_coords_and_dims = ", aux_coords_and_dims)
            return self._create_cubelist_from_boosting(
                optimised_coeffs, aux_coords_and_dims, attributes, predictor_names
            )
        else:
            return self._create_cubelist(
                optimised_coeffs, historic_forecasts, aux_coords_and_dims, attributes
            )

    def compute_initial_guess(
        self,
        truths,
        forecast_predictor,
        predictor,
        estimate_coefficients_from_linear_model_flag,
        number_of_realizations,
        sm=None,
    ):
        """
        Function to compute initial guess of the alpha, beta, gamma
        and delta components of the EMOS coefficients by linear regression
        of the forecast predictor and the truths, if requested. Otherwise,
        default values for the coefficients will be used.

        If the predictor is "mean", then the order of the initial_guess is
        [alpha, beta, gamma, delta]. Otherwise, if the predictor is
        "realizations" then the order of the initial_guess is
        [alpha, beta0, beta1, beta2, gamma, delta], where the number of beta
        variables will correspond to the number of realizations. In this
        example initial guess with three beta variables, there will
        correspondingly be three realizations.

        The default values for the initial guesses are in
        [alpha, beta, gamma, delta] ordering:

        * For the ensemble mean, the default initial guess: [0, 1, 0, 1]
          assumes that the raw forecast is skilful and the expected adjustments
          are small.

        * For the ensemble realizations, the default initial guess is
          effectively: [0, 1/3., 1/3., 1/3., 0, 1], such that
          each realization is assumed to have equal weight.

        If linear regression is enabled, the alpha and beta coefficients
        associated with the ensemble mean or ensemble realizations are
        modified based on the results from the linear regression fit.

        Args:
            truths (numpy.ndarray):
                Array containing the truth fields.
            forecast_predictor (numpy.ndarray):
                Array containing the fields to be used as the predictor,
                either the ensemble mean or the ensemble realizations.
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.
            estimate_coefficients_from_linear_model_flag (bool):
                Flag whether coefficients should be estimated from
                the linear regression, or static estimates should be used.
            number_of_realizations (int or None):
                Number of realizations within the forecast predictor. If no
                realizations are present, this option is None.
            sm (Optional[statsmodels.api]):
                Statsmodels instance.

        Returns:
            list of float:
                List of coefficients to be used as initial guess.
                Order of coefficients is [alpha, beta, gamma, delta].

        """
        if (
            predictor.lower() == "mean"
            and not estimate_coefficients_from_linear_model_flag
        ):
            initial_guess = [0, 1, 0, 1]
        elif predictor.lower() == "realizations" and (
            not estimate_coefficients_from_linear_model_flag or not sm
        ):
            initial_beta = np.repeat(
                np.sqrt(1.0 / number_of_realizations), number_of_realizations
            ).tolist()
            initial_guess = [0] + initial_beta + [0, 1]
        elif estimate_coefficients_from_linear_model_flag:
            truths_flattened = flatten_ignoring_masked_data(truths)
            if predictor.lower() == "mean":
                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor
                )
                if (truths_flattened.size == 0) or (
                    forecast_predictor_flattened.size == 0
                ):
                    gradient, intercept = [np.nan, np.nan]
                else:
                    gradient, intercept, _, _, _ = stats.linregress(
                        forecast_predictor_flattened, truths_flattened
                    )
                initial_guess = [intercept, gradient, 0, 1]
            elif predictor.lower() == "realizations":
                forecast_predictor_flattened = flatten_ignoring_masked_data(
                    forecast_predictor, preserve_leading_dimension=True
                )
                val = sm.add_constant(forecast_predictor_flattened.T)
                est = sm.OLS(truths_flattened, val).fit()
                intercept = est.params[0]
                gradient = est.params[1:]
                initial_guess = [intercept] + gradient.tolist() + [0, 1]

        return np.array(initial_guess, dtype=np.float32)

    @staticmethod
    def mask_cube(cube, landsea_mask):
        """
        Mask the input cube using the given landsea_mask. Sea points are
        filled with nans and masked.

        Args:
            cube (iris.cube.Cube):
                A cube to be masked, on the same grid as the landsea_mask.
                The last two dimensions on this cube must match the dimensions
                in the landsea_mask cube.
            landsea_mask(iris.cube.Cube):
                A cube containing a land-sea mask. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Raises:
            IndexError: if the cube and landsea_mask shapes are not compatible.
        """
        try:
            cube.data[..., ~landsea_mask.data.astype(np.bool)] = np.nan
        except IndexError as err:
            msg = "Cube and landsea_mask shapes are not compatible. {}".format(err)
            raise IndexError(msg)
        else:
            cube.data = np.ma.masked_invalid(cube.data)

    def _prepare_input_units(self, historic_forecasts, truths):
        # Make sure inputs have the same units.
        if self.desired_units:
            historic_forecasts.convert_units(self.desired_units)
            truths.convert_units(self.desired_units)

        if historic_forecasts.units != truths.units:
            msg = (
                "The historic forecast units of {} do not match "
                "the truths units {}. These units must match, so that "
                "the coefficients can be estimated."
            )
            raise ValueError(msg)

    def _prepare_inputs(
        self,
        historic_forecasts,
        truths,
        scale_parameter_predictor=iris.analysis.VARIANCE,
    ):

        historic_forecasts, truths = filter_non_matching_cubes(
            historic_forecasts, truths
        )
        check_forecast_consistency(historic_forecasts)

        number_of_realizations = None
        if self.predictor.lower() == "mean":
            forecast_predictor = collapsed(
                historic_forecasts, "realization", iris.analysis.MEAN
            )
        elif self.predictor.lower() == "realizations":
            forecast_predictor = historic_forecasts
            number_of_realizations = len(forecast_predictor.coord("realization").points)
            enforce_coordinate_ordering(forecast_predictor, "realization")

        forecast_var = collapsed(
            historic_forecasts, "realization", scale_parameter_predictor
        )
        return forecast_predictor, forecast_var, number_of_realizations

    def guess_and_minimise(
        self,
        truths,
        historic_forecasts,
        forecast_predictor,
        forecast_var,
        number_of_realizations,
    ):
        """Function to consolidate calls to compute the initial guess, compute
        the optimised coefficients using minimisation and store the resulting
        coefficients within a CubeList.

        Args:
            truths (iris.cube.Cube):
                Truths from the training dataset.
            historic_forecasts (iris.cube.Cube):
                Historic forecasts from the training dataset.
            forecast_predictor (iris.cube.Cube):
                Predictor of the forecast within the minimisation. This
                is either ensemble mean or the ensemble realizations.
            forecast_var (iris.cube.Cube):
                Variance of the forecast for use in the minimisation.
            number_of_realizations (int or None):
                Number of realizations within the forecast predictor. If no
                realizations are present, this option is None.

        Returns:
            iris.cube.CubeList:
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.

        Returns:
            iris.cube.CubeList:
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.

        """
        sm = self._get_statsmodels_availability()
        if self.each_point:
            index = [
                forecast_predictor.coord(axis="y"),
                forecast_predictor.coord(axis="x"),
            ]

            argument_list = (
                (
                    truths_slice.data,
                    fp_slice.data,
                    self.predictor,
                    self.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG,
                    number_of_realizations,
                    sm,
                )
                for (truths_slice, fp_slice) in zip(
                    truths.slices_over(index), forecast_predictor.slices_over(index)
                )
            )

            with Pool(os.cpu_count()) as pool:
                initial_guess = pool.starmap(self.compute_initial_guess, argument_list)

        else:
            if self.minimise_each_point:
                self.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = False

            # Computing initial guess for EMOS coefficients
            initial_guess = self.compute_initial_guess(
                truths.data,
                forecast_predictor.data,
                self.predictor,
                self.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG,
                number_of_realizations,
                sm=sm,
            )
            if np.any(np.isnan(initial_guess)):
                initial_guess = self.compute_initial_guess(
                    truths.data,
                    forecast_predictor.data,
                    self.predictor,
                    False,
                    number_of_realizations,
                    sm=sm,
                )

            if self.minimise_each_point:
                initial_guess = np.broadcast_to(
                    initial_guess,
                    (
                        len(truths.coord(axis="y").points)
                        * len(truths.coord(axis="x").points),
                        len(initial_guess),
                    ),
                )

        # Calculate coefficients if there are no nans in the initial guess.
        optimised_coeffs = self.minimiser(
            initial_guess,
            forecast_predictor,
            truths,
            forecast_var,
            self.predictor,
            self.distribution.lower(),
        )
        coefficients_cubelist = self.create_coefficients_cubelist(
            optimised_coeffs, historic_forecasts
        )

        return coefficients_cubelist

    def minimise_boosting(self, truths, historic_forecasts, forecast_predictors, forecast_vars):
        if self.each_point | self.minimise_each_point:
            index = [
                forecast_predictors[0].coord(axis="y"),
                forecast_predictors[0].coord(axis="x"),
            ]

            argument_list = []
            for fp, fv in zip(forecast_predictors, forecast_vars):
                for truth_slice, fp_slice, fv_slice in zip(truths.slices_over(index), fp.slices_over(index), fv.slices_over(index)):
                    argument_list.append([truth_slice, fp_slice, fv_slice])

            with Pool(os.cpu_count()) as pool:
                optimised_coeffs = pool.starmap(self.minimiser.process, argument_list)

        else:
            optimised_coeffs = self.minimiser.process(
                truths, forecast_predictors, forecast_vars,
            )

        predictor_names = ["intercept"] + [fp.name() for fp in forecast_predictors]
        coefficients_cubelist = self.create_coefficients_cubelist(
            optimised_coeffs, historic_forecasts[0], predictor_names=predictor_names
        )
        return coefficients_cubelist

    def process(self, historic_forecasts, truths, landsea_mask=None):
        """
        Using Nonhomogeneous Gaussian Regression/Ensemble Model Output
        Statistics, estimate the required coefficients from historical
        forecasts.

        The main contents of this method is:

        1. Check that the predictor is valid.
        2. Filter the historic forecasts and truths to ensure that these
           inputs match in validity time.
        3. Apply unit conversion to ensure that the historic forecasts and
           truths have the desired units for calibration.
        4. Calculate the variance of the historic forecasts. If the chosen
           predictor is the mean, also calculate the mean of the historic
           forecasts.
        5. If a land-sea mask is provided then mask out sea points in the truths
           and predictor from the historic forecasts.
        6. Calculate initial guess at coefficient values by performing a
           linear regression, if requested, otherwise default values are
           used.
        7. Perform minimisation.

        Args:
            historic_forecasts (iris.cube.Cube or iris.cube.CubeList):
                Either a cube containing historic forecasts from the training
                dataset or a cubelist containing a cube for each predictor.
            truths (iris.cube.Cube):
                Truths from the training dataset.
            landsea_mask (iris.cube.Cube):
                The optional cube containing a land-sea mask. If provided, only
                land points are used to calculate the coefficients. Within the
                land-sea mask cube land points should be specified as ones,
                and sea points as zeros.

        Returns:
            iris.cube.CubeList:
                CubeList constructed using the coefficients provided and using
                metadata from the historic_forecasts cube. Each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.

        Raises:
            ValueError: If either the historic_forecasts or truths cubes were not
                passed in.
            ValueError: If the units of the historic and truth cubes do not
                match.

        """
        if not (historic_forecasts and truths):
            raise ValueError("historic_forecasts and truths cubes must be provided.")

        # Ensure predictor is valid.
        check_predictor(self.predictor)
        print("historic_forecasts = ", historic_forecasts)
        print("truths = ", truths)
        if self.boosting:
            if isinstance(historic_forecasts, iris.cube.Cube):
                historic_forecasts = iris.cube.CubeList([historic_forecasts])

            for hf in historic_forecasts:
                if hf.name() == truths.name():
                    self._prepare_input_units(hf, truths)

            forecast_predictors = iris.cube.CubeList()
            forecast_vars = iris.cube.CubeList([])
            for hf in historic_forecasts:
                (
                    forecast_predictor,
                    forecast_var,
                    number_of_realizations,
                ) = self._prepare_inputs(
                    hf, truths, scale_parameter_predictor=iris.analysis.STD_DEV
                )
                forecast_predictors.append(forecast_predictor)
                forecast_vars.append(forecast_var)

            # If a landsea_mask is provided mask out the sea points
            if landsea_mask:
                [self.mask_cube(fp, landsea_mask) for fp in forecast_predictors]
                [self.mask_cube(fv, landsea_mask) for fv in forecast_vars]
                [self.mask_cube(t, landsea_mask) for t in truths]

            print("truths = ", truths)
            print("forecast_predictors = ", forecast_predictors)
            print("forecast_vars = ", forecast_vars)

            coefficients_cubelist = self.minimise_boosting(
                truths,
                historic_forecasts,
                forecast_predictors,
                forecast_vars,
            )
            print("coefficients_cubelist = ", coefficients_cubelist)
        else:
            self._prepare_inputs(historic_forecasts, truths)
            (
                forecast_predictor,
                forecast_var,
                number_of_realizations,
            ) = self._prepare_inputs(historic_forecasts, truths)

            # If a landsea_mask is provided mask out the sea points
            if landsea_mask:
                self.mask_cube(forecast_predictor, landsea_mask)
                self.mask_cube(forecast_var, landsea_mask)
                self.mask_cube(truths, landsea_mask)

            coefficients_cubelist = self.guess_and_minimise(
                truths,
                historic_forecasts,
                forecast_predictor,
                forecast_var,
                number_of_realizations,
            )

        return coefficients_cubelist


class CalibratedForecastDistributionParameters(BasePlugin):
    """
    Class to calculate calibrated forecast distribution parameters given an
    uncalibrated input forecast and EMOS coefficients.
    """

    def __init__(self, predictor="mean", boosting=False):
        """
        Create a plugin that uses the coefficients created using EMOS from
        historical forecasts and corresponding truths and applies these
        coefficients to the current forecast to generate location and scale
        parameters that represent the calibrated distribution at each point.

        Args:
            predictor (str):
                String to specify the form of the predictor used to calculate
                the location parameter when estimating the EMOS coefficients.
                Currently the ensemble mean ("mean") and the ensemble
                realizations ("realizations") are supported as the predictors.


        """
        check_predictor(predictor)
        self.predictor = predictor
        self.boosting = boosting

        self.coefficients_cubelist = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = "<CalibratedForecastDistributionParameters: predictor: {}>"
        return result.format(self.predictor)

    def _diagnostic_match(self, current_forecast):
        """Check that the forecast diagnostic matches the coefficients used to
        construct the coefficients.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Raises:
            ValueError: If the forecast diagnostic and coefficients cube
                diagnostic does not match.
        """
        for cube in self.coefficients_cubelist:
            diag = cube.attributes["diagnostic_standard_name"]
            if isinstance(current_forecast, iris.cube.CubeList):
                current_forecast_names = [cf.name() for cf in current_forecast]
            else:
                current_forecast_names = [current_forecast.name()]
            if diag not in current_forecast_names:
                msg = (
                    f"The available forecast diagnostics ({current_forecast_names}) "
                    "do not match the diagnostic used to construct the "
                    f"coefficients ({diag})"
                )
                raise ValueError(msg)

    def _spatial_domain_match(self, current_forecast):
        """
        Check that the domain of the current forecast and coefficients cube
        match.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Raises:
            ValueError: If the points or bounds of the specified axis of the
                current_forecast and coefficients_cube do not match.
        """
        msg = (
            "The points or bounds of the {} axis given by the current forecast {} "
            "do not match those given by the coefficients cube {}."
        )

        for axis in ["x", "y"]:
            for coeff_cube in self.coefficients_cubelist:
                if (
                    (
                        current_forecast.coord(axis=axis).collapsed().points
                        != coeff_cube.coord(axis=axis).collapsed().points
                    ).all()
                    or (
                        current_forecast.coord(axis=axis).collapsed().bounds
                        != coeff_cube.coord(axis=axis).collapsed().bounds
                    ).all()
                ):
                    raise ValueError(
                        msg.format(
                            axis,
                            current_forecast.coord(axis=axis).collapsed(),
                            coeff_cube.coord(axis=axis).collapsed(),
                        )
                    )

    def _calculate_location_parameter_from_boosting(self, current_forecast):
        """
        Function to calculate the location parameter when the ensemble mean at
        each grid point is the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Returns:
            numpy.ndarray:
                Location parameter calculated using the ensemble mean as the
                predictor.

        """
        location_parameter = 0
        for cf in current_forecast:
            forecast_predictor = collapsed(
                cf, "realization", iris.analysis.MEAN
            )
            constr = iris.Constraint(predictor_name=cf.name())
            location_parameter += (
                    self.coefficients_cubelist.extract(
                        "ngb_coefficient_beta").extract(constr)[0].data
                    * forecast_predictor.data).astype(np.float32)
        print("location_parameter = ", location_parameter)
        return location_parameter

    def _calculate_scale_parameter_from_boosting(self, current_forecast):
        """
        Calculation of the scale parameter using the ensemble variance
        adjusted using the gamma and delta coefficients calculated by NGB.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Returns:
            numpy.ndarray:
                Scale parameter for defining the distribution of the calibrated
                forecast.

        """
        scale_parameter = 0
        for cf in current_forecast:
            forecast_var = cf.collapsed(
                "realization", iris.analysis.VARIANCE
            )
            constr = iris.Constraint(predictor_name=cf.name())
            scale_parameter += (
                    self.coefficients_cubelist.extract(
                        "ngb_coefficient_gamma").extract(constr)[0].data
                    * forecast_var.data).astype(np.float32)
        print("scale_parameter = ", scale_parameter)
        return scale_parameter

    def _calculate_location_parameter_from_mean(self, current_forecast):
        """
        Function to calculate the location parameter when the ensemble mean at
        each grid point is the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Returns:
            numpy.ndarray:
                Location parameter calculated using the ensemble mean as the
                predictor.

        """
        forecast_predictor = collapsed(
            current_forecast, "realization", iris.analysis.MEAN
        )

        # Calculate location parameter = a + b*X, where X is the
        # raw ensemble mean. In this case, b = beta.
        location_parameter = (
            self.coefficients_cubelist.extract_strict("emos_coefficient_alpha").data
            + self.coefficients_cubelist.extract_strict("emos_coefficient_beta").data
            * forecast_predictor.data
        ).astype(np.float32)

        return location_parameter

    def _calculate_location_parameter_from_realizations(self, current_forecast):
        """
        Function to calculate the location parameter when the ensemble
        realizations are the predictor.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Returns:
            numpy.ndarray:
                Location parameter calculated using the ensemble realizations
                as the predictor.
        """
        forecast_predictor = current_forecast

        # Calculate location parameter = a + b1*X1 .... + bn*Xn, where X is the
        # ensemble realizations. The number of b and X terms depends upon the
        # number of ensemble realizations. In this case, b = beta^2.
        beta_values = np.array([], dtype=np.float32)
        beta_values = self.coefficients_cubelist.extract_strict(
            "emos_coefficient_beta"
        ).data.copy()
        a_and_b = np.append(
            self.coefficients_cubelist.extract_strict("emos_coefficient_alpha").data,
            beta_values ** 2,
        )
        forecast_predictor_flat = convert_cube_data_to_2d(forecast_predictor)
        xy_shape = next(forecast_predictor.slices_over("realization")).shape
        col_of_ones = np.ones(np.prod(xy_shape), dtype=np.float32)
        ones_and_predictor = np.column_stack((col_of_ones, forecast_predictor_flat))
        location_parameter = (
            np.dot(ones_and_predictor, a_and_b).reshape(xy_shape).astype(np.float32)
        )
        return location_parameter

    def _calculate_scale_parameter(self, current_forecast):
        """
        Calculation of the scale parameter using the ensemble variance
        adjusted using the gamma and delta coefficients calculated by EMOS.

        Further information is available in the :mod:`module level docstring \
<improver.calibration.ensemble_calibration>`.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.

        Returns:
            numpy.ndarray:
                Scale parameter for defining the distribution of the calibrated
                forecast.

        """
        forecast_var = current_forecast.collapsed(
            "realization", iris.analysis.VARIANCE
        )
        # Calculating the scale parameter, based on the raw variance S^2,
        # where predicted variance = c + dS^2, where c = (gamma)^2 and
        # d = (delta)^2
        scale_parameter = (
            self.coefficients_cubelist.extract_strict("emos_coefficient_gamma").data
            ** 2
            + self.coefficients_cubelist.extract_strict("emos_coefficient_delta").data
            ** 2
            * forecast_var.data
        ).astype(np.float32)
        return scale_parameter

    def _create_output_cubes(self, current_forecast, location_parameter, scale_parameter):
        """
        Creation of output cubes containing the location and scale parameters.

        Args:
            current_forecast (iris.cube.Cube):
                The cube containing the current forecast.
            location_parameter (numpy.ndarray):
                Location parameter of the calibrated distribution.
            scale_parameter (numpy.ndarray):
                Scale parameter of the calibrated distribution.

        Returns:
            (tuple): tuple containing:
                **location_parameter_cube** (iris.cube.Cube):
                    Location parameter of the calibrated distribution with
                    associated metadata.
                **scale_parameter_cube** (iris.cube.Cube):
                    Scale parameter of the calibrated distribution with
                    associated metadata.
        """
        template_cube = next(current_forecast.slices_over("realization"))
        template_cube.remove_coord("realization")

        location_parameter_cube = create_new_diagnostic_cube(
            "location_parameter",
            template_cube.units,
            template_cube,
            template_cube.attributes,
            data=location_parameter,
        )
        scale_parameter_cube = create_new_diagnostic_cube(
            "scale_parameter",
            f"({template_cube.units})^2",
            template_cube,
            template_cube.attributes,
            data=scale_parameter,
        )
        return location_parameter_cube, scale_parameter_cube

    def process(self, current_forecast, coefficients_cubelist, landsea_mask=None):
        """
        Apply the EMOS coefficients to the current forecast, in order to
        generate location and scale parameters for creating the calibrated
        distribution.

        Args:
            current_forecast (iris.cube.Cube or iris.cube.CubeList):
                The cube or cubelist containing the current forecast and
                any predictors, if using Nonhomogeneous Gaussian Boosting.
            coefficients_cubelist (iris.cube.CubeList):
                CubeList of EMOS coefficients where each cube within the
                cubelist is for a separate EMOS coefficient e.g. alpha, beta,
                gamma, delta.
            landsea_mask (iris.cube.Cube or None):
                The optional cube containing a land-sea mask. If provided sea
                points will be masked in the output cube.
                This cube needs to have land points set to 1 and
                sea points to 0.

        Returns:
            (tuple): tuple containing:
                **location_parameter_cube** (iris.cube.Cube):
                    Cube containing the location parameter of the calibrated
                    distribution calculated using either the ensemble mean or
                    the ensemble realizations. The location parameter
                    represents the point at which a resulting PDF would be
                    centred.
                **scale_parameter_cube** (iris.cube.Cube):
                    Cube containing the scale parameter of the calibrated
                    distribution calculated using either the ensemble mean or
                    the ensemble realizations. The scale parameter represents
                    the statistical dispersion of the resulting PDF, so a
                    larger scale parameter will result in a broader PDF.

        """
        self.coefficients_cubelist = coefficients_cubelist
        # Check coefficients_cube and forecast cube are compatible.

        if self.boosting:
            print(type(current_forecast))
            self._diagnostic_match(current_forecast)
            for cf in current_forecast:
                self._spatial_domain_match(cf)
                for coeff_cube in coefficients_cubelist:
                    forecast_coords_match(coeff_cube, cf)
            location_parameter = self._calculate_location_parameter_from_boosting(current_forecast)
            scale_parameter = self._calculate_scale_parameter_from_boosting(current_forecast)
        else:
            self._spatial_domain_match(current_forecast)
            self._diagnostic_match(current_forecast)
            for cube in coefficients_cubelist:
                forecast_coords_match(cube, current_forecast)
            if self.predictor.lower() == "mean":
                location_parameter = self._calculate_location_parameter_from_mean(current_forecast)
            else:
                location_parameter = self._calculate_location_parameter_from_realizations(current_forecast)

            scale_parameter = self._calculate_scale_parameter(current_forecast)

        location_parameter_cube, scale_parameter_cube = self._create_output_cubes(
            current_forecast[0], location_parameter, scale_parameter
        )

        # Use a mask to confine calibration to land regions by masking the
        # sea.
        if landsea_mask:
            # Calibration is applied to all grid points, but the areas
            # where a mask is valid is then masked out at the end. The cube
            # containing a land-sea mask has sea points defined as zeroes and
            # the land points as ones, so the mask needs to be flipped here.
            flip_mask = np.logical_not(landsea_mask.data)
            scale_parameter_cube.data = np.ma.masked_where(
                flip_mask, scale_parameter_cube.data
            )
            location_parameter_cube.data = np.ma.masked_where(
                flip_mask, location_parameter_cube.data
            )

        return location_parameter_cube, scale_parameter_cube


class ApplyEMOS(PostProcessingPlugin):
    """
    Class to calibrate an input forecast given EMOS coefficients
    """

    @staticmethod
    def _get_attribute(coefficients, attribute_name, optional=False):
        """Get the value for the requested attribute, ensuring that the
        attribute is present consistently across the cubes within the
        coefficients cubelist.

        Args:
            coefficients (iris.cube.CubeList):
                EMOS coefficients
            attribute_name (str):
                Name of expected attribute
            optional (bool):
                Indicate whether the attribute is allowed to be optional.

        Returns:
            None or Any:
                Returns None if the attribute is not present. Otherwise,
                the value of the attribute is returned.

        Raises:
            ValueError: If coefficients do not share the expected attributes.
        """
        attributes = [
            str(c.attributes[attribute_name])
            for c in coefficients
            if c.attributes.get(attribute_name) is not None
        ]

        if not attributes and optional:
            return None
        if not attributes and not optional:
            msg = (
                f"The {attribute_name} attribute must be specified on all "
                "coefficients cubes."
            )
            raise AttributeError(msg)

        if len(set(attributes)) == 1 and len(attributes) == len(coefficients):
            return coefficients[0].attributes[attribute_name]

        msg = (
            "Coefficients must share the same {0} attribute. "
            "{0} attributes provided: {1}".format(attribute_name, attributes)
        )
        raise AttributeError(msg)

    @staticmethod
    def _get_forecast_type(forecast):
        """Identifies whether the forecast is in probability, realization
        or percentile space

        Args:
            forecast (iris.cube.Cube)
        """
        try:
            find_percentile_coordinate(forecast)
        except CoordinateNotFoundError:
            if forecast.name().startswith("probability_of"):
                return "probabilities"
            return "realizations"
        return "percentiles"

    def _convert_to_realizations(self, forecast, realizations_count, ignore_ecc_bounds):
        """Convert an input forecast of probabilities or percentiles into
        pseudo-realizations

        Args:
            forecast (iris.cube.Cube)
            realizations_count (int):
                Number of pseudo-realizations to generate from the input
                forecast
            ignore_ecc_bounds (bool)
        """
        if not realizations_count:
            raise ValueError(
                "The 'realizations_count' argument must be defined "
                "for forecasts provided as {}".format(self.forecast_type)
            )

        if self.forecast_type == "probabilities":
            conversion_plugin = ConvertProbabilitiesToPercentiles(
                ecc_bounds_warning=ignore_ecc_bounds
            )
        if self.forecast_type == "percentiles":
            conversion_plugin = ResamplePercentiles(
                ecc_bounds_warning=ignore_ecc_bounds
            )

        forecast_as_percentiles = conversion_plugin(
            forecast, no_of_percentiles=realizations_count
        )
        forecast_as_realizations = RebadgePercentilesAsRealizations()(
            forecast_as_percentiles
        )

        return forecast_as_realizations

    def _calibrate_forecast(self, forecast, randomise, random_seed):
        """
        Generate calibrated probability, percentile or realization output

        Args:
            forecast (iris.cube.Cube):
                Uncalibrated input forecast
            randomise (bool):
                If True, order realization output randomly rather than using
                the input forecast.  If forecast type is not realizations, this
                is ignored.
            random_seed (int):
                For realizations input if randomise is True, random seed for
                generating re-ordered percentiles.  If randomise is False, the
                random seed may still be used for splitting ties.

        Returns:
            iris.cube.Cube:
                Calibrated forecast
        """
        if self.forecast_type == "probabilities":
            conversion_plugin = ConvertLocationAndScaleParametersToProbabilities(
                distribution=self.distribution["name"],
                shape_parameters=self.distribution["shape"],
            )
            result = conversion_plugin(
                self.distribution["location"], self.distribution["scale"], forecast
            )

        else:
            conversion_plugin = ConvertLocationAndScaleParametersToPercentiles(
                distribution=self.distribution["name"],
                shape_parameters=self.distribution["shape"],
            )

            if self.forecast_type == "percentiles":
                perc_coord = find_percentile_coordinate(forecast)
                result = conversion_plugin(
                    self.distribution["location"],
                    self.distribution["scale"],
                    forecast,
                    percentiles=perc_coord.points,
                )
            else:
                no_of_percentiles = len(forecast.coord("realization").points)
                percentiles = conversion_plugin(
                    self.distribution["location"],
                    self.distribution["scale"],
                    forecast,
                    no_of_percentiles=no_of_percentiles,
                )
                result = EnsembleReordering().process(
                    percentiles,
                    forecast,
                    random_ordering=randomise,
                    random_seed=random_seed,
                )

        return result

    def process(
        self,
        forecast,
        coefficients,
        land_sea_mask=None,
        realizations_count=None,
        ignore_ecc_bounds=True,
        predictor="mean",
        randomise=False,
        random_seed=None,
        boosting=False,
    ):
        """Calibrate input forecast using pre-calculated coefficients

        Args:
            forecast (iris.cube.CubeList):
                Uncalibrated forecast as probabilities, percentiles or
                realizations
            coefficients (iris.cube.CubeList):
                EMOS coefficients
            land_sea_mask (iris.cube.Cube or None):
                Land sea mask where a value of "1" represents land points and
                "0" represents sea.  If set, allows calibration of land points
                only.
            realizations_count (int or None):
                Number of realizations to use when generating the intermediate
                calibrated forecast from probability or percentile inputs
            ignore_ecc_bounds (bool):
                If True, allow percentiles from probabilities to exceed the ECC
                bounds range.  If input is not probabilities, this is ignored.
            predictor (str):
                Predictor to be used to calculate the location parameter of the
                calibrated distribution.  Value is "mean" or "realizations".
            randomise (bool):
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.
            random_seed (int or None):
                Used in generating calibrated realizations.  If input forecast
                is probabilities or percentiles, this is ignored.
            boosting (bool):
                If True, enable nonhomogeneous boosting following
                Messner et al., 2017 allowing multiple predictors to be provided.
                If False, coefficients are estimated using EMOS.

        Returns:
            iris.cube.Cube:
                Calibrated forecast in the form of the input (ie probabilities
                percentiles or realizations)
        """
        if boosting and isinstance(forecast, iris.cube.Cube):
            msg = ("A forecast cubelist must be supplied, if the coefficients have been "
                   "calculated using Nonhomogeneous Gaussian Boosting.")
            raise ValueError(msg)
        elif not boosting and isinstance(forecast, iris.cube.CubeList):
            msg = ("A forecast cube (not cubelist) must be supplied, if the coefficients have been "
                   "calculated using Ensemble Model Output Statistics.")
            raise ValueError(msg)

        if boosting:
            self.forecast_type = self._get_forecast_type(forecast[0])

            if self.forecast_type != "realizations":
                forecast_as_realizations = iris.cube.CubeList()
                for diag_cube in forecast:
                    forecast_as_realizations.append(self._convert_to_realizations(
                        diag_cube.copy(), realizations_count, ignore_ecc_bounds
                    ))
            else:
                forecast_as_realizations = iris.cube.CubeList()
                for diag_cube in forecast:
                    forecast_as_realizations.append(diag_cube.copy())
            # Identify diagnostic to be calibrated using an attribute.
            constr = iris.Constraint(coefficients[0].attributes["diagnostic_standard_name"])
            forecast = forecast.extract(constr)[0]
            calibration_plugin = CalibratedForecastDistributionParameters(
                predictor=predictor, boosting=boosting
            )
        else:
            self.forecast_type = self._get_forecast_type(forecast)

            forecast_as_realizations = forecast.copy()
            if self.forecast_type != "realizations":
                forecast_as_realizations = self._convert_to_realizations(
                    forecast.copy(), realizations_count, ignore_ecc_bounds
                )
            calibration_plugin = CalibratedForecastDistributionParameters(
                predictor=predictor
            )

        location_parameter, scale_parameter = calibration_plugin(
            forecast_as_realizations, coefficients, landsea_mask=land_sea_mask
        )

        self.distribution = {
            "name": self._get_attribute(coefficients, "distribution"),
            "location": location_parameter,
            "scale": scale_parameter,
            "shape": self._get_attribute(
                coefficients, "shape_parameters", optional=True
            ),
        }

        result = self._calibrate_forecast(forecast, randomise, random_seed)

        if land_sea_mask:
            # fill in masked sea points with uncalibrated data
            merge_land_and_sea(result, forecast)

        return result
