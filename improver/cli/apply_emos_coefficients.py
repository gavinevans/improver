#!/usr/bin/env python
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
"""Script to apply coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    realizations_count: int = None,
    randomise=False,
    random_seed: int = None,
    ignore_ecc_bounds=False,
    predictor="mean",
    land_sea_mask_name: str = None
):
    """Applying coefficients for Ensemble Model Output Statistics.

    Load in arguments for applying coefficients for Ensemble Model Output
    Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). The coefficients are applied to the forecast
    that is supplied, so as to calibrate the forecast. The calibrated
    forecast is written to a cube. If no coefficients are provided the input
    forecast is returned unchanged.

    Args:
        input_cubes (iris.cube.CubeList):
            A list of cubes containing:
            - A Cube containing the forecast to be calibrated. The input format
            could be either realizations, probabilities or percentiles.
            - A cubelist containing the coefficients used for calibration or None.
            If none then then input is returned unchanged.
            - Optionally, a cube containing the land-sea mask on the same domain
            as the forecast that is to be calibrated. Land points are
            specified by ones and sea points are specified by zeros.
            If not None this argument will enable land-only calibration, in
            which sea points are returned without the application of
            calibration.
            - Optionally, a cube containing a probability forecast that will be
            used as a template when generating probability output when the input
            format of the forecast cube is not probabilities i.e. realizations
            or percentiles.
        realizations_count (int):
            Option to specify the number of ensemble realizations that will be
            created from probabilities or percentiles when applying the EMOS
            coefficients.
        randomise (bool):
            Option to reorder the post-processed forecasts randomly. If not
            set, the ordering of the raw ensemble is used. This option is
            only valid when the input format is realizations.
        random_seed (int):
            Option to specify a value for the random seed for testing
            purposes, otherwise the default random seen behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the randomise option to order the input
            percentiles randomly, rather than use the ordering from the raw
            ensemble, or for splitting tied values within the raw ensemble,
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
        ignore_ecc_bounds (bool):
            If True, where the percentiles exceed the ECC bounds range,
            raises a warning rather than an exception. This occurs when the
            current forecasts is in the form of probabilities and is
            converted to percentiles, as part of converting the input
            probabilities into realizations.
        predictor (str):
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.
        land_sea_mask_name (str):
            Name of the land-sea mask cube. If supplied, a land-sea mask cube
            is expected within the list of input cubes and this land-sea mask
            will be used to calibrate land points only.

    Returns:
        iris.cube.Cube:
            The calibrated forecast cube.

    Raises:
        ValueError:
            If the current forecast is a coefficients cube.
        ValueError:
            If the coefficients cube does not have the right name of
            "emos_coefficients".
        ValueError:
            If the forecast type is 'percentiles' or 'probabilities' and the
            realizations_count argument is not provided.
    """
    import warnings

    from improver.calibration import split_forecasts_and_coeffs
    from improver.calibration.ensemble_calibration import ApplyEMOS

    (forecast, coefficients, land_sea_mask, prob_template) = split_forecasts_and_coeffs(
        cubes, land_sea_mask_name
    )

    if coefficients is None:
        msg = (
            "There are no coefficients provided for calibration. The "
            "uncalibrated forecast will be returned."
        )
        warnings.warn(msg)
        return forecast

    calibration_plugin = ApplyEMOS()
    result = calibration_plugin(
        forecast,
        coefficients,
        land_sea_mask=land_sea_mask,
        prob_template=prob_template,
        realizations_count=realizations_count,
        ignore_ecc_bounds=ignore_ecc_bounds,
        predictor=predictor,
        randomise=randomise,
        random_seed=random_seed,
    )

    return result
