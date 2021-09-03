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

import unittest

import numpy as np
import pandas as pd

from improver.ensemble_calibration.ensemble_calibration import (
    estimate_coefficients_from_regimes)
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import (SetupCubes,
                             EnsembleCalibrationAssertions)
from improver.tests.ensemble_calibration.ensemble_calibration. \
    test_EstimateCoefficientsForEnsembleCalibration import (
        SetupExpectedCoefficients)
from improver.utilities.warnings_handler import ManageWarnings


IGNORED_MESSAGES = [
    "Collapsing a non-contiguous coordinate",  # Originating from Iris
    "The statsmodels can not be imported",
    "invalid escape sequence",  # Originating from statsmodels
    "can't resolve package from",  # Originating from statsmodels
    "Minimisation did not result in convergence",  # From calibration code
    "The final iteration resulted in",  # From calibration code
]
WARNING_TYPES = [
    UserWarning,
    ImportWarning,
    DeprecationWarning,
    ImportWarning,
    UserWarning,
    UserWarning,
]


class TestEstimateCoefficientsFromRegimes(
        SetupCubes, EnsembleCalibrationAssertions, SetupExpectedCoefficients):

    """Test the process method"""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up multiple cubes for testing."""
        super().setUp()
        self.current_cycle = "20171110T0000Z"
        self.distribution = "gaussian"

        self.coeff_names = ["gamma", "delta", "alpha", "beta"]
        self.coeff_names_realizations = (
            ['gamma', 'delta', 'alpha', 'beta0', 'beta1', 'beta2'])

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_single_regime(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution. In this case,
        a linear least-squares regression is used to construct the initial
        guess."""
        desired_units = "K"
        predictor_of_mean_flag = "mean"
        max_iterations = 1000
        data = np.array([[10, 11, 2017, 12, 1],
                         [11, 11, 2017, 12, 1],
                         [12, 11, 2017, 12, 1],
                         [13, 11, 2017, 12, 1],
                         [14, 11, 2017, 12, 1]])
        input_df = pd.DataFrame(data, index=range(5),
                                columns=["day", "month", "year",
                                         "hour", "T0"])
        result = estimate_coefficients_from_regimes(
            self.distribution, self.current_cycle, desired_units,
            predictor_of_mean_flag, max_iterations,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube, None, input_df)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_gaussian)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)


if __name__ == '__main__':
    unittest.main()
