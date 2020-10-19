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
Unit tests for the
`ensemble_calibration.Boosting`
class.

"""
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import Boosting
from improver.calibration.utilities import convert_cube_data_to_2d
from improver.utilities.warnings_handler import ManageWarnings

from .helper_functions import EnsembleCalibrationAssertions, SetupCubes

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

def :
    data = np.array([])
    set_up_variable_cube()

# class SetupNormalInputs(SetupCubes):
#
#     """Create a class for setting up cubes for testing."""
#
#     @ManageWarnings(
#         ignored_messages=["Collapsing a non-contiguous coordinate."],
#         warning_types=[UserWarning],
#     )
#     def setUp(self):
#         """Set up expected inputs."""
#         super().setUp()
#         # Set up cubes and associated data arrays for temperature.
#         self.forecast_predictor_mean = self.historic_temperature_forecast_cube.collapsed(
#             "realization", iris.analysis.MEAN
#         )
#         self.forecast_predictor_realizations = (
#             self.historic_temperature_forecast_cube.copy()
#         )
#         self.forecast_variance = self.historic_temperature_forecast_cube.collapsed(
#             "realization", iris.analysis.VARIANCE
#         )
#         self.truth = self.historic_temperature_forecast_cube.collapsed(
#             "realization", iris.analysis.MAX
#         )
#         self.forecast_predictor_data = self.forecast_predictor_mean.data.flatten().astype(
#             np.float64
#         )
#         self.forecast_predictor_data_realizations = convert_cube_data_to_2d(
#             self.historic_temperature_forecast_cube.copy()
#         ).astype(np.float64)
#         self.forecast_variance_data = self.forecast_variance.data.flatten().astype(
#             np.float64
#         )
#         self.truth_data = self.truth.data.flatten().astype(np.float64)

#
# class Test_process(EnsembleCalibrationAssertions):
#
#     def setUp(self):
#         """Set up expected output."""
#         super().setUp()
#         self.tolerance = 1e-4
#         self.plugin = Plugin(tolerance=self.tolerance)
#         self.expected_mean_coefficients = [0.3965, 0.958, 0.0459, 0.6047]
#         self.expected_realizations_coefficients = [
#             0.2692,
#             0.0126,
#             0.5965,
#             0.7952,
#             0.0265,
#             0.2175,
#         ]
#
#     def

class Test_process():
