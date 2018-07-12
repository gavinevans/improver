# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the weights.ChooseDefaultWeightsTriangular plugin."""


import unittest

from iris.coords import AuxCoord
from iris.tests import IrisTest
import numpy as np
import cf_units

from improver.blending.weights import ChooseDefaultWeightsTriangular
from improver.tests.blending.weights.test_WeightsUtilities import set_up_cube


class Test___repr__(IrisTest):
    """Tests for the __repr__function"""

    def test_basic(self):
        """Test the repr function formats the arguments correctly"""
        triangular_weights_instance = ChooseDefaultWeightsTriangular(
            units='hours')
        result = str(triangular_weights_instance)
        expected = ("<ChooseDefaultTriangularWeights "
                    "parameters_units=hours>")
        self.assertEqual(result, expected)

    def test_basic_no_units(self):
        """Test the repr function formats the arguments correctly"""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        result = str(triangular_weights_instance)
        expected = ("<ChooseDefaultTriangularWeights "
                    "parameters_units=no_unit>")
        self.assertEqual(result, expected)


class Test_triangular_weights(IrisTest):
    """Tests for the triangular_weights function"""

    def test_basic(self):
        """Test that the function returns a numpy array.
           Also check that the length of the weights is correct and they add
           up to 1.0"""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(15)
        midpoint = 5
        width = 3
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(coord_vals))
        self.assertEqual(weights.sum(), 1.0)

    def test_basic_weights(self):
        """Test that the function returns the correct triangular weights in a
           simple case"""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(15)
        midpoint = 5
        width = 3
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        expected_weights = np.array([0., 0., 0.,
                                     0.11111111, 0.22222222, 0.33333333,
                                     0.22222222, 0.11111111, 0.,
                                     0., 0., 0.,
                                     0., 0., 0.])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_midpoint_at_edge(self):
        """Test that the correct triangular weights are returned for a case
           where the midpoint is close to the end of the input coordinate.
           In this case the triangle is cut off at the end of the coordinate"""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(15)
        midpoint = 1
        width = 3
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        expected_weights = np.array([0.25, 0.375, 0.25,
                                     0.125, 0., 0.,
                                     0., 0., 0.,
                                     0., 0., 0.,
                                     0., 0., 0.])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_large_width(self):
        """Test the case where the width of the triangle is larger than the
           coordinate input.
           In this case all the weights are non-zero but still form the
           shape of a triangle."""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(15)
        midpoint = 5
        width = 10
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        expected_weights = np.array([0.055556, 0.066667, 0.077778,
                                     0.088889, 0.1, 0.111111,
                                     0.1, 0.088889, 0.077778,
                                     0.066667, 0.055556, 0.044444,
                                     0.033333, 0.022222, 0.011111])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_non_integer_midpoint(self):
        """Test the case where the midpoint of the triangle is not a point in
           the input coordinate.
           In this case we do not sample the peak of the triangle."""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(15)
        midpoint = 3.5
        width = 2
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        expected_weights = np.array([0., 0., 0.125,
                                     0.375, 0.375, 0.125,
                                     0., 0., 0.,
                                     0., 0., 0.,
                                     0., 0., 0.])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_non_integer_width(self):
        """Test when the width of the triangle does not fall on a grid point.
           This only affects the slope of the triangle slightly."""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(15)
        midpoint = 5
        width = 3.5
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        expected_weights = np.array([0., 0., 0.04,
                                     0.12, 0.2, 0.28,
                                     0.2, 0.12, 0.04,
                                     0., 0., 0.,
                                     0., 0., 0.])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_unevenly_spaced_coord(self):
        """Test the case where the input coordinate is not equally spaced.
           This represents the case where the data changes to 3 hourly. In this
           case the weights are assigned according to the value in the
           coordinate."""
        triangular_weights_instance = ChooseDefaultWeightsTriangular()
        coord_vals = np.arange(10)
        coord_vals = np.append(coord_vals, [12, 15, 18, 21, 24])
        midpoint = 8
        width = 5
        weights = triangular_weights_instance.triangular_weights(coord_vals,
                                                                 midpoint,
                                                                 width)
        expected_weights = np.array([0., 0., 0.,
                                     0., 0.05, 0.1,
                                     0.15, 0.2, 0.25,
                                     0.2, 0.05, 0.,
                                     0., 0., 0.])
        self.assertArrayAlmostEqual(weights, expected_weights)


class Test___init__(IrisTest):
    """Tests for the __init__ method in ChooseDefaultWeightsTriangular class"""

    def test_cf_unit_input(self):
        """Test the case where an instance of cf_units.Unit is passed in"""
        units = cf_units.Unit("hour")
        weights_instance = ChooseDefaultWeightsTriangular(units=units)
        expected_unit = units
        self.assertEqual(weights_instance.parameters_units, expected_unit)

    def test_string_input(self):
        """Test the case where a string is passed and gets converted to a
           cf_units.Unit instance"""
        units = "hour"
        weights_instance = ChooseDefaultWeightsTriangular(units=units)
        expected_unit = cf_units.Unit("hour")
        self.assertEqual(weights_instance.parameters_units, expected_unit)


class Test_process(IrisTest):
    """Tests for the process method in ChooseDefaultWeightsTriangular."""

    def setUp(self):
        """Set up cubes used in unit tests"""
        self.cube = set_up_cube()
        self.cube.add_aux_coord(AuxCoord(np.arange(2), 'forecast_period',
                                         units='hours'), 0)
        self.coord_name = "forecast_period"
        self.units = cf_units.Unit("hours")

    def test_same_units(self):
        """Test plugin produces the correct weights when the parameters for
           the triangle are in the same units as the input cube's coordinate"""
        width = 2
        weights_instance = ChooseDefaultWeightsTriangular(units=self.units)
        midpoint = 1
        weights = weights_instance.process(self.cube,
                                           self.coord_name,
                                           midpoint, width)
        expected_weights = np.array([0.33333333, 0.66666667])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_different_units(self):
        """"Test plugin produces the correct weights when the parameters for
            the triangle are in different units to the input cube's
            coordinate"""
        width = 7200
        weights_instance = ChooseDefaultWeightsTriangular(units="seconds")
        midpoint = 1
        weights = weights_instance.process(self.cube,
                                           self.coord_name,
                                           midpoint, width)
        expected_weights = np.array([0.33333333, 0.66666667])
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_unconvertable_units(self):
        """"Test plugin produces the correct weights when the parameters for
            the triangle cannot be converted to the same units as the
            coordinate"""
        width = 7200
        weights_instance = ChooseDefaultWeightsTriangular(units="m")
        midpoint = 3600
        message = r"Unable to convert from 'Unit\('m'\)' to 'Unit\('hours'\)'"
        with self.assertRaisesRegex(ValueError, message):
            weights_instance.process(
                self.cube, self.coord_name, midpoint, width)


if __name__ == '__main__':
    unittest.main()
