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
"""Unit tests for the
   weighted_blend.TriangularWeightedBlendAcrossAdjacentPoints plugin."""

import unittest

from cf_units import Unit

import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError

import numpy as np

from improver.blending.blend_across_adjacent_points import \
    TriangularWeightedBlendAcrossAdjacentPoints
from improver.tests.blending.weights.helper_functions import (
    cubes_for_triangular_weighted_blend_tests)
from improver.utilities.warnings_handler import ManageWarnings


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        forecast_period = 1
        result = str(TriangularWeightedBlendAcrossAdjacentPoints(
            'time', forecast_period, 'hours', width, 'weighted_mean'))
        msg = ('<TriangularWeightedBlendAcrossAdjacentPoints:'
               ' coord = time, central_point = 1.00, '
               'parameter_units = hours, width = 3.00, mode = weighted_mean>')
        self.assertEqual(result, msg)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        forecast_period = 1
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'time', forecast_period, 'hours', width, 'weighted_mean')
        expected_coord = "time"
        expected_width = 3.0
        expected_parameter_units = "hours"
        self.assertEqual(plugin.coord, expected_coord)
        self.assertEqual(plugin.width, expected_width)
        self.assertEqual(plugin.parameter_units, expected_parameter_units)

    def test_raises_expression(self):
        """Test that the __init__ raises the right error"""
        message = ("weighting_mode: no_mode is not recognised, "
                   "must be either weighted_maximum or weighted_mean")
        with self.assertRaisesRegex(ValueError, message):
            TriangularWeightedBlendAcrossAdjacentPoints(
                'time', 1, 'hours', 3.0, 'no_mode')


class Test__find_central_point(IrisTest):
    """Test the _find_central_point."""

    def setUp(self):
        """Set up a test cube."""
        self.cube, self.central_cube, self.forecast_period = (
            cubes_for_triangular_weighted_blend_tests())
        self.width = 1.0

    def test_central_point_available(self):
        """Test that the central point is available within the input cube."""
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', self.width,
            'weighted_mean')
        central_cube = plugin._find_central_point(self.cube)
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         central_cube.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'),
                         central_cube.coord('time'))
        self.assertArrayEqual(self.central_cube.data, central_cube.data)

    def test_central_point_not_available(self):
        """Test that the central point is not available within the
           input cube."""
        forecast_period = 2
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', forecast_period, 'hours', self.width,
            'weighted_mean')
        msg = "The central point of"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._find_central_point(self.cube)


class Test_process(IrisTest):
    """Test the process method."""

    def setUp(self):
        """Set up a test cube."""
        self.cube, self.central_cube, self.forecast_period = (
            cubes_for_triangular_weighted_blend_tests())

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_1(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending."""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        result = plugin.process(self.cube)
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayEqual(self.central_cube.data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_2(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending."""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        result = plugin.process(self.cube)
        expected_data = np.array([[1.333333, 1.333333],
                                  [1.333333, 1.333333]])
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_1_max_mode(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending. This time
           use the weighted_maximum mode"""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_maximum')
        result = plugin.process(self.cube)
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayEqual(self.central_cube.data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_2_max_mode(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending. This time
           use the weighted_maximum mode"""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_maximum')
        result = plugin.process(self.cube)
        expected_data = np.array([[0.6666666, 0.6666666],
                                  [0.6666666, 0.6666666]])
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_central_point_not_in_allowed_range(self):
        """Test that an exception is generated when the central cube is not
           within the allowed range."""
        width = 1.0
        forecast_period = 2
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', forecast_period, 'hours', width,
            'weighted_mean')
        msg = "The central point of"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_alternative_parameter_units(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 7200 seconds. """
        forecast_period = 0
        width = 7200.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', forecast_period, 'seconds', width,
            'weighted_mean')
        result = plugin.process(self.cube)
        expected_data = np.array([[1.333333, 1.333333],
                                  [1.333333, 1.333333]])
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)


if __name__ == '__main__':
    unittest.main()
