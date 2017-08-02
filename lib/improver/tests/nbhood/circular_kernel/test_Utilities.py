# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the nbhood.circular_kernel.Utilities plugin."""


import unittest

from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.nbhood.circular_kernel import Utilities


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(Utilities())
        msg = '<Utilities>'
        self.assertEqual(str(result), msg)


class Test_circular_kernel(IrisTest):

    """Test neighbourhood processing plugin on the OS National Grid."""

    def test_basic(self):
        """Test that the plugin returns a Numpy array."""
        ranges = (2, 2)
        fullranges = (0, 2, 2)
        weighted_mode = False
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertIsInstance(result, np.ndarray)

    def test_single_point_weighted(self):
        """Test behaviour for a unitary range, with weighting."""
        ranges = (1, 1)
        fullranges = (1, 1)
        weighted_mode = True
        expected = [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayAlmostEqual(result, expected)

    def test_single_point_unweighted(self):
        """Test behaviour for a unitary range without weighting.

        Note that this gives one more grid cell range than weighted! As the
        affected area is one grid cell more in each direction, an equivalent
        range of 2 was chosen for this test.
        """
        ranges = (1, 1)
        fullranges = (1, 1)
        weighted_mode = False
        expected = [[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayEqual(result, expected)

    def test_range5_weighted(self):
        """Test behaviour for a range of 5, with weighting."""
        ranges = (5, 5)
        fullranges = (5, 5)
        weighted_mode = True
        expected = [
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.2,  0.32, 0.36, 0.32, 0.2,  0.,   0.,   0.],
            [0.,   0.,   0.28, 0.48, 0.6,  0.64, 0.6,  0.48, 0.28, 0.,   0.],
            [0.,   0.2,  0.48, 0.68, 0.8,  0.84, 0.8,  0.68, 0.48, 0.2,  0.],
            [0.,   0.32, 0.6,  0.8,  0.92, 0.96, 0.92, 0.8,  0.6,  0.32, 0.],
            [0.,   0.36, 0.64, 0.84, 0.96, 1.,   0.96, 0.84, 0.64, 0.36, 0.],
            [0.,   0.32, 0.6,  0.8,  0.92, 0.96, 0.92, 0.8,  0.6,  0.32, 0.],
            [0.,   0.2,  0.48, 0.68, 0.8,  0.84, 0.8,  0.68, 0.48, 0.2,  0.],
            [0.,   0.,   0.28, 0.48, 0.6,  0.64, 0.6,  0.48, 0.28, 0.,   0.],
            [0.,   0.,   0.,   0.2,  0.32, 0.36, 0.32, 0.2,  0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayAlmostEqual(result, expected)

    def test_range5_unweighted(self):
        """Test behaviour for a range of 5 without weighting."""
        ranges = (5, 5)
        fullranges = (5, 5)
        weighted_mode = False
        expected = [[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                    [0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayEqual(result, expected)

    def test_single_point_weighted_extra_dims(self):
        """Test behaviour for a unitary range, with weighting.
        And added dimensions."""
        ranges = (1, 1)
        fullranges = (0, 0, 1, 1)
        weighted_mode = True
        expected = [[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayAlmostEqual(result, expected)

    def test_single_point_unweighted_extra_dims(self):
        """Test behaviour for a unitary range without weighting.
        And added dimensions."""
        ranges = (1, 1)
        fullranges = (0, 0, 1, 1)
        weighted_mode = False
        expected = [[[[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayEqual(result, expected)

    def test_range5_weighted_extra_dims(self):
        """Test behaviour for a range of 5, with weighting."""
        ranges = (5, 5)
        fullranges = (0, 0, 5, 5)
        weighted_mode = True
        expected = [[[
            [0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,  0.,   0.,   0.2,  0.32, 0.36, 0.32, 0.2,  0.,   0.,   0.],
            [0.,  0.,   0.28, 0.48, 0.6,  0.64, 0.6,  0.48, 0.28, 0.,   0.],
            [0.,  0.2,  0.48, 0.68, 0.8,  0.84, 0.8,  0.68, 0.48, 0.2,  0.],
            [0.,  0.32, 0.6,  0.8,  0.92, 0.96, 0.92, 0.8,  0.6,  0.32, 0.],
            [0.,  0.36, 0.64, 0.84, 0.96, 1.,   0.96, 0.84, 0.64, 0.36, 0.],
            [0.,  0.32, 0.6,  0.8,  0.92, 0.96, 0.92, 0.8,  0.6,  0.32, 0.],
            [0.,  0.2,  0.48, 0.68, 0.8,  0.84, 0.8,  0.68, 0.48, 0.2,  0.],
            [0.,  0.,   0.28, 0.48, 0.6,  0.64, 0.6,  0.48, 0.28, 0.,   0.],
            [0.,  0.,   0.,   0.2,  0.32, 0.36, 0.32, 0.2,  0.,   0.,   0.],
            [0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayAlmostEqual(result, expected)

    def test_range5_unweighted_extra_dims(self):
        """Test behaviour for a range of 5 without weighting."""
        ranges = (5, 5)
        fullranges = (0, 0, 5, 5)
        weighted_mode = False
        expected = [[[[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]]]
        result = (
            Utilities.circular_kernel(fullranges, ranges, weighted_mode))
        self.assertArrayEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
