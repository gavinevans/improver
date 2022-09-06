# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
""" Tests for the hail_fraction_adjustment function."""

import numpy as np
import pytest
from iris.cube import Cube
from numpy.testing import assert_almost_equal

from improver.precipitation_type.hail_fraction_adjustment import (
    hail_fraction_adjustment,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


COMMON_ATTRS = {
    "source": "Unit test",
    "institution": "Met Office",
    "title": "Post-Processed IMPROVER unit test",
}


@pytest.fixture()
def precip_phase_prob(phase_name: str) -> Cube:
    """Create a precipitation phase probability at the surface cube."""
    coord_name = f"probability_of_falling_{phase_name}_level_above_surface"

    phase_prob = np.array(
        [
            [[1.0, 0.2, 1.0], [0.0, 0.8, 0.5], [0.0, 0.1, 0.3]],
            [[0.8, 0.1, 1.0], [0.0, 0.2, 0.5], [0.0, 0.1, 0.3]],
        ],
        dtype=np.float32,
    )
    phase_prob_cube = set_up_variable_cube(
        phase_prob,
        coord_name,
        "1",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )
    return phase_prob_cube


def hail_fraction() -> Cube:
    """Create a hail fraction cube."""
    hail_fraction_data = np.array(
        [
            [[0.5, 0.0, 1.0], [0.0, 1.0, 0.1], [0.0, 0.2, 0.9]],
            [[0.5, 0.1, 0.5], [0.0, 0.5, 0.1], [0.0, 0.2, 0.5]],
        ],
        dtype=np.float32,
    )
    hail_fraction_cube = set_up_variable_cube(
        hail_fraction_data,
        "hail_fraction",
        "1",
        spatial_grid="equalarea",
        attributes=COMMON_ATTRS,
        standard_grid_metadata="gl_ens",
    )
    return hail_fraction_cube


@pytest.mark.parametrize("phase_name", ("rain", "sleet", "snow"))
def test_adjust_precip_phase_prob(precip_phase_prob):
    """Test the result of adjusting the precipitation phase probability at the
    surface to account for the hail fraction."""

    expected_data = np.array(
        [
            [[0.5, 0.2, 0.0], [0.0, 0.0, 0.45], [0.0, 0.08, 0.03]],
            [[0.4, 0.09, 0.5], [0.0, 0.1, 0.45], [0.0, 0.08, 0.15]],
        ],
        dtype=np.float32,
    )

    result = hail_fraction_adjustment(precip_phase_prob, hail_fraction())
    assert_almost_equal(result.data, expected_data)
    assert result.name() == precip_phase_prob.name()
    assert result.attributes == precip_phase_prob.attributes
