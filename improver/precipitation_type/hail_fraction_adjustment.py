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
"""Adjust the precipitation phase probability using hail fraction."""

from iris.cube import Cube


def hail_fraction_adjustment(precip_phase_prob: Cube, hail_fraction: Cube) -> Cube:
    """Adjust the precipitation phase probability at the surface by accounting for
    the presence of hail using the hail fraction provided.

    Adjusted precip phase prob = precip phase prob * (1 - hail fraction)

    Args:
        precip_phase_prob (iris.cube.Cube):
            Precipitation phase probability i.e. the probability of rain, sleet or
            snow occurring at the surface.
        hail_fraction (iris.cube.Cube):
            Fraction of precipitation that is predicted to fall as hail.

    Returns:
        Cube of the precipitation phase probability at the surface after accounting
        for hail fraction.

    """
    adjusted_precip_phase_prob = precip_phase_prob * (1 - hail_fraction)
    adjusted_precip_phase_prob.rename(precip_phase_prob.name())
    return adjusted_precip_phase_prob
