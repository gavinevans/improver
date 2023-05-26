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
"""Estimate the the appropriate rescaling based on the difference in altitude between
the grid point and the site."""

from typing import Dict, Optional

import iris
import numpy as np
from numpy.polynomial.polynomial import Polynomial as polyfit
from numpy.polynomial import Polynomial as poly1d
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin


class EstimateDzRescaling(PostProcessingPlugin):

    """Estimate a rescaling of the input forecasts based on the difference in
    altitude between the grid point and the site."""

    def __init__(self, dz_lower_bound=None, dz_upper_bound=None, polyfit_deg=None):
        """_summary_

        Args:
            dz_lower_bound: The lowest acceptable value for the difference in
                altitude between the grid point and the site. Sites with a lower
                (or more negative) difference in altitude will be excluded.
                Defaults to None.
            dz_upper_bound: The highest acceptable value for the difference in
                altitude between the grid point and the site. Sites with a larger
                positive difference in altitude will be excluded. Defaults to None.
            polyfit_deg: The degree chosen for the fitting polynomial. Please see
                numpy.polynomial.polynomial.Polynomial.fit for further information.
                Defaults to None.
        """
        self.dz_lower_bound = dz_lower_bound
        self.dz_upper_bound = dz_upper_bound
        self.polyfit_deg = polyfit_deg

    def _fit_polynomial(self, forecasts, truths, dz):
        data_filter = (
            (truths.data > 0.0)
            & (dz > self.dz_lower_bound)
            & (dz < self.dz_upper_bound)
        )

        log_error_ratio = np.log(forecasts[data_filter] / truths[data_filter])

        scale_factor = poly1d(
            polyfit.fit(dz[data_filter], log_error_ratio, self.polyfit_deg)
        )

        if self.polyfit_deg > 0:
            scale_factor[0] = 0.0
        return scale_factor

    def _compute_scaled_dz(self, scale_factor, dz):
        scaled_dz = dz.copy()
        scaled_dz.data = np.exp(-1.0 * scale_factor * dz.data)

        scaled_dz_lower = np.exp(-1.0 * scale_factor * self.dz_lower_bound)
        scaled_dz_upper = np.exp(-1.0 * scale_factor * self.dz_upper_bound)

        scaled_dz[(scaled_dz.data < scaled_dz_lower)] = scaled_dz_lower
        scaled_dz[(scaled_dz.data > scaled_dz_upper)] = scaled_dz_upper

        return scaled_dz

    def process(self, forecasts, truths, dz):

        constr = iris.Constraint(percentile=50.0)
        forecasts = forecasts.extract(constr)

        scale_factor = self._fit_polynomial(forecasts, truths)
        return self._compute_scaled_dz(scale_factor, dz)
