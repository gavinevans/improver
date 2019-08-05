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
"""
This module defines the Accumulation class for calculating precipitation
accumulations from advected radar fields. It is also possible to create longer
accumulations from shorter intervals.
"""
import warnings
from datetime import timedelta
import numpy as np

import iris
from improver.utilities.cube_manipulation import expand_bounds
from improver.utilities.temporal import iris_time_to_datetime
from improver.utilities.cube_units import (enforce_coordinate_units_and_dtypes,
                                           enforce_diagnostic_units_and_dtypes)


class Accumulation:
    """
    Class to calculate precipitation accumulations from radar rates fields
    provided at discrete time intervals. The plugin will calculate
    accumulations between each pair of rates fields provided. These will be
    used to construct the accumulation period requested when possible, and
    cubes of this desired period are then returned.
    """
    def __init__(self, accumulation_units='m', accumulation_period=None,
                 forecast_periods=None):
        """
        Initialise the plugin.

        Kwargs:
            accumulation_units (str):
                The physical units in which the accumulation should be
                returned. The default is metres.
            accumulation_period (int):
                The desired accumulation period in seconds. This period
                must be evenly divisible by the time intervals of the input
                cubes. The default is None, in which case an accumulation is
                calculated across the span of time covered by the input rates
                cubes.
            forecast_periods (list):
                The forecast periods in seconds that define the end of an
                accumulation period.

        """
        self.accumulation_units = accumulation_units
        self.accumulation_period = accumulation_period
        self.forecast_periods = forecast_periods

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<Accumulation: accumulation_units={}, '
                  'accumulation_period={}s, '
                  'forecast_periods={}s>')
        return result.format(self.accumulation_units, self.accumulation_period,
                             self.forecast_periods)

    @staticmethod
    def sort_cubes_by_time(cubes):
        """
        Sort cubes in time ascending order (from earliest time to latest time).

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation rate cubes.
        Returns:
            tuple: tuple containing:
                **cubes** (iris.cube.CubeList):
                    The cubelist in ascending time order.
                **times** (list):
                    A list of the validity times of the precipitation rate
                    cubes in integer seconds since 1970-01-01 00:00:00.
        """
        times = np.array([cube.coord('time').points[0] for cube in cubes])
        time_sorted = np.argsort(times)
        times = times[time_sorted]
        cubes = iris.cube.CubeList(np.array(cubes)[time_sorted])
        return cubes, times

    def _check_inputs(self, cubes):
        """Check the inputs prior to calculating the accumulations.

        Args:
            cubes: iris.cube.CubeList
                Cube list of precipitation rates that will be checked for their
                appropriateness in calculating the requested accumulations.
                The timesteps between the cubes in this cubelist are
                expected to be regular.

        Returns:
            (tuple): tuple containing

                **cubes**: iris.cube.CubeList
                    Modified version of the input cube list of precipitation
                    rates that have had the units of the coordinates and
                    cube data enforced. The cube list has also been sorted by
                    time.

                **time_interval** (float):
                    Interval between the timesteps from the input cubelist.

                **integral** (int):
                    Integer value based on how many times the accumulation
                    period is divisible by the time interval. This therefore
                    indicates how many timesteps are expected within an
                    accumulation period.

        Raises:
            ValueError: The input rates cubes must be at regularly spaced
                time intervals.
            ValueError: The specified accumulation period is not cleanly
                divisible by the time interval.

        """
        # Standardise inputs to expected units.
        cubes = enforce_coordinate_units_and_dtypes(
            cubes, ['time', 'forecast_reference_time', 'forecast_period'],
            inplace=False)

        enforce_diagnostic_units_and_dtypes(cubes)

        # Sort cubes into time order and calculate intervals.
        cubes, times = self.sort_cubes_by_time(cubes)

        try:
            time_interval, = np.unique(np.diff(times, axis=0))
        except ValueError:
            msg = ("Accumulation is designed to work with "
                   "rates cubes at regular time intervals. Cubes "
                   "provided are unevenly spaced in time; time intervals are "
                   "{}.".format(np.diff(times, axis=0)))
            raise ValueError(msg)

        if self.accumulation_period is None:
            # If no accumulation period is specified, assume that the input
            # cubes will be used to construct a single accumulation period.
            self.accumulation_period, = (
                cubes[-1].coord("forecast_period").points)

        if self.forecast_periods is None:
            self.forecast_periods = [
                cube.coord("forecast_period").points for cube in cubes]

        fraction, integral = np.modf(self.accumulation_period /
                                     time_interval)

        if fraction != 0:
            msg = ("The specified accumulation period ({}) is not divisible "
                   "by the time intervals between rates cubes ({}). As "
                   "a result it is not possible to calculate the desired "
                   "total accumulation period.".format(
                       self.accumulation_period, time_interval))
            raise ValueError(msg)
        return cubes, time_interval, integral

    def _get_cube_subsets(self, cubes, cube, integral):
        """Finding the subset of cubes from the input cubelist that are
        within the accumulation period, based on the input cube defining the
        upper bound of the accumulation period and the length of the
        accumulation period.

        Args:
            cubes (iris.cube.CubeList):
                Cubelist containing all the rates cubes that will be used
                to calculate the accumulation.
            cube (iris.cube.Cube):
                Cube defining the upper bound of the accumulation period.
            integral (int)
                Integer value based on how many times the accumulation
                period is divisible by the time interval. This therefore
                indicates how many timesteps are expected within an
                accumulation period.

        Returns:
            cube_subset (iris.cube.Cubelist) or None:
                Cubelist that defines the cubes used to calculate
                the accumulations.

        Warns:
            Warning: The provided cubes result in a partial period given the
                specified accumulation_period, i.e. the number of cubes is
                insufficient to give a set of complete periods. Only complete
                periods will be returned.
        """
        fp_point = cube.coord("forecast_period").points[0]
        if fp_point not in self.forecast_periods:
            return None
        end_point, = iris_time_to_datetime(cube.coord("time"))
        start_point = (
            end_point - timedelta(seconds=int(self.accumulation_period)))
        constr = iris.Constraint(
            time=lambda cell: start_point <= cell.point <= end_point)

        cube_subset = cubes.extract(constr)

        if len(cube_subset) != int(integral + 1):
            msg = (
                "The provided cubes result in a partial period given the "
                "specified accumulation_period, i.e. the number of cubes "
                "is insufficient to give a set of complete periods. Only "
                "complete periods will be returned.")
            warnings.warn(msg)
            return None
        return cube_subset

    @staticmethod
    def _calculate_accumulation(cube_subset, time_interval):
        """Calculate the accumulation for the requested accumulation period
        by finding the subset of cubes from the input cubelist that are
        within the accumulation period, based on the input cube defining the
        upper bound of the accumulation period and the length of the
        accumulation period.

        Args:
            cube_subset (iris.cube.CubeList):
                Cubelist containing all the rates cubes that will be used
                to calculate the accumulation.
            time_interval (float):
                Interval between the timesteps from the input cubelist.

        Returns:
            accumulation (numpy.ndarray) or None:
                If either the forecast period given by the input cube is not
                a requested forecast_period at which to calculate the
                accumulations, or the number of input cubelist is only
                sufficient to partially cover the desired accumulation, then
                None is returned.
                If an accumulation can be successfully computed, then a
                numpy array is returned.

        """
        accumulation = 0.
        # Accumulations are calculated using the mean precipitation rate
        # calculated from the rates cubes that bookend the desired period.
        iterator = zip(cube_subset[0:-1], cube_subset[1:])
        for start_cube, end_cube in iterator:
            accumulation += ((start_cube.data + end_cube.data) *
                             time_interval * 0.5)
        return accumulation

    @staticmethod
    def _set_metadata(cube_subset):
        """Set the metadata on the accumulation cube. This includes
        expanding the bounds to cover the accumulation period with the
        point within the time and forecast_period coordinates recorded as the
        upper bound of the accumulation period.

        Args:
            cube_subset(iris.cube.CubeList):
                Cubelist containing the subset of cubes used to calculate
                the accumulations. The bounds from these cubes will be used
                to set the metadata on the output accumulation cube.

        Returns:
            accumulation_cube (iris.cube.Cube):
                Accumulation cube with the desired metadata.

        """
        cube_name = 'lwe_thickness_of_precipitation_amount'
        accumulation_cube = expand_bounds(
            cube_subset[0].copy(),
            iris.cube.CubeList(cube_subset),
            expanded_coords={'time': 'upper', 'forecast_period': 'upper'})
        accumulation_cube.rename(cube_name)
        accumulation_cube.units = 'm'
        return accumulation_cube

    def process(self, cubes):
        """
        Calculate period precipitation accumulations based upon precipitation
        rate fields. All calculations are performed in SI units, so
        precipitation rates are converted to "m/s" and times into seconds
        before calculations are performed. The output units of accumulation
        are set by the plugin keyword argument accumulation_units.

        Args:
            cubes (iris.cube.CubeList):
                A cubelist containing input precipitation rate cubes.
        Returns:
            accumulation_cubes (iris.cube.CubeList):
                A cubelist containing precipitation accumulation cubes where
                the accumulation periods are determined by plugin argument
                accumulation_period.
        """
        cubes, time_interval, integral = self._check_inputs(cubes)

        accumulation_cubes = iris.cube.CubeList()
        for cube in cubes:
            cube_subset = self._get_cube_subsets(cubes, cube, integral)
            if cube_subset is None:
                continue
            accumulation = self._calculate_accumulation(
                cube_subset, time_interval)
            accumulation_cube = self._set_metadata(cube_subset)

            # Calculate new data and insert into cube.
            accumulation_cube.data = accumulation
            accumulation_cube.convert_units(self.accumulation_units)
            accumulation_cubes.append(accumulation_cube)

        return accumulation_cubes
