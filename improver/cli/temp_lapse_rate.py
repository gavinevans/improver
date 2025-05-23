#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate temperature lapse rates for given temperature and
orogrophy datasets."""

from improver import cli
from improver.constants import DALR


@cli.clizefy
@cli.with_output
def process(
    temperature: cli.inputcube,
    orography: cli.inputcube = None,
    land_sea_mask: cli.inputcube = None,
    *,
    max_height_diff: float = 35,
    nbhood_radius: int = 7,
    max_lapse_rate: float = -3 * DALR,
    min_lapse_rate: float = DALR,
    dry_adiabatic=False,
    model_id_attr: str = None,
):
    """Calculate temperature lapse rates in units of K m-1 over orography grid.

    Args:
        temperature (iris.cube.Cube):
            Air temperature data. This is required even when returning DALR,
            as this defines the grid on which lapse rates are required.
        orography (iris.cube.Cube):
            Orography data.
        land_sea_mask (iris.cube.Cube):
            Binary land-sea mask data. True for land-points, False for sea.
        max_height_diff (float):
            Maximum allowable height difference between the central point and
            points in the neighbourhood over which the lapse rate will be
            calculated.
        nbhood_radius (int):
            Radius of neighbourhood in grid points around each point. The
            neighbourhood is a square array with side length
            2*nbhood_radius + 1. The default value of 7 is from the reference
            paper (see plugin documentation).
        max_lapse_rate (float):
            Maximum lapse rate allowed, in K m-1.
        min_lapse_rate (float):
            Minimum lapse rate allowed, in K m-1.
        dry_adiabatic (bool):
            If True, returns a cube containing the dry adiabatic lapse rate
            rather than calculating the true lapse rate.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending. This is inherited from the input temperature cube.

    Returns:
        iris.cube.Cube:
            Lapse rate (K m-1)

    Raises:
        ValueError: If minimum lapse rate is greater than maximum.
        ValueError: If Maximum height difference is less than zero.
        ValueError: If neighbourhood radius is less than zero.
        RuntimeError: If calculating the true lapse rate and orography or
                      land mask arguments are not given.
    """
    import numpy as np

    from improver.metadata.utilities import (
        create_new_diagnostic_cube,
        generate_mandatory_attributes,
    )
    from improver.temperature.lapse_rate import LapseRate

    if dry_adiabatic:
        attributes = generate_mandatory_attributes(
            [temperature], model_id_attr=model_id_attr
        )
        result = create_new_diagnostic_cube(
            "air_temperature_lapse_rate",
            "K m-1",
            temperature,
            attributes,
            data=np.full_like(temperature.data, DALR).astype(np.float32),
        )
        return result

    if min_lapse_rate > max_lapse_rate:
        msg = "Minimum lapse rate specified is greater than the maximum."
        raise ValueError(msg)

    if max_height_diff < 0:
        msg = "Maximum height difference specified is less than zero."
        raise ValueError(msg)

    if nbhood_radius < 0:
        msg = "Neighbourhood radius specified is less than zero."
        raise ValueError(msg)

    if orography is None or land_sea_mask is None:
        msg = "Missing orography and/or land mask arguments."
        raise RuntimeError(msg)

    result = LapseRate(
        max_height_diff=max_height_diff,
        nbhood_radius=nbhood_radius,
        max_lapse_rate=max_lapse_rate,
        min_lapse_rate=min_lapse_rate,
    )(temperature, orography, land_sea_mask, model_id_attr=model_id_attr)
    return result
