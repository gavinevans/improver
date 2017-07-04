#!/usr/bin/env bats
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

@test "ecc -h" {
  run improver ecc -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-ecc [-h] [--no_of_percentiles NUMBER_OF_PERCENTILES]
                    [--sampling_method [PERCENTILE_SAMPLING_METHOD]]
                    (--reordering | --rebadging)
                    [--raw_forecast_filepath RAW_FORECAST_FILE]
                    [--random_ordering]
                    [--fixed_random_seed FIXED_RANDOM_SEED]
                    [--member_numbers MEMBER_NUMBERS]
                    INPUT_FILE OUTPUT_FILE

Apply Ensemble Copula Coupling to a file with one cube.

positional arguments:
  INPUT_FILE            A path to an input NetCDF file to be processed
  OUTPUT_FILE           The output path for the processed NetCDF

optional arguments:
  -h, --help            show this help message and exit
  --no_of_percentiles NUMBER_OF_PERCENTILES
                        The number of percentiles to be generated.
  --sampling_method [PERCENTILE_SAMPLING_METHOD]
                        Method to be used for generating the list of
                        percentiles with forecasts generated at each
                        percentile. "quantile" is the default option.
  --reordering          The option used to create ensemble members from
                        percentiles by reordering the input percentiles based
                        on the order of the raw ensemble forecast.
  --rebadging           The option used to create ensemble members from
                        percentiles by rebadging the input percentiles.

Reordering options:
  Options for reordering the input percentiles using the raw ensemble
  forecast as required to create ensemble members.

  --raw_forecast_filepath RAW_FORECAST_FILE
                        A path to an raw forecast NetCDF file to be processed.
  --random_ordering     Decide whether or not to use random ordering within
                        the ensemble reordering step.
  --fixed_random_seed FIXED_RANDOM_SEED
                        Decide whether or not a fixed random seed is used to
                        create the random numbers, either used for the
                        random_ordering or for splitting tied values.

Rebadging options:
  Options for rebadging the input percentiles as ensemble members.

  --member_numbers MEMBER_NUMBERS
                        A list of ensemble member numbers to use when
                        rebadging the percentiles into members.
__HELP__
  [[ "$output" == "$expected" ]]
}