#!/usr/bin/env bats
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


@test "weighted-blending invalid method" {
  # Run blending with linear and nonlinear: check it fails.
  run improver weighted-blending --wts_calc_method 'linear nonlinear' 'time' \
      "$IMPROVER_ACC_TEST_DIR/weighted_blending/basic_lin/multiple_probabilities_rain_*H.nc" \
      "NO_OUTPUT_FILE"
  [[ "${status}" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver weighted-blending [-h] [--profile]
                                  [--profile_file PROFILE_FILE]
                                  [--wts_calc_method WEIGHTS_CALCULATION_METHOD]
                                  [--cycletime CYCLETIME]
                                  [--model_id_attr MODEL_ID_ATTR]
                                  [--spatial_weights_from_mask]
                                  [--fuzzy_length FUZZY_LENGTH]
                                  [--y0val LINEAR_STARTING_POINT]
                                  [--ynval LINEAR_END_POINT]
                                  [--cval NON_LINEAR_FACTOR]
                                  [--wts_dict WEIGHTS_DICTIONARY]
                                  [--weighting_coord WEIGHTING_COORD]
                                  COORDINATE_TO_AVERAGE_OVER INPUT_FILES
                                  [INPUT_FILES ...] OUTPUT_FILE
improver weighted-blending: error: argument --wts_calc_method: invalid choice: 'linear nonlinear' (choose from 'linear', 'nonlinear', 'dict')

__TEXT__
  [[ "$output" =~ "$expected" ]]
}
