# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for regression model artifact and backend registry contracts."""

import pytest

from improver.calibration.regression_model import (
    ARTIFACT_SCHEMA_VERSION,
    RegressionModelArtifact,
    get_backend_spec,
)


def test_regression_model_artifact_round_trip_dict_payload():
    """Test that an artifact can be reconstructed from a dictionary payload."""
    artifact = RegressionModelArtifact.create(
        method="qrf",
        model={"fitted": True},
        target_name="air_temperature",
        feature_config={"air_temperature": ["mean", "std"]},
        transformation="log",
        pre_transform_addition=1,
        unique_site_id_keys=["wmo_id"],
    )

    payload = artifact.__dict__.copy()
    result = RegressionModelArtifact.from_payload(payload)

    assert isinstance(result, RegressionModelArtifact)
    assert result.schema_version == ARTIFACT_SCHEMA_VERSION
    assert result.method == "qrf"
    assert result.target_name == "air_temperature"
    assert result.feature_config == {"air_temperature": ["mean", "std"]}
    assert result.transformation == "log"
    assert result.pre_transform_addition == 1
    assert result.unique_site_id_keys == ["wmo_id"]


def test_regression_model_artifact_missing_required_field():
    """Test that missing required fields in payload raise a clear error."""
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "method": "qrf",
        "model": object(),
        "target_name": "air_temperature",
        "feature_config": {"air_temperature": ["mean"]},
        "transformation": None,
        # pre_transform_addition missing
        "unique_site_id_keys": ["wmo_id"],
    }

    with pytest.raises(ValueError, match="missing required fields"):
        RegressionModelArtifact.from_payload(payload)


def test_regression_model_artifact_unsupported_schema_version():
    """Test that unsupported schema versions are rejected."""
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION + 1,
        "method": "qrf",
        "model": object(),
        "target_name": "air_temperature",
        "feature_config": {"air_temperature": ["mean"]},
        "transformation": None,
        "pre_transform_addition": 0,
        "unique_site_id_keys": ["wmo_id"],
    }

    with pytest.raises(ValueError, match="Unsupported model artifact schema version"):
        RegressionModelArtifact.from_payload(payload)


def test_regression_model_artifact_string_site_id_normalized():
    """Test that a single string site identifier is normalized to a list."""
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "method": "qrf",
        "model": object(),
        "target_name": "air_temperature",
        "feature_config": {"air_temperature": ["mean"]},
        "transformation": None,
        "pre_transform_addition": 0,
        "unique_site_id_keys": "wmo_id",
    }

    result = RegressionModelArtifact.from_payload(payload)
    assert result.unique_site_id_keys == ["wmo_id"]


def test_get_backend_spec_unsupported_method():
    """Test unsupported regression methods fail with clear message."""
    with pytest.raises(ValueError, match="Unsupported regression method 'catboost'"):
        get_backend_spec("catboost")
