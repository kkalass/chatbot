# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for settings-to-config mapping helpers."""

from src.ingest.build_from_settings import (
    build_ingestion_config,
)
from src.shared.settings import Settings


class TestConfigMappers:
    def test_build_ingestion_config_maps_expected_fields(self) -> None:
        settings = Settings(
            split_length=180,
            split_overlap=15,
            ingest_file_batch_size=4,
        )

        config = build_ingestion_config(settings)

        assert config.split_length == 180
        assert config.split_overlap == 15
        assert config.batch_size == 4
