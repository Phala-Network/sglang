"""Low dependency parsing for HiCache storage backend configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


class StorageBackendConfigError(ValueError):
    """Raised when a storage backend extra config cannot be loaded safely."""


def _load_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    if suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
            import tomli as tomllib

        with path.open("rb") as config_file:
            return tomllib.load(config_file)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - image dependency
            raise StorageBackendConfigError(
                "YAML storage backend config requires PyYAML"
            ) from exc
        with path.open("r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
    raise StorageBackendConfigError("Unsupported storage backend config format")


def load_storage_backend_extra_config(
    storage_backend_extra_config: Optional[str],
) -> dict[str, Any]:
    """Load inline or ``@file`` config and require a mapping root.

    The caller owns extraction of component-specific options such as prefetch
    thresholds. Errors intentionally omit config values and file contents.
    """
    if not storage_backend_extra_config:
        return {}

    try:
        if storage_backend_extra_config.startswith("@"):
            config_path = storage_backend_extra_config[1:]
            if not config_path:
                raise StorageBackendConfigError(
                    "Storage backend config file path is empty"
                )
            payload = _load_file(Path(config_path))
        else:
            payload = json.loads(storage_backend_extra_config)
    except StorageBackendConfigError:
        raise
    except Exception as exc:
        raise StorageBackendConfigError(
            f"Unable to load storage backend config ({type(exc).__name__})"
        ) from None

    if not isinstance(payload, Mapping):
        raise StorageBackendConfigError(
            "Storage backend config must contain a mapping object"
        )
    return dict(payload)
