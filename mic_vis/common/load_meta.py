"""Metadata loaders for Bluesky-produced files."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import yaml


DEFAULT_PLAN_ARGS_PATH = "entry/instrument/bluesky/metadata/plan_args"


def load_bluesky_meta(path: str | Path) -> Mapping[str, Any]:
    """Load Bluesky ``plan_args`` metadata from an HDF5 file."""

    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        node: Any = handle
        for part in DEFAULT_PLAN_ARGS_PATH.split("/"):
            node = node[part]

        if isinstance(node, h5py.Group):
            return {
                key: _decode_h5_value(value[()])
                for key, value in node.items()
                if isinstance(value, h5py.Dataset)
            }
        if isinstance(node, h5py.Dataset):
            decoded = _decode_h5_value(node[()])
            if isinstance(decoded, Mapping):
                return decoded
            if isinstance(decoded, str):
                parsed = _parse_text_mapping(decoded)
                if isinstance(parsed, Mapping):
                    return parsed
            raise ValueError(f"{file_path.name} plan_args is not a mapping")
        raise ValueError(f"{file_path.name} does not contain a readable plan_args node")


def _decode_h5_value(value: Any) -> Any:
    if hasattr(value, "shape") and getattr(value, "shape", None) == () and hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:
                continue
            if isinstance(parsed, Mapping):
                return parsed
        return stripped
    return value


def _parse_text_mapping(value: str) -> Mapping[str, Any] | None:
    stripped = value.strip()
    if not stripped:
        return {}
    for parser in (yaml.safe_load, json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
        except Exception:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    return None
