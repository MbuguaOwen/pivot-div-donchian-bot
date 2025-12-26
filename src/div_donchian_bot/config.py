from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
import re

_ENV_PATTERN = re.compile(r"\$\{([^:}]+)(?::([^}]*))?\}")

def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        def repl(m):
            key = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.getenv(key, default)
        return _ENV_PATTERN.sub(repl, value)
    return value

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(path: Path) -> Dict[str, Any]:
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _expand_env(d)

def load_config(config_path: str) -> Dict[str, Any]:
    base_path = Path(config_path)
    cfg = load_yaml(base_path)
    # record config directory for downstream relative file resolution (pair overrides, etc.)
    cfg["_config_dir"] = str(base_path.parent)
    return cfg

def load_pair_override(config_dir: Path, overrides_dir: str, symbol: str) -> Dict[str, Any]:
    path = Path(config_dir) / overrides_dir / f"{symbol}.yaml"
    if path.exists():
        return load_yaml(path)
    return {}
