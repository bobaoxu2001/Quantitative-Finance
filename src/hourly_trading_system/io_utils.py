"""Filesystem helpers for crash-safe state persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """
    Write ``text`` to ``path`` atomically.

    Content is written to a temporary file in the same directory, flushed and
    fsynced, then moved into place with ``os.replace``. Concurrent readers (for
    example a monitoring dashboard polling a state file) therefore observe
    either the previous or the new content -- never a truncated file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    """Serialise ``payload`` as JSON and write it atomically."""
    atomic_write_text(path, json.dumps(payload, indent=indent, default=str))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
