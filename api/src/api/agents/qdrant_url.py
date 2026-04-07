"""Resolve Qdrant HTTP URL for host vs Docker Compose."""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse


def resolve_qdrant_url() -> str:
    """
    Compose service name `qdrant` only resolves on the Docker network.
    For local `uv run` / notebooks on the host, map it to localhost.

    When QDRANT_URL is unset: use qdrant:6333 inside a container (same as the old
    hardcoded default), and localhost:6333 on the host — otherwise Docker API would
    point at itself, not the qdrant service.
    """
    in_docker = os.path.exists("/.dockerenv")
    raw = (os.getenv("QDRANT_URL") or "").strip()
    if not raw:
        raw = "http://qdrant:6333" if in_docker else "http://localhost:6333"
    parsed = urlparse(raw)
    if parsed.hostname == "qdrant" and not in_docker:
        port = parsed.port
        netloc = f"localhost:{port}" if port else "localhost"
        raw = urlunparse(parsed._replace(netloc=netloc))
    return raw
