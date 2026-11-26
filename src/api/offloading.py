from __future__ import annotations

import json
import os
import time
from urllib.parse import urlparse

import requests


class TrustedNodeRouter:
    """Small bounded client for explicitly configured trusted inference peers."""

    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout_seconds = timeout_seconds
        try:
            entries = json.loads(os.getenv("SARA_TRUSTED_NODES", "[]"))
        except json.JSONDecodeError as exc:
            raise ValueError("SARA_TRUSTED_NODES must be valid JSON") from exc
        if not isinstance(entries, list):
            raise ValueError("SARA_TRUSTED_NODES must be a JSON list")
        self.nodes = []
        seen = set()
        for item in entries:
            if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key] for key in ("id", "url", "token")):
                raise ValueError("Each trusted node requires non-empty id, url, and token fields")
            if item["id"] in seen:
                raise ValueError("Trusted node IDs must be unique")
            parsed = urlparse(item["url"])
            if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}):
                raise ValueError("Trusted inference peers must use HTTPS; plain HTTP is allowed only for localhost")
            self.nodes.append(item)
            seen.add(item["id"])

    @property
    def available(self) -> bool:
        return bool(self.nodes)

    def status(self) -> dict:
        return {"available": self.available, "trusted_nodes": [{"node_id": node["id"], "endpoint": node["url"], "transport": urlparse(node["url"]).scheme} for node in self.nodes], "timeout_seconds": self.timeout_seconds}

    def infer(self, payload: dict) -> tuple[dict, dict]:
        failures = []
        for node in self.nodes:
            start = time.perf_counter()
            try:
                response = requests.post(
                    node["url"].rstrip("/") + "/api/internal/inference",
                    json=payload,
                    headers={"Authorization": f"Bearer {node['token']}"},
                    timeout=(0.5, self.timeout_seconds),
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict) or not isinstance(result.get("prediction"), str):
                    raise ValueError("Peer returned an invalid inference result")
                return result, {"success": True, "node_id": node["id"], "latency_ms": (time.perf_counter() - start) * 1000}
            except (requests.RequestException, ValueError) as exc:
                failures.append({"node_id": node["id"], "error": f"{type(exc).__name__}: {exc}"})
        raise RuntimeError(f"All trusted inference peers failed: {failures}")
