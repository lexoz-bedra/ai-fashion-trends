from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "llm_cache"

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gemma": {
        "api_base": os.environ.get("LLM_API_BASE", "http://localhost:11434/v1"),
        "model_id": os.environ.get("LLM_MODEL_ID", "gemma3:12b"),
        "api_key": os.environ.get("LLM_API_KEY", "ollama"),
        "max_tokens": 1024,
        "temperature": 0.3,
    },
}


def register_model(name: str, config: dict[str, Any]) -> None:
    _MODEL_REGISTRY[name] = config


def call_model(
    prompt: str,
    model_name: str = "gemma",
    use_cache: bool = True,
    cache_dir: Path | None = None,
    **kwargs: Any,
) -> str:
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in registry. Available: {list(_MODEL_REGISTRY.keys())}")

    config = {**_MODEL_REGISTRY[model_name], **kwargs}

    cache = _LlmCache(cache_dir or _DEFAULT_CACHE_DIR) if use_cache else None
    cache_key = _cache_key(prompt, config["model_id"])

    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("LLM cache hit for key %s", cache_key[:16])
            return cached

    result = _call_openai_compatible(prompt, config)

    if cache:
        cache.put(cache_key, prompt, config["model_id"], result)

    return result


def _call_openai_compatible(prompt: str, config: dict[str, Any]) -> str:
    import requests

    url = config["api_base"].rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('api_key', '')}",
    }
    payload = {
        "model": config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.get("max_tokens", 1024),
        "temperature": config.get("temperature", 0.3),
    }

    logger.info("Calling LLM: %s at %s", config["model_id"], config["api_base"])
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]


class _LlmCache:

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "cache.jsonl"
        self._index: dict[str, str] | None = None

    def get(self, key: str) -> str | None:
        idx = self._load_index()
        return idx.get(key)

    def put(self, key: str, prompt: str, model_id: str, response: str) -> None:
        idx = self._load_index()
        if key in idx:
            return
        entry = {"key": key, "model": model_id, "prompt": prompt[:200], "response": response}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        idx[key] = response

    def _load_index(self) -> dict[str, str]:
        if self._index is not None:
            return self._index
        self._index = {}
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self._index[obj["key"]] = obj["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        return self._index


def _cache_key(prompt: str, model_id: str) -> str:
    raw = f"{model_id}|{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()
