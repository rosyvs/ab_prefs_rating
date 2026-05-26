from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ab_prefs_demo.data_model import PreferenceRecord


def initialize_store(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"responses": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_records(path: Path) -> list[dict]:
    initialize_store(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    responses = payload.get("responses")
    if not isinstance(responses, list):
        raise ValueError(f"Expected 'responses' list in {path}")
    return responses


def append_record(path: Path, record: PreferenceRecord) -> None:
    initialize_store(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    responses = payload.get("responses")
    if not isinstance(responses, list):
        raise ValueError(f"Expected 'responses' list in {path}")
    responses.append(asdict(record))
    payload["responses"] = responses
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
