import json
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


def _is_individual_test(name: str) -> bool:
    name_lower = name.lower()
    if "solution" in name_lower and "telephone" not in name_lower and "phone" not in name_lower:
        return False
    return True


def _enrich(item: dict) -> dict:
    parts = [item.get("name", "")]
    if item.get("description"):
        parts.append(item["description"])
    keys = item.get("keys") or item.get("test_type") or []
    if isinstance(keys, list):
        parts.extend(keys)
    job_levels = item.get("job_levels") or []
    if isinstance(job_levels, list):
        parts.extend(job_levels)
    languages = item.get("languages") or []
    if isinstance(languages, list):
        parts.extend(languages)
    item["_search_text"] = " | ".join(str(p) for p in parts if p)
    item["test_type"] = keys if isinstance(keys, list) else []
    item["url"] = item.get("url") or item.get("link", "")
    item["name"] = item.get("name", "")
    item["description"] = item.get("description") or ""
    return item


def load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return []
    with open(CATALOG_PATH) as f:
        items = json.load(f)
    filtered = [item for item in items if _is_individual_test(item.get("name", ""))]
    return [_enrich(item) for item in filtered]
