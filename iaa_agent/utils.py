from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def hour_bucket(hour: int, size: int = 3) -> int:
    return max(0, min(23, int(hour))) // size


def time_bucket_label(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    if 22 <= hour or hour < 2:
        return "night"
    return "late_night"


def minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def circular_minute_diff(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 1440 - diff)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in scores.values() if v > 0)
    if total <= 0:
        n = len(scores) or 1
        return {k: 1.0 / n for k in scores}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    n = len(values)
    return -sum((count / n) * math.log(count / n, 2) for count in counts.values())


def category_family(category: str) -> str:
    text = (category or "").lower()
    rules = [
        ("food", ["restaurant", "diner", "food", "cafe", "coffee", "bakery", "pizza", "deli", "burger", "sandwich", "dessert", "ice cream", "breakfast"]),
        ("nightlife", ["bar", "pub", "club", "nightlife", "brewery", "lounge", "wine"]),
        ("transport", ["subway", "train", "station", "airport", "bus", "road", "bridge", "parking"]),
        ("home", ["home", "residential", "apartment"]),
        ("work", ["office", "coworking", "conference"]),
        ("outdoor", ["park", "outdoors", "beach", "plaza", "garden", "trail"]),
        ("shop", ["shop", "store", "mall", "boutique", "market"]),
        ("arts", ["museum", "theater", "cinema", "arts", "gallery", "music", "stadium", "historic"]),
        ("health", ["gym", "fitness", "medical", "hospital", "doctor", "pharmacy"]),
        ("education", ["college", "school", "university", "academic", "library"]),
    ]
    for family, needles in rules:
        if any(needle in text for needle in needles):
            return family
    return text.strip() or "unknown"


def cosine_dict(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return safe_div(dot, na * nb)


def grid_cell(lat: float, lon: float, precision: int = 2) -> str:
    return f"{round(float(lat), precision)}:{round(float(lon), precision)}"


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

