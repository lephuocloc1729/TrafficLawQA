"""Helpers for the compact public ``conditions`` fact contract.

The public representation is a map keyed by condition predicate, for example
``{"active_time": [{"from": "16:00", "to": "20:00"}]}``. Flat predicate
facts remain supported internally and are bundled at serialization boundaries.
"""

from __future__ import annotations

from typing import Any


CONDITION_PREDICATES = frozenset({
    "calendar_parity",
    "maximum_axle_load",
    "maximum_payload_mass",
    "maximum_speed",
    "maximum_vehicle_height",
    "maximum_vehicle_mass",
    "maximum_vehicle_width",
    "minimum_distance",
    "minimum_speed",
    "active_time",
})

CONDITION_FIELDS = (
    "value", "unit", "scope", "conditions", "status", "confidence",
    "fill_from", "operator", "applies_to",
)


def _as_condition_map(value: Any) -> dict[str, Any]:
    """Normalize the old typed-list and current keyed-map bundle shapes."""
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if str(key) in CONDITION_PREDICATES}
    result: dict[str, Any] = {}
    if not isinstance(value, list):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        predicate = item.get("type", item.get("predicate"))
        if predicate not in CONDITION_PREDICATES:
            continue
        payload = {key: child for key, child in item.items() if key not in {"type", "predicate"}}
        existing = result.get(predicate)
        if existing is None:
            result[predicate] = payload
        elif isinstance(existing, list):
            existing.append(payload)
        else:
            result[predicate] = [existing, payload]
    return result


def bundle_condition_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bundle flat condition facts under one map-valued ``conditions`` fact."""
    ordinary: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        predicate = fact.get("predicate")
        if predicate == "conditions":
            normalized = _as_condition_map(fact.get("value"))
            for key, raw in normalized.items():
                grouped.setdefault(key, []).extend(raw if isinstance(raw, list) else [raw])
        elif predicate in CONDITION_PREDICATES:
            grouped.setdefault(str(predicate), []).append({
                key: fact[key] for key in CONDITION_FIELDS if fact.get(key) is not None
            })
        else:
            ordinary.append(fact)

    if not grouped:
        return ordinary

    result: dict[str, Any] = {}
    for predicate, entries in grouped.items():
        # Repeated time windows are naturally a list of interval objects.
        if predicate == "active_time":
            windows: list[Any] = []
            for entry in entries:
                value = entry.get("value")
                values = value if isinstance(value, list) else [value]
                windows.extend(item for item in values if item is not None)
            windows = list({str(item): item for item in windows}.values())
            metadata = {key: value for key, value in entries[0].items() if key != "value"}
            result[predicate] = (
                windows if not metadata else {**metadata, "value": windows or None}
            )
            continue

        # Keep distinct applicability/threshold variants without losing data.
        unique: list[dict[str, Any]] = []
        for entry in entries:
            if entry not in unique:
                unique.append(entry)
        if len(unique) == 1:
            result[predicate] = unique[0]
            continue
        values = [entry.get("value") for entry in unique]
        if all(value == values[0] for value in values):
            # Merge metadata where possible; preserve explicit ambiguity/status.
            merged = dict(unique[0])
            for entry in unique[1:]:
                for key, value in entry.items():
                    merged.setdefault(key, value)
            result[predicate] = merged
        else:
            result[predicate] = unique

    ordinary.append({"predicate": "conditions", "value": result})
    return ordinary


def expand_condition_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand public condition maps for deterministic internal evaluation."""
    expanded: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        if fact.get("predicate") != "conditions":
            expanded.append(fact)
            continue
        for predicate, raw in _as_condition_map(fact.get("value")).items():
            entries = raw if isinstance(raw, list) else [raw]
            for entry in entries:
                if not isinstance(entry, dict):
                    entry = {"value": entry}
                expanded.append({"predicate": predicate, **entry})
    return expanded
