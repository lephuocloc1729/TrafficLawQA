"""Geometry-based links between detected main and supplementary signs."""

from __future__ import annotations

from src.core.symbolic.fact_schema import SignObservation, SignRelation


def link_supplementary_signs(
    observations: list[SignObservation],
) -> list[SignRelation]:
    """Link each panel to the closest horizontally aligned main sign above it."""
    main_signs = [
        item for item in observations
        if item.role == "main" and item.bbox and len(item.bbox) == 4
    ]
    relations: list[SignRelation] = []
    for panel in observations:
        if panel.role != "supplementary" or not panel.bbox or len(panel.bbox) != 4:
            continue
        px1, py1, px2, py2 = panel.bbox
        panel_width = max(1.0, px2 - px1)
        panel_height = max(1.0, py2 - py1)
        candidates: list[tuple[float, float, SignObservation]] = []
        for main in main_signs:
            mx1, my1, mx2, my2 = main.bbox or [0, 0, 0, 0]
            main_width = max(1.0, mx2 - mx1)
            main_height = max(1.0, my2 - my1)
            overlap = max(0.0, min(px2, mx2) - max(px1, mx1)) / min(panel_width, main_width)
            vertical_gap = py1 - my2
            max_direct_gap = max(panel_height, main_height) * 1.5
            if overlap >= 0.4 and -panel_height * 0.25 <= vertical_gap <= max_direct_gap:
                # Distance takes precedence so a farther sign cannot win just
                # because its bounding box overlaps more of the panel.
                candidates.append((max(0.0, vertical_gap), overlap, main))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], -item[1]))
        gap, overlap, target = candidates[0]
        target_bbox = target.bbox or [0, 0, 0, 0]
        main_height = target_bbox[3] - target_bbox[1]
        gap_limit = max(panel_height, main_height)
        confidence = "high" if overlap >= 0.7 and gap <= gap_limit else "medium"
        if len(candidates) > 1:
            next_gap, next_overlap, _ = candidates[1]
            if (
                abs(overlap - next_overlap) < 0.1
                and abs(gap - next_gap) < max(8, panel_height * 0.25)
            ):
                confidence = "low"
        relations.append(SignRelation(
            relation_id=f"supplements:{panel.observation_id}->{target.observation_id}",
            relation_type="supplements",
            source_observation_id=panel.observation_id,
            target_observation_id=target.observation_id,
            confidence=confidence,
            source="detector_geometry",
        ))
    return relations
