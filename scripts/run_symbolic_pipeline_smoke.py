#!/usr/bin/env python3
"""Run deterministic end-to-end smoke scenarios for intent/rule matching."""

from __future__ import annotations

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.symbolic.fact_schema import TrafficFacts
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.reasoner import reason_over_facts
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation
from src.core.symbolic.traffic_rules import TRAFFIC_RULES


DEFAULT_FIXTURE = ROOT / "examples" / "symbolic_pipeline_smoke.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--quiet", action="store_true", help="Print only failing cases and the final count.")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    fixed_now = datetime.fromisoformat(fixture["fixed_now"]).astimezone(
        ZoneInfo("Asia/Ho_Chi_Minh")
    )
    failures: list[str] = []

    for case in fixture["cases"]:
        observations = [
            normalize_payload_to_observation(item)
            for item in case["observations"]
        ]
        for observation in observations:
            observation.confidence = "high"
        facts = TrafficFacts(
            schema_version="2.0",
            scene_id=case["id"],
            sign_observations=observations,
            sign_inventory_status=("complete" if case.get("complete_sign_inventory") else "partial"),
            sign_inventory_evidence=([{
                "source": "smoke_fixture",
                "evidence": "Fixture explicitly lists all signs in the scene.",
            }] if case.get("complete_sign_inventory") else []),
        )
        parsed = parse_query_intent_deterministic(case["question"], now=fixed_now)
        trace = reason_over_facts(facts, parsed, rules=TRAFFIC_RULES)
        decision = trace.decision
        actual_intents = [item.to_dict() for item in parsed.query_intent.intents]
        actual_rules = [
            item["rule_id"] for item in decision.rule_evaluations
            if item["status"] != "not_applicable"
        ]
        expected_rule_ids = case["expected_rule_ids"]
        checks = {
            "intent": any(
                item.get("kind") == case["expected_intent"]["kind"]
                and ("action" not in case["expected_intent"] or item.get("action") == case["expected_intent"]["action"])
                and ("actor" not in case["expected_intent"] or item.get("actor") == case["expected_intent"]["actor"])
                for item in actual_intents
            ),
            "rules": sorted(actual_rules) == sorted(expected_rule_ids),
            "decision": decision.decision == case["expected_decision"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        if not args.quiet or not all(checks.values()):
            print(f"[{status}] {case['id']}")
            print(f"  intent: {actual_intents}")
            print(f"  matched rules: {actual_rules}")
            print(f"  decision: {decision.decision}; fired: {decision.rules_fired}")
            for evaluation in decision.rule_evaluations:
                print(f"  {evaluation['rule_id']}: {evaluation['status']}; missing={evaluation['missing_facts']}")
        if not all(checks.values()):
            failures.append(case["id"])
            print(f"  expected: {case['expected_intent']}; rules={expected_rule_ids}; decision={case['expected_decision']}")

    print(f"\n{len(fixture['cases']) - len(failures)}/{len(fixture['cases'])} scenarios passed; {len(TRAFFIC_RULES)} rules loaded.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
