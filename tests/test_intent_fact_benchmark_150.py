import json
import re
from pathlib import Path

from src.core.symbolic.condition_facts import CONDITION_PREDICATES
from src.core.symbolic.fact_schema import TrafficFacts
from src.core.symbolic.intent_schema import QueryIntent, validate_intent_payload
from src.core.symbolic.reasoner import reason_over_facts


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "examples" / "intent_fact_benchmark_150_train.json"
TEST_PATH = ROOT / "examples" / "intent_fact_benchmark_150_test.json"


def _load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _project_expected_facts(actual: dict, expected: dict) -> dict:
    """Select the question-specific expected facts from the full trace facts."""
    signs = []
    for expected_sign in expected.get("signs", []):
        actual_sign = next(
            (item for item in actual.get("signs", []) if item.get("id") == expected_sign.get("id")),
            None,
        )
        if actual_sign is None:
            continue

        selected = []
        for expected_fact in expected_sign.get("facts", []):
            if expected_fact.get("predicate") == "conditions":
                requested = expected_fact.get("value", {})
                actual_condition_fact = next(
                    (
                        item for item in actual_sign.get("facts", [])
                        if item.get("predicate") == "conditions" and isinstance(item.get("value"), dict)
                    ),
                    None,
                )
                if actual_condition_fact is not None:
                    values = {
                        key: actual_condition_fact["value"][key]
                        for key in requested
                        if key in actual_condition_fact["value"]
                    }
                    if values:
                        selected.append({"predicate": "conditions", "value": values})
            else:
                selected.extend(
                    item for item in actual_sign.get("facts", [])
                    if item.get("predicate") == expected_fact.get("predicate")
                )

        if selected:
            signs.append({
                **{key: value for key, value in actual_sign.items() if key != "facts"},
                "facts": selected,
            })

    relations = [
        item for item in actual.get("relations", [])
        if item in expected.get("relations", [])
    ]
    scene_facts = [
        item
        for expected_fact in expected.get("scene_facts", [])
        for item in actual.get("scene_facts", [])
        if item.get("predicate") == expected_fact.get("predicate")
        and item.get("value") == expected_fact.get("value")
    ]
    return {"signs": signs, "relations": relations, "scene_facts": scene_facts}


def _compact(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_150_case_benchmark_matches_rule_engine_and_answer_facts() -> None:
    train = _load_cases(TRAIN_PATH)
    test = _load_cases(TEST_PATH)
    assert len(train) == 120
    assert len(test) == 30
    assert {item["case_id"] for item in train}.isdisjoint(
        {item["case_id"] for item in test}
    )

    for split_name, cases in (("train", train), ("test", test)):
        seen_conditions = set()
        seen_intent_kinds = set()
        for case in cases:
            facts = TrafficFacts(**case["facts"])
            intent_data = case["intent_json"]
            for intent in intent_data["intents"]:
                error = validate_intent_payload(intent)
                assert error is None, f"{split_name}/{case['case_id']}: {error}"
                seen_intent_kinds.add(intent["kind"])
            seen_conditions.update(
                fact.predicate for fact in facts.context_facts
                if fact.predicate in CONDITION_PREDICATES
            )
            query = QueryIntent(
                schema_version=intent_data["schema_version"],
                query_id=case["case_id"],
                raw_query=intent_data["question"],
                intents=intent_data["intents"],
            )
            trace = reason_over_facts(facts, query)

            kind = query.intents[0].kind
            if kind in {"CHECK_PERMISSION", "UNKNOWN"}:
                expected = re.match(
                    r"^(ALLOWED|PROHIBITED|CONDITIONAL|UNKNOWN)\b",
                    case["expected_answer_text"],
                )
                assert expected is not None, f"{split_name}/{case['case_id']}: missing decision label"
                assert trace.decision is not None
                assert trace.decision.decision == expected.group(1), (
                    f"{split_name}/{case['case_id']}: expected {expected.group(1)}, "
                    f"got {trace.decision.decision}; {trace.decision.reason}"
                )

            projected = _project_expected_facts(trace.facts, case["expected_answer_facts"])
            assert _compact(projected) == _compact(case["expected_answer_facts"]), (
                f"{split_name}/{case['case_id']}: expected answer facts do not match trace facts"
            )

        assert seen_intent_kinds == {"CHECK_PERMISSION", "INFORMATION", "UNKNOWN"}
        assert seen_conditions == CONDITION_PREDICATES, (
            f"{split_name}: missing condition coverage "
            f"{sorted(CONDITION_PREDICATES - seen_conditions)}"
        )
