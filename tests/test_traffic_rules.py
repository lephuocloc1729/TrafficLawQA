from __future__ import annotations

import ast
from pathlib import Path
import unittest

from src.core.symbolic.fact_schema import SignObservation
from src.core.symbolic.intent_schema import Intent
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.reasoner import decide_symbolically, _evaluate_intent
from src.core.symbolic.rule_schema import RuleEvaluation
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation, normalize_sign_code
from src.core.symbolic.traffic_rules import (
    TRAFFIC_RULEBOOK_METADATA,
    TRAFFIC_RULES,
    _evaluate_candidate,
)


def yolo_class_codes() -> tuple[dict[str, set[str]], set[str], set[str]]:
    source = Path(__file__).parents[1] / "src/deps/detection/yolo26s.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    class_maps = {}
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"_EXACT_CLASSES", "_AMBIGUOUS_CLASSES", "_FAMILY_CLASSES"}
        ):
            class_maps[node.target.id] = ast.literal_eval(node.value)

    class_codes = {
        label: {normalize_sign_code(value[0])}
        for label, value in class_maps["_EXACT_CLASSES"].items()
    }
    runtime_codes = set().union(*class_codes.values())
    for label, (family_code, variants, *_rest) in class_maps["_FAMILY_CLASSES"].items():
        class_codes[label] = {
            normalize_sign_code(family_code),
            *(normalize_sign_code(code) for code in variants),
        }
        runtime_codes.update(class_codes[label])
    return class_codes, runtime_codes, set(class_maps["_AMBIGUOUS_CLASSES"])


class TrafficRulesTest(unittest.TestCase):
    def test_runtime_registry_is_scoped_to_the_100_yolo_classes(self) -> None:
        class_codes, yolo_codes, ambiguous_classes = yolo_class_codes()
        self.assertEqual(len(class_codes) + len(ambiguous_classes), 100)
        self.assertEqual(len(TRAFFIC_RULES), 121)
        self.assertEqual(TRAFFIC_RULEBOOK_METADATA["rule_draft_count"], 118)
        self.assertEqual(TRAFFIC_RULEBOOK_METADATA["rule_draft_registry"], "qcvn41-yolov26s100-rule-drafts-1")

        runtime_codes = {
            normalize_sign_code(code)
            for rule in TRAFFIC_RULES
            for code in rule.trigger_sign_codes
        }
        self.assertTrue(runtime_codes)
        self.assertLessEqual(runtime_codes, yolo_codes)
        self.assertTrue(all(rule.actions and rule.legal_basis for rule in TRAFFIC_RULES))

    def test_classes_without_direct_rules_are_explicitly_known(self) -> None:
        class_codes, _yolo_codes, ambiguous_classes = yolo_class_codes()
        rule_codes = {
            normalize_sign_code(code)
            for rule in TRAFFIC_RULES
            for code in rule.trigger_sign_codes
        }
        uncovered = sorted(
            label for label, codes in class_codes.items()
            if not codes.intersection(rule_codes)
        )
        self.assertEqual(uncovered, ["dead_end", "market"])
        self.assertEqual(ambiguous_classes, {"other_prohibition_signs"})

    def test_permission_rule_evaluations_use_supported_outcome_states(self) -> None:
        for index, rule in enumerate(TRAFFIC_RULES):
            with self.subTest(rule_id=rule.rule_id):
                observation = SignObservation(
                    f"sign-{index}",
                    role="main",
                    sign_code=rule.trigger_sign_codes[0],
                    identity_status="identified",
                    confidence="high",
                )
                intent = Intent(
                    f"intent-{index}",
                    "CHECK_PERMISSION",
                    action={"value": rule.actions[0], "status": "provided"},
                    actor={"value": rule.actor_scope[0], "status": "provided"},
                )
                evaluation = _evaluate_candidate(rule, intent, observation, [])
                if evaluation.status in {"applied", "incomplete"}:
                    self.assertIn(evaluation.decision, {"PROHIBITED", "ALLOWED", "CONDITIONAL"})

    def test_confirmed_prohibition_wins_aggregate_permission_result(self) -> None:
        evaluations = [
            RuleEvaluation("rule-allow", "intent-1", decision="ALLOWED", status="applied"),
            RuleEvaluation("rule-prohibit", "intent-1", decision="PROHIBITED", status="applied"),
        ]
        result = _evaluate_intent("intent-1", evaluations, sign_inventory_status="complete")
        self.assertEqual(result.decision, "PROHIBITED")

    def test_incomplete_prohibition_is_not_treated_as_confirmed(self) -> None:
        evaluations = [
            RuleEvaluation("rule-prohibit", "intent-1", decision="PROHIBITED", status="incomplete",
                           missing_facts=["active_time"]),
        ]
        result = _evaluate_intent("intent-1", evaluations, sign_inventory_status="complete")
        self.assertEqual(result.decision, "CONDITIONAL")

    def test_p130_prohibition_still_reaches_the_reasoner(self) -> None:
        observation = normalize_payload_to_observation({"sign_code": "P.130"})
        decision = decide_symbolically(
            [observation],
            parse_query_intent_deterministic("Xe ô tô có được đỗ tại đây không?"),
        )
        self.assertEqual(decision.decision, "PROHIBITED")
        self.assertIn("P130_NO_STOPPING_NO_PARKING", decision.rules_fired)


if __name__ == "__main__":
    unittest.main()
