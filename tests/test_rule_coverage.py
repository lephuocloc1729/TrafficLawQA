from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from src.core.symbolic.rule_coverage import load_default_rule_coverage_report
from src.core.symbolic.sign_normalizer import normalize_sign_code
from src.core.symbolic.traffic_rules import TRAFFIC_RULES


def yolo_class_codes() -> tuple[dict[str, set[str]], set[str]]:
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
        runtime_codes.add(normalize_sign_code(family_code))
        runtime_codes.update(normalize_sign_code(code) for code in variants)
    return class_codes, runtime_codes


class RuleCoverageTest(unittest.TestCase):
    def test_source_catalog_coverage_is_separate_from_yolo_runtime_rule_scope(self) -> None:
        report = load_default_rule_coverage_report()
        self.assertEqual(report["catalog_item_count"], 442)
        self.assertEqual(report["coded_sign_count"], 402)
        self.assertEqual(report["family_heading_count"], 40)
        self.assertEqual(report["duplicate_catalog_code_count"], 0)
        self.assertEqual(report["duplicate_canonical_sign_id_count"], 0)
        self.assertEqual(report["meaning_available_count"], 402)
        self.assertEqual(report["meaning_missing_count"], 0)
        self.assertEqual(report["rule_ids_without_catalog_code"], [])
        self.assertIn("not evidence that the sign has no legal effect", report["review_note"])

    def test_runtime_rules_only_reference_codes_from_yolo26s_vtsdb100(self) -> None:
        class_codes, yolo_codes = yolo_class_codes()
        self.assertEqual(len(class_codes) + 1, 100)  # one ambiguous class has no canonical code

        runtime_codes = {
            normalize_sign_code(code)
            for rule in TRAFFIC_RULES
            for code in rule.trigger_sign_codes
        }
        self.assertTrue(runtime_codes)
        self.assertLessEqual(runtime_codes, yolo_codes)

        codes_with_rules = {
            normalize_sign_code(code)
            for rule in TRAFFIC_RULES
            for code in rule.trigger_sign_codes
        }
        missing_rule_classes = sorted(
            label for label, code in class_codes.items()
            if not code.intersection(codes_with_rules)
        )
        self.assertEqual(missing_rule_classes, ["dead_end", "market"])
        self.assertEqual(len(TRAFFIC_RULES), 121)

    def test_rule_draft_registry_only_contains_yolo_mapped_trigger_codes(self) -> None:
        path = (
            Path(__file__).parents[1]
            / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
        )
        registry = json.loads(path.read_text(encoding="utf-8"))
        _class_codes, yolo_codes = yolo_class_codes()
        self.assertEqual(registry["registry_version"], "qcvn41-yolov26s100-rule-drafts-1")
        self.assertEqual(len(registry["rules"]), 118)
        self.assertTrue(all(
            normalize_sign_code(rule["code"]) in yolo_codes
            for rule in registry["rules"]
        ))


if __name__ == "__main__":
    unittest.main()
