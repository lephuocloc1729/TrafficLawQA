from __future__ import annotations

import unittest

from src.core.symbolic.rule_coverage import load_default_rule_coverage_report


class RuleCoverageTest(unittest.TestCase):
    def test_current_coverage_report_keeps_meaning_and_rule_coverage_separate(self) -> None:
        report = load_default_rule_coverage_report()

        self.assertEqual(report["catalog_item_count"], 442)
        self.assertEqual(report["coded_sign_count"], 402)
        self.assertEqual(report["family_heading_count"], 40)
        self.assertEqual(report["duplicate_catalog_code_count"], 0)
        self.assertEqual(report["duplicate_canonical_sign_id_count"], 0)
        self.assertEqual(report["meaning_available_count"], 402)
        self.assertEqual(report["meaning_missing_count"], 0)
        self.assertEqual(report["verified_rule_sign_count"], 0)
        self.assertEqual(report["draft_rule_sign_count"], 265)
        self.assertEqual(report["no_rule_record_sign_count"], 137)
        self.assertEqual(report["rule_ids_without_catalog_code"], [])
        i406 = next(item for item in report["items"] if item["official_code"] == "I.406")
        self.assertEqual(i406["draft_rule_count"], 2)
        self.assertEqual(i406["verified_rule_count"], 0)
        for code in ("I.407a", "I.407b", "I.407c"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("R.E.9a", "R.E.9b", "R.E.10a", "R.E.10b", "R.E.9d", "R.E.10d"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("DP.127a", "DP.127b", "DP.127c"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        i418 = next(item for item in report["items"] if item["official_code"] == "I.418")
        self.assertEqual(i418["draft_rule_count"], 1)
        self.assertEqual(i418["verified_rule_count"], 0)
        for code in ("IE.468a", "IE.468b", "IE.468c", "IE.471", "IE.473"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.408", "I.408a", "I.409", "I.410"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.413a", "I.413b", "I.413c", "I.417a", "I.417b", "I.417c"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.414a", "I.414b", "I.414c", "I.414d", "I.415", "I.416"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.437", "I.440", "I.441a", "I.441b", "I.441c", "I.446", "I.448"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.423a", "I.423b", "I.423c", "I.424a", "I.424b", "I.424c", "I.424d"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.425", *(f"I.445{variant}" for variant in "abcdefgh")):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertGreater(row["draft_rule_count"], 0)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in (
            "I.426", "I.427a", "I.427b", "I.428a", "I.428b", "I.428c", "I.429",
            "I.430", "I.431", "I.432", "I.433a", "I.433b", "I.433c", "I.433d",
            "I.433e", "I.435",
        ):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 0)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in ("I.434a", "I.434b"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        i436 = next(item for item in report["items"] if item["official_code"] == "I.436")
        self.assertEqual(i436["draft_rule_count"], 2)
        for code in ("R.415a", "R.415b", "R.420", "R.421"):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 1)
            self.assertEqual(row["verified_rule_count"], 0)
        for code in (*(f"R.412{variant}" for variant in "ijklmnop"), *(f"R.413{variant}" for variant in "ijklmnop")):
            row = next(item for item in report["items"] if item["official_code"] == code)
            self.assertEqual(row["draft_rule_count"], 0)
        i445e = next(item for item in report["items"] if item["official_code"] == "I.445e")
        self.assertEqual(i445e["draft_rule_count"], 2)
        self.assertEqual(report["uncoded_figure_review_count"], 118)
        self.assertEqual(report["unresolved_figure_review_count"], 6)
        self.assertEqual(report["figure_linked_to_catalog_sign_count"], 13)
        self.assertEqual(report["out_of_scope_figure_count"], 87)
        self.assertEqual(report["technical_diagram_count"], 12)
        self.assertTrue(all(item["catalog_review_status"] != "verified" for item in report["items"]))
        self.assertIn("not evidence that the sign has no legal effect", report["review_note"])
        warning_rows = [item for item in report["items"] if item["official_code"].startswith("W.")]
        self.assertEqual(len(warning_rows), 96)
        warning_signs = [item for item in warning_rows if item["catalog_entry_kind"] == "sign_or_variant"]
        warning_families = [item for item in warning_rows if item["catalog_entry_kind"] == "family_heading"]
        self.assertEqual(len(warning_signs), 83)
        self.assertEqual(len(warning_families), 13)
        self.assertTrue(all(item["rule_status"] == "draft_rule_review_required" for item in warning_signs))
        self.assertTrue(all(item["rule_status"] == "variant_required" for item in warning_families))


if __name__ == "__main__":
    unittest.main()
