"""Streamlit workspace for reviewing, annotating, and triaging draft sign rules."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Streamlit may put only src/ui on sys.path (especially when launched from
# another working directory). Resolve the repository root from this file so
# the app's package imports work regardless of where the command is run.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.symbolic.rule_review_store import (
    DEFAULT_REVIEW_DB,
    latest_reviews,
    record_review,
    review_history,
)
from src.core.symbolic.sign_catalog import DEFAULT_CATALOG_PATH, OfficialSignCatalog
from src.core.symbolic.traffic_rules import RULEBOOK_PATH, load_rulebook


st.set_page_config(page_title="Traffic Rule Review", page_icon="🚦", layout="wide")


def _rule_fingerprint(rule: Any) -> str:
    return hashlib.sha256(
        json.dumps(rule.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _rule_sign_codes(rule: Any) -> str:
    return ", ".join(rule.trigger_sign_codes) or "No trigger sign code"


def _matches(rule: Any, query: str) -> bool:
    haystack = " ".join(
        [rule.rule_id, _rule_sign_codes(rule), rule.reason, *rule.actions]
    ).casefold()
    return query.casefold() in haystack


@st.cache_data(show_spinner=False)
def _load_review_data(rulebook_path: str, catalog_path: str) -> tuple[list[dict], dict, dict]:
    _, rules = load_rulebook(rulebook_path)
    catalog = OfficialSignCatalog.load(catalog_path)
    serialized = [rule.to_dict() for rule in rules]
    # Keep schema-backed objects out of Streamlit's cache serialization.
    return serialized, {item["rule_id"]: item for item in serialized}, {
        item.get("official_code"): item for item in catalog.items
    }


st.title("Traffic Sign Rule Review")
st.caption("Review legal mappings and record team decisions. Draft files remain unchanged.")
st.warning(
    "A review marked APPROVED here is a team review record only. It does not change the rulebook, "
    "mark a rule legally verified, or make the engine use it as a verified rule."
)

try:
    serialized_rules, rules_by_id, catalog_by_code = _load_review_data(
        str(RULEBOOK_PATH), str(DEFAULT_CATALOG_PATH)
    )
except Exception as exc:  # Show a useful app-level error instead of a traceback.
    st.error(f"Could not load the rule or sign catalog: {exc}")
    st.stop()

if not serialized_rules:
    st.info("No rules were found in the current rulebook.")
    st.stop()

review_db_path = Path(DEFAULT_REVIEW_DB)
reviews = latest_reviews(review_db_path)
counts = {"Unreviewed": 0, "Approved": 0, "Needs revision": 0, "Rejected": 0}
for rule in serialized_rules:
    review = reviews.get(rule["rule_id"])
    verdict = review["verdict"] if review else "UNREVIEWED"
    if verdict == "UNREVIEWED":
        counts["Unreviewed"] += 1
    elif verdict == "APPROVED":
        counts["Approved"] += 1
    elif verdict == "NEEDS_REVISION":
        counts["Needs revision"] += 1
    else:
        counts["Rejected"] += 1

metric_columns = st.columns(4)
for column, (label, count) in zip(metric_columns, counts.items()):
    column.metric(label, count)

with st.sidebar:
    st.header("Review queue")
    reviewer = st.text_input("Reviewer name or team ID", key="rule_review_reviewer")
    status_filter = st.selectbox(
        "Decision status",
        ["All", "Unreviewed", "Approved", "Needs revision", "Rejected"],
    )
    search = st.text_input("Search rule ID, sign code, action, or text")
    st.caption(f"Rule source: `{Path(RULEBOOK_PATH).name}`")
    st.caption(f"Sign catalog: `{Path(DEFAULT_CATALOG_PATH).name}`")

filtered_rules = []
for item in serialized_rules:
    current = reviews.get(item["rule_id"])
    current_verdict = current["verdict"] if current else "UNREVIEWED"
    display_status = {
        "UNREVIEWED": "Unreviewed",
        "APPROVED": "Approved",
        "NEEDS_REVISION": "Needs revision",
        "REJECTED": "Rejected",
    }[current_verdict]
    if status_filter != "All" and status_filter != display_status:
        continue
    # Rehydrate only the basic searchable fields; rule IDs/codes/actions are plain JSON.
    haystack = " ".join(
        [item["rule_id"], ", ".join(item.get("trigger_sign_codes", [])), item.get("reason", ""),
         *item.get("actions", [])]
    ).casefold()
    if search and search.casefold() not in haystack:
        continue
    filtered_rules.append(item)

if not filtered_rules:
    st.info("No rules match the selected filters.")
    st.stop()

labels = [
    f"{item['rule_id']} · {', '.join(item.get('trigger_sign_codes', [])) or 'no sign code'} · "
    f"{(item.get('reason') or 'No summary').replace(chr(10), ' ')[:90]}"
    for item in filtered_rules
]
selected_label = st.selectbox("Select a draft rule", labels)
selected = filtered_rules[labels.index(selected_label)]
rule_id = selected["rule_id"]
current_review = reviews.get(rule_id)
selected_fingerprint = hashlib.sha256(
    json.dumps(selected, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()

header_left, header_right = st.columns([3, 1])
with header_left:
    st.subheader(rule_id)
    st.write(f"**Sign code(s):** {_rule_sign_codes(type('Rule', (), selected)())}")
with header_right:
    st.metric("Engine verification", selected.get("verification_status", "unverified").upper())

if current_review and current_review.get("rule_fingerprint") != selected_fingerprint:
    st.error("The rule changed since its latest review. Re-review this updated draft before relying on that decision.")
elif current_review:
    st.info(
        f"Latest team decision: **{current_review['verdict'].replace('_', ' ').title()}** "
        f"by {current_review['reviewer']} · {current_review['reviewed_at']}"
    )

rule_tab, source_tab, history_tab = st.tabs(["Rule", "Official source", "Review history"])
with rule_tab:
    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("**Applicability**")
        st.write("Intent: CHECK_PERMISSION")
        st.write("Actions:", ", ".join(selected.get("actions", [])) or "Not specified")
        st.write("Actor scope:", ", ".join(selected.get("actor_scope", [])) or "Not specified")
        st.write("Excluded actors:", ", ".join(selected.get("excluded_actor_scope", [])) or "None listed")
        st.write("Decision:", selected.get("outcome", "UNKNOWN"))
    with top_right:
        st.markdown("**Rule explanation**")
        st.write(selected.get("reason") or "No reason text provided.")
    st.markdown("**Conditions**")
    if selected.get("conditions"):
        for condition in selected["conditions"]:
            label = f"{condition.get('predicate', 'unknown')} · {condition.get('operator', 'equals')}"
            with st.expander(label):
                st.write("Expected value:", condition.get("expected_value"))
                st.write("Required:", condition.get("required", True))
                st.write("If missing:", condition.get("on_missing", "CONDITIONAL"))
                st.write("If not matched:", condition.get("on_mismatch", "NOT_APPLICABLE"))
    else:
        st.info("No explicit conditions are declared.")
    st.markdown("**Effects**")
    if selected.get("effects"):
        for effect in selected["effects"]:
            label = f"{effect.get('effect_type', 'unknown')} · {effect.get('action') or 'no action'}"
            with st.expander(label):
                st.write("Value:", effect.get("value"))
                st.write("Unit:", effect.get("unit"))
                st.write("Value predicate:", effect.get("value_predicate"))
                st.write("Scope:", effect.get("scope") or "Not specified")
                st.write("Description:", effect.get("description") or "Not specified")
    else:
        st.info("No structured effects are declared.")
    with st.expander("Full rule JSON"):
        st.json(selected)

with source_tab:
    codes = selected.get("trigger_sign_codes", [])
    if not codes:
        st.warning("This rule has no trigger sign code.")
    for code in codes:
        st.markdown(f"#### {code}")
        sign = catalog_by_code.get(code)
        if not sign:
            st.error("No exact-code catalog entry was found; verify the code/variant mapping.")
        else:
            st.write("**Official name:**", sign.get("official_name") or "Not provided")
            st.write("**Meaning:**", sign.get("legal_meaning") or "Not provided")
            st.write("**Catalog review status:**", sign.get("review_status", "unverified"))
            if sign.get("issues"):
                st.write("**Catalog issues:**", "; ".join(sign["issues"]))
        refs = [ref for ref in selected.get("legal_basis", []) if isinstance(ref, dict)]
        if not refs and sign:
            refs = sign.get("legal_basis", [])
        if refs:
            for index, ref in enumerate(refs, 1):
                st.markdown(f"**Legal basis {index}**")
                st.write("Document:", ref.get("document", "Not specified"))
                st.write("Edition:", ref.get("edition", "Not specified"))
                st.write("Section / figure:", ref.get("section_or_figure", "Not specified"))
                st.write("Source status:", ref.get("verification_status", "unverified"))
                if ref.get("source_url"):
                    st.markdown(f"[Open official source]({ref['source_url']})")
        else:
            st.error("No legal basis is attached to this rule or its catalog entry.")
    st.caption("Sign catalog extraction is not itself legal approval. Check the cited official text and current validity.")

with history_tab:
    history = review_history(rule_id, review_db_path)
    if not history:
        st.info("No team review has been recorded for this rule.")
    for item in history:
        with st.expander(f"{item['verdict'].replace('_', ' ').title()} · {item['reviewer']} · {item['reviewed_at']}", expanded=not history):
            st.write(item["notes"])
            if item.get("proposed_changes"):
                st.markdown("**Suggested edits**")
                st.write(item["proposed_changes"])
            st.json(item["checks"])
            st.caption(f"Reviewed rule fingerprint: `{item['rule_fingerprint'][:16]}`")

st.divider()
st.subheader("Record a team review")
st.caption("This app records an append-only review event in a local SQLite database. It never edits the rule or catalog JSON.")
with st.form(f"review_form_{rule_id}"):
    verdict = st.selectbox("Decision", ["NEEDS_REVISION", "APPROVED", "REJECTED"])
    st.markdown("**Approval checklist** — all boxes are required before selecting APPROVED.")
    checked = {
        "official_source_and_current_validity_checked": st.checkbox("I checked the cited official text and its current validity."),
        "sign_variant_and_effect_checked": st.checkbox("I confirmed the sign variant and intended legal effect."),
        "scope_conditions_and_exceptions_checked": st.checkbox("I checked actor scope, conditions, exceptions, and supplementary signs."),
        "scenario_tests_reviewed": st.checkbox("I reviewed representative examples, including missing/ambiguous-fact behavior."),
    }
    notes = st.text_area("Review notes (required)", placeholder="Explain the decision and cite any issue or ambiguity.")
    proposed_changes = st.text_area("Suggested rule edits (optional)", placeholder="Describe proposed edits; this is a review note, not an automatic change.")
    submitted = st.form_submit_button("Save review decision", type="primary")

if submitted:
    if not reviewer.strip():
        st.error("Enter your name or team ID in the sidebar before saving a review.")
    elif verdict == "APPROVED" and not all(checked.values()):
        st.error("Approval is blocked until every checklist item is confirmed.")
    else:
        try:
            record_review(
                rule_id=rule_id,
                verdict=verdict,
                reviewer=reviewer,
                notes=notes,
                checks=checked,
                proposed_changes=proposed_changes,
                rule_fingerprint=selected_fingerprint,
                path=review_db_path,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("Review saved. The rule remains unverified in the engine until separately updated and validated.")
            st.rerun()
