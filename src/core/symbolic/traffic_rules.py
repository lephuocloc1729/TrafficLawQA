from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.core.symbolic.sign_schema import Decision, QuestionIntent, RuleMatch, TrafficSignFact


MOTOR_VEHICLES = {"car", "motorbike", "truck", "bus", "motor_vehicle", "unknown"}
VEHICLES = {*MOTOR_VEHICLES, "bicycle", "vehicle"}
DEFAULT_RULE_INTENTS = frozenset(
    {
        "CHECK_PERMISSION",
        "QUERY_SIGN_MEANING",
        "ASK_CONDITION",
        "QUERY_TIME_CONDITION",
        "QUERY_PROHIBITED_SUBJECTS",
    }
)


@dataclass(frozen=True)
class TrafficRule:
    rule_id: str
    sign_codes: frozenset[str]
    actions: frozenset[str]
    decision: Decision
    reason: str
    confidence: str = "high"
    allowed_intents: frozenset[str] = DEFAULT_RULE_INTENTS
    target_vehicle: str = "unknown"
    required_time_context: str | None = None
    sign_families: frozenset[str] = frozenset()
    legal_basis: tuple[tuple[str, str], ...] = ()

    def matches(self, fact: TrafficSignFact, intent: QuestionIntent) -> bool:
        if intent.intent not in self.allowed_intents:
            return False
        if (
            fact.sign_code not in self.sign_codes
            and fact.sign_family not in self.sign_families
        ):
            return False
        intent_can_ask_about_sign = intent.intent in {
            "QUERY_SIGN_MEANING",
            "ASK_CONDITION",
            "QUERY_TIME_CONDITION",
            "QUERY_PROHIBITED_SUBJECTS",
        }
        if intent.action not in self.actions and not intent_can_ask_about_sign:
            return False
        return vehicle_is_covered(intent.vehicle, self.target_vehicle or fact.target_vehicle)

    def _resolved_decision(
        self,
        fact: TrafficSignFact,
        intent: QuestionIntent,
    ) -> tuple[Decision, str, str]:
        if self.rule_id == "P127_SPEED_LIMIT" and fact.metadata.get("speed_limit") is None:
            return (
                "CONDITIONAL",
                f"{self.reason} Chưa đọc được trị số tốc độ trên biển.",
                "medium",
            )

        if intent.intent in {"ASK_CONDITION", "QUERY_TIME_CONDITION"}:
            condition = self.required_time_context or "additional_sign_context"
            return (
                "CONDITIONAL",
                f"{self.reason} Điều kiện cần kiểm tra: {condition}.",
                "medium",
            )

        if not self.required_time_context:
            return self.decision, self.reason, self.confidence

        if intent.time_context == self.required_time_context:
            return self.decision, self.reason, self.confidence

        other_parity = {
            "odd_days": "even_days",
            "even_days": "odd_days",
        }.get(self.required_time_context)
        if intent.time_context == other_parity:
            return (
                "ALLOWED",
                (
                    f"{self.reason} Câu hỏi xác định thời điểm thuộc ngày đối diện "
                    "với điều kiện cấm của biển."
                ),
                "medium",
            )

        if not intent.time_context:
            return (
                "CONDITIONAL",
                (
                    f"{self.reason} Câu hỏi chưa cung cấp ngữ cảnh thời gian đủ rõ, "
                    "nên kết luận symbolic giữ ở mức có điều kiện."
                ),
                "medium",
            )

        return (
            "UNKNOWN",
            (
                f"{self.reason} Ngữ cảnh thời gian của câu hỏi không khớp điều kiện "
                "của biển, nên rule này không đủ căn cứ để cấm tại thời điểm đó."
            ),
            "low",
        )

    def to_match(self, fact: TrafficSignFact, intent: QuestionIntent) -> RuleMatch:
        decision, reason, confidence = self._resolved_decision(fact, intent)
        return RuleMatch(
            rule_id=self.rule_id,
            decision=decision,
            matched_fact_ids=[fact.sign_code],
            matched_facts=[fact.to_dict()],
            legal_basis=_rule_legal_basis(self.legal_basis, fact.related_articles),
            reason=reason,
            confidence=confidence,  # type: ignore[arg-type]
            metadata={
                "action_scope": sorted(self.actions),
                "vehicle_scope": self.target_vehicle,
                "time_condition": self.required_time_context,
                "sign_families": sorted(self.sign_families),
            },
        )


def _rule_legal_basis(
    explicit_basis: tuple[tuple[str, str], ...],
    fact_basis: list[dict[str, str]],
) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    for law_id, article_id in explicit_basis:
        articles.append({"law_id": law_id, "article_id": article_id})
    for article in fact_basis:
        if article not in articles:
            articles.append(article)
    return articles


TRAFFIC_RULES: tuple[TrafficRule, ...] = (
    TrafficRule(
        rule_id="P130_NO_STOPPING_NO_PARKING",
        sign_codes=frozenset({"P.130"}),
        actions=frozenset({"parking", "stopping"}),
        decision="PROHIBITED",
        reason="Biển P.130 cấm dừng xe và đỗ xe đối với phương tiện trong phạm vi hiệu lực của biển.",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P131A_NO_PARKING",
        sign_codes=frozenset({"P.131a"}),
        actions=frozenset({"parking"}),
        decision="PROHIBITED",
        reason="Biển P.131a cấm đỗ xe đối với phương tiện trong phạm vi hiệu lực của biển.",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P131B_ODD_DAY_NO_PARKING",
        sign_codes=frozenset({"P.131b"}),
        actions=frozenset({"parking"}),
        decision="PROHIBITED",
        reason="Biển P.131b cấm đỗ xe vào ngày lẻ.",
        required_time_context="odd_days",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P131C_EVEN_DAY_NO_PARKING",
        sign_codes=frozenset({"P.131c"}),
        actions=frozenset({"parking"}),
        decision="PROHIBITED",
        reason="Biển P.131c cấm đỗ xe vào ngày chẵn.",
        required_time_context="even_days",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P123A_NO_LEFT_TURN",
        sign_codes=frozenset({"P.123a"}),
        actions=frozenset({"turn_left"}),
        decision="PROHIBITED",
        reason="Biển P.123a cấm rẽ trái đối với phương tiện trong phạm vi hiệu lực của biển.",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P123B_NO_RIGHT_TURN",
        sign_codes=frozenset({"P.123b"}),
        actions=frozenset({"turn_right"}),
        decision="PROHIBITED",
        reason="Biển P.123b cấm rẽ phải đối với phương tiện trong phạm vi hiệu lực của biển.",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P102_NO_ENTRY",
        sign_codes=frozenset({"P.102"}),
        actions=frozenset({"entering", "circulation"}),
        decision="PROHIBITED",
        reason="Biển P.102 cấm phương tiện đi vào theo chiều đặt biển.",
        target_vehicle="vehicle",
    ),
    TrafficRule(
        rule_id="P127_SPEED_LIMIT",
        sign_codes=frozenset({"P.127"}),
        actions=frozenset({"speed"}),
        decision="CONDITIONAL",
        reason="Biển P.127 quy định tốc độ tối đa; cần đọc trị số trên biển hoặc biển phụ để kết luận mức cụ thể.",
        allowed_intents=DEFAULT_RULE_INTENTS,
        target_vehicle="motor_vehicle",
    ),
)


def vehicle_is_covered(vehicle: str, target_vehicle: str) -> bool:
    vehicle = (vehicle or "unknown").strip().lower()
    target_vehicle = (target_vehicle or "unknown").strip().lower()

    if target_vehicle == "unknown" or vehicle == "unknown":
        return True
    if target_vehicle == "motor_vehicle":
        return vehicle in MOTOR_VEHICLES
    if target_vehicle == "vehicle":
        return vehicle in VEHICLES
    return vehicle == target_vehicle


def find_matching_rules(
    facts: Iterable[TrafficSignFact],
    intent: QuestionIntent,
    rules: Iterable[TrafficRule] = TRAFFIC_RULES,
) -> list[RuleMatch]:
    matches: list[RuleMatch] = []
    for fact in facts:
        for rule in rules:
            if rule.matches(fact, intent):
                matches.append(rule.to_match(fact, intent))
    return matches
