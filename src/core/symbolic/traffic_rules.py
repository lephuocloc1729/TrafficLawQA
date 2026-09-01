from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.core.symbolic.sign_schema import Decision, QuestionIntent, RuleMatch, TrafficSignFact


MOTOR_VEHICLES = {"car", "motorbike", "truck", "bus", "motor_vehicle", "unknown"}
VEHICLES = {*MOTOR_VEHICLES, "bicycle", "vehicle"}


@dataclass(frozen=True)
class TrafficRule:
    rule_id: str
    sign_codes: frozenset[str]
    actions: frozenset[str]
    decision: Decision
    reason: str
    confidence: str = "high"
    allowed_intents: frozenset[str] = frozenset({"CHECK_PERMISSION"})
    target_vehicle: str = "unknown"

    def matches(self, fact: TrafficSignFact, intent: QuestionIntent) -> bool:
        if intent.intent not in self.allowed_intents:
            return False
        if fact.sign_code not in self.sign_codes:
            return False
        if intent.action not in self.actions:
            return False
        return vehicle_is_covered(intent.vehicle, self.target_vehicle or fact.target_vehicle)

    def to_match(self, fact: TrafficSignFact) -> RuleMatch:
        return RuleMatch(
            rule_id=self.rule_id,
            decision=self.decision,
            matched_fact_ids=[fact.sign_code],
            matched_facts=[fact.to_dict()],
            legal_basis=fact.related_articles,
            reason=self.reason,
            confidence=self.confidence,  # type: ignore[arg-type]
        )


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
        decision="CONDITIONAL",
        reason="Biển P.131b cấm đỗ xe vào ngày lẻ; cần xét ngày cụ thể để kết luận cho mọi thời điểm.",
        target_vehicle="motor_vehicle",
    ),
    TrafficRule(
        rule_id="P131C_EVEN_DAY_NO_PARKING",
        sign_codes=frozenset({"P.131c"}),
        actions=frozenset({"parking"}),
        decision="CONDITIONAL",
        reason="Biển P.131c cấm đỗ xe vào ngày chẵn; cần xét ngày cụ thể để kết luận cho mọi thời điểm.",
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
        allowed_intents=frozenset({"CHECK_PERMISSION", "QUERY_SIGN_MEANING", "UNKNOWN"}),
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
                matches.append(rule.to_match(fact))
    return matches
