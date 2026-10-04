from __future__ import annotations

import json
import re
import unicodedata as ud
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.core.symbolic.intent_schema import (
    Intent,
    IntentParseResult,
    IntentSlot,
    IntentTarget,
    QueryIntent,
)
from src.core.symbolic.sign_schema import normalize_confidence


VALID_INTENTS = {
    "CHECK_PERMISSION",
    "QUERY_SIGN_MEANING",
    "ASK_CONDITION",
    "ASK_PENALTY",
    "QUERY_PROHIBITED_SUBJECTS",
    "QUERY_TIME_CONDITION",
    "QUERY_DIRECTION_SCOPE",
    "QUERY_DISTANCE_SCOPE",
    "IDENTIFY_SIGN",
    "UNKNOWN",
}
VALID_ACTIONS = {
    "parking", "stopping", "turn_left", "turn_right", "entering", "speed",
    "circulation", "crossing", "overtaking", "unknown",
}
VALID_VEHICLES = {
    "car", "motorbike", "truck", "bus", "bicycle", "pedestrian",
    "motor_vehicle", "vehicle", "unknown",
}

INTENT_PATTERNS = (
    ("ASK_PENALTY", ("muc phat", "phat bao nhieu", "bi phat bao nhieu", "bao nhieu tien", "tien phat", "xu phat", "vi pham se bi gi")),
    ("QUERY_PROHIBITED_SUBJECTS", ("cam nhung phuong tien nao", "cam phuong tien nao", "loai xe nao bi cam", "xe nao bi cam", "cam xe nao")),
    ("QUERY_DIRECTION_SCOPE", ("huong nao", "chieu nao", "re huong", "ben nao", "phia nao")),
    ("QUERY_DISTANCE_SCOPE", ("bao xa", "khoang cach", "pham vi", "bao nhieu met")),
    ("QUERY_TIME_CONDITION", ("thoi gian nao", "thoi diem nao", "hieu luc vao luc nao", "co hieu luc khi nao")),
    ("ASK_CONDITION", ("dieu kien gi", "truong hop nao", "ap dung khi nao", "toc do toi da", "gioi han toc do", "bao nhieu km h")),
    ("QUERY_SIGN_MEANING", ("y nghia gi", "co nghia gi", "bien nay la gi", "noi dung bien")),
    ("IDENTIFY_SIGN", ("nhan dien bien", "xac dinh bien", "bien nao trong hinh", "ma bien gi")),
)
ACTION_PATTERNS = (
    ("parking", ("do xe", "dau xe", "dung do", "duoc do", "phep do", "cam do", "do tai", "do khong")),
    ("stopping", ("dung xe", "tam dung", "dung tai")),
    ("turn_left", ("re trai", "queo trai", "quay dau sang trai")),
    ("turn_right", ("re phai", "queo phai")),
    ("entering", ("di vao", "vao duong", "luu thong vao", "duoc vao", "cam vao")),
    ("speed", ("toc do", "van toc", "gioi han toc do", "chay bao nhieu")),
    ("circulation", ("luu thong", "di qua", "duoc di", "duoc chay", "tham gia giao thong")),
    ("crossing", ("qua duong", "sang duong")),
    ("overtaking", ("vuot xe", "vuot phuong tien")),
)
VEHICLE_PATTERNS = (
    ("bus", ("xe buyt", "xe bus", "xe khach")),
    ("truck", ("xe tai", "o to tai", "xe cho hang", "container")),
    ("motorbike", ("xe may", "mo to", "xe gan may")),
    ("bicycle", ("xe dap", "xe dap dien")),
    ("pedestrian", ("nguoi di bo", "di bo", "bo hanh")),
    ("car", ("o to", "xe hoi", "xe con", "xe oto")),
    ("motor_vehicle", ("xe co gioi", "phuong tien co gioi")),
    ("vehicle", ("phuong tien", "cac xe")),
)
TIME_MENTION_RE = re.compile(
    r"\b(?:bay gio|hien tai|luc nay|hom nay|ngay mai|hom qua|thu [2-7]|"
    r"chu nhat|ngay le|ngay chan|\d{1,2}\s*(?:gio|h)(?:\s*\d{1,2}\s*(?:phut|p))?|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"
)
PERMISSION_TERMS = ("duoc phep", "co duoc", "duoc khong", "co bi cam", "bi cam khong", "duoc", "cam")
LOCATION_PATTERNS = (
    ("current_sign_location", ("tai day", "tai vi tri nay", "tren doan duong nay", "o day")),
    ("sign_scope", ("trong pham vi bien", "trong khu vuc nay", "tren tuyen duong nay")),
    ("direction_scope", ("theo chieu dat bien", "theo huong dat bien")),
)


def normalize_question_text(question: str) -> str:
    text = ud.normalize("NFD", question or "").lower()
    text = "".join(char for char in text if ud.category(char) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def extract_action(question: str) -> str:
    text = normalize_question_text(question)
    return next((action for action, phrases in ACTION_PATTERNS if any(_contains(text, p) for p in phrases)), "unknown")


def extract_vehicle(question: str) -> str:
    text = normalize_question_text(question)
    hits = []
    for vehicle, phrases in VEHICLE_PATTERNS:
        for phrase in phrases:
            match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
            if match:
                hits.append((match.start(), -len(phrase), vehicle))
                break
    return sorted(hits)[0][2] if hits else "unknown"


def extract_time_context(question: str) -> str | None:
    text = normalize_question_text(question)
    raw = question or ""
    odd = bool(re.search(r"ngày\s+lẻ\b", raw, flags=re.IGNORECASE))
    even = bool(re.search(r"ngày\s+chẵn\b", raw, flags=re.IGNORECASE))
    if not odd and not even and "ngay chan" in text:
        even = True
    if odd == even:
        return None
    return "odd_days" if odd else "even_days"


def extract_location_context(question: str) -> str | None:
    text = normalize_question_text(question)
    return next((kind for kind, phrases in LOCATION_PATTERNS if any(_contains(text, p) for p in phrases)), None)


def _now_in_zone(now: datetime | None, timezone_name: str) -> datetime:
    timezone = ZoneInfo(timezone_name)
    if now is None:
        return datetime.now(timezone)
    return now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)


def _context_facts(question: str, intent_kinds: list[str], now: datetime | None, timezone_name: str) -> list[dict[str, Any]]:
    text = normalize_question_text(question)
    parity = extract_time_context(question)
    if parity:
        return [{
            "fact_id": "query-time-parity",
            "predicate": "calendar_parity",
            "value": parity,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        }]

    mention = TIME_MENTION_RE.search(text)
    if mention:
        if mention.group(0) in {"bay gio", "hien tai", "luc nay"}:
            current = _now_in_zone(now, timezone_name)
            return [{
                "fact_id": "query-observed-at",
                "predicate": "observed_at",
                "value": current.isoformat(),
                "status": "resolved",
                "source": "system_clock",
                "raw_text": question,
                "confidence": "high",
                "metadata": {"resolution": "user_referred_to_now", "timezone": timezone_name},
            }]
        return [{
            "fact_id": "query-time-reference",
            "predicate": "temporal_reference",
            "value": None,
            "status": "needs_resolution",
            "source": "user_query",
            "raw_text": question,
            "confidence": "low",
            "metadata": {"unresolved_expression": mention.group(0)},
        }]

    if "CHECK_PERMISSION" in intent_kinds:
        current = _now_in_zone(now, timezone_name)
        return [{
            "fact_id": "query-observed-at",
            "predicate": "observed_at",
            "value": current.isoformat(),
            "status": "resolved",
            "source": "system_clock",
            "raw_text": None,
            "confidence": "high",
            "metadata": {"resolution": "default_current", "timezone": timezone_name},
        }]
    return []


def _intent_kinds(question: str, action: str) -> list[str]:
    text = normalize_question_text(question)
    found = [kind for kind, phrases in INTENT_PATTERNS if any(_contains(text, p) for p in phrases)]
    if action != "unknown" and any(_contains(text, p) for p in PERMISSION_TERMS):
        found.append("CHECK_PERMISSION")
    if not found:
        found = ["UNKNOWN"]
    return list(dict.fromkeys(found))


def _answer_spec(kind: str) -> dict[str, Any]:
    answer_types = {
        "CHECK_PERMISSION": "decision_with_explanation",
        "QUERY_SIGN_MEANING": "sign_meaning",
        "ASK_CONDITION": "applicability_conditions",
        "ASK_PENALTY": "penalty_and_legal_basis",
        "QUERY_PROHIBITED_SUBJECTS": "prohibited_subjects",
        "QUERY_TIME_CONDITION": "time_condition",
        "QUERY_DIRECTION_SCOPE": "direction_scope",
        "QUERY_DISTANCE_SCOPE": "distance_scope",
        "IDENTIFY_SIGN": "sign_identity",
    }
    return {"kind": answer_types.get(kind, "clarification"), "include_legal_basis": True}


def parse_query_intent_deterministic(
    question: str,
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Ho_Chi_Minh",
) -> IntentParseResult:
    action_value = extract_action(question)
    vehicle_value = extract_vehicle(question)
    kinds = _intent_kinds(question, action_value)
    location = extract_location_context(question)
    action = IntentSlot(action_value, question if action_value != "unknown" else None,
                        "provided" if action_value != "unknown" else "not_stated",
                        "user_query")
    actor = IntentSlot(vehicle_value, question if vehicle_value != "unknown" else None,
                       "provided" if vehicle_value != "unknown" else "not_stated",
                       "user_query")
    target = IntentTarget(kind="scene" if location else "unresolved")
    intents = [
        Intent(
            intent_id=f"intent-{index + 1}",
            kind=kind,
            target=target,
            action=action,
            actor=actor,
            requested_answer=_answer_spec(kind),
            confidence="high" if kind != "UNKNOWN" else "low",
        )
        for index, kind in enumerate(kinds)
    ]
    if any(intent.kind == "CHECK_PERMISSION" for intent in intents):
        for intent in intents:
            if intent.kind == "CHECK_PERMISSION" and action_value != "unknown":
                intent.confidence = "high" if vehicle_value != "unknown" else "medium"
    return IntentParseResult(
        schema_version="2.0",
        query_intent=QueryIntent(
            schema_version="2.0",
            query_id="query-1",
            raw_query=question,
            intents=intents,
            parser={"method": "deterministic", "confidence": "medium"},
        ),
        context_facts=_context_facts(question, kinds, now, timezone_name),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I | re.S).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Intent parser response must be a JSON object")
    return payload


def _validate_llm_query(payload: dict[str, Any], question: str, now: datetime | None, timezone_name: str) -> IntentParseResult:
    parsed_intents = []
    raw_intents = payload.get("intents", [])
    for index, item in enumerate(raw_intents if isinstance(raw_intents, list) else []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "UNKNOWN").upper()
        if kind not in VALID_INTENTS:
            kind = "UNKNOWN"
        action_obj = item.get("action") if isinstance(item.get("action"), dict) else {}
        actor_obj = item.get("actor") if isinstance(item.get("actor"), dict) else {}
        action_value = str(action_obj.get("value") or "unknown").lower()
        actor_value = str(actor_obj.get("value") or "unknown").lower()
        if action_value not in VALID_ACTIONS:
            action_value = "unknown"
        if actor_value not in VALID_VEHICLES:
            actor_value = "unknown"
        target_obj = item.get("target") if isinstance(item.get("target"), dict) else {}
        target_kind = str(target_obj.get("kind") or "unresolved")
        if target_kind not in {"scene", "sign_observation", "sign_code", "unresolved"}:
            target_kind = "unresolved"
        target = IntentTarget(
            kind=target_kind,
            observation_ids=[str(x) for x in target_obj.get("observation_ids", []) if isinstance(x, str)],
            sign_codes=[str(x) for x in target_obj.get("sign_codes", []) if isinstance(x, str)],
        )
        requested = item.get("requested_answer") if isinstance(item.get("requested_answer"), dict) else _answer_spec(kind)
        parsed_intents.append(Intent(
            intent_id=str(item.get("intent_id") or f"intent-{index + 1}"),
            kind=kind,
            target=target,
            action=IntentSlot(action_value, action_obj.get("raw_text"), str(action_obj.get("status") or "not_stated"), "user_query"),
            actor=IntentSlot(actor_value, actor_obj.get("raw_text"), str(actor_obj.get("status") or "not_stated"), "user_query"),
            requested_answer=requested,
            confidence=normalize_confidence(item.get("confidence")),
            unresolved_items=[str(x) for x in item.get("unresolved_items", [])],
        ))

    if not parsed_intents:
        parsed_intents = parse_query_intent_deterministic(
            question, now=now, timezone_name=timezone_name
        ).query_intent.intents
    context_facts = payload.get("context_facts")
    if not isinstance(context_facts, list):
        context_facts = _context_facts(question, [item.kind for item in parsed_intents], now, timezone_name)
    else:
        context_facts = [
            item for item in context_facts
            if isinstance(item, dict) and isinstance(item.get("predicate"), str)
        ]
        if not any(item.get("predicate") in {"observed_at", "calendar_parity", "temporal_reference"} for item in context_facts):
            context_facts.extend(_context_facts(question, [item.kind for item in parsed_intents], now, timezone_name))
    return IntentParseResult(
        schema_version="2.0",
        query_intent=QueryIntent(
            schema_version="2.0",
            query_id=str(payload.get("query_id") or "query-1"),
            raw_query=question,
            intents=parsed_intents,
            parser={"method": "llm", "confidence": normalize_confidence(payload.get("confidence"))},
            unresolved_items=[str(x) for x in payload.get("unresolved_items", [])],
        ),
        context_facts=context_facts,
    )


def parse_query_intent_with_llm(
    question: str,
    llm_client: Any,
    temperature: float = 0.0,
    max_tokens: int = 500,
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Ho_Chi_Minh",
) -> IntentParseResult:
    prompt = (
        "Phân tích câu hỏi giao thông Việt Nam thành strict JSON, không thêm văn xuôi. "
        "Một câu hỏi có thể có nhiều intents. Intent chỉ mô tả yêu cầu; dữ kiện thời gian/vị trí "
        "đưa vào context_facts kèm source=user_query. Không suy đoán dữ kiện ảnh.\n"
        'Schema: {"schema_version":"2.0","intents":[{"intent_id":"intent-1",'
        '"kind":"CHECK_PERMISSION|QUERY_SIGN_MEANING|ASK_CONDITION|ASK_PENALTY|'
        'QUERY_PROHIBITED_SUBJECTS|QUERY_TIME_CONDITION|QUERY_DIRECTION_SCOPE|'
        'QUERY_DISTANCE_SCOPE|IDENTIFY_SIGN|UNKNOWN","target":{"kind":"scene|'
        'sign_observation|sign_code|unresolved","observation_ids":[],"sign_codes":[]},'
        '"action":{"value":"parking|stopping|turn_left|turn_right|entering|speed|circulation|crossing|overtaking|unknown",'
        '"raw_text":null,"status":"provided|not_stated"},"actor":{"value":"car|motorbike|truck|bus|'
        'bicycle|pedestrian|motor_vehicle|vehicle|unknown","raw_text":null,"status":"provided|not_stated"},'
        '"requested_answer":{"kind":"...","include_legal_basis":true},"confidence":"high|medium|low",'
        '"unresolved_items":[]}],"context_facts":[],"unresolved_items":[]}\n'
        f"Câu hỏi: {question}"
    )
    response = llm_client.generate_content(
        contents=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _validate_llm_query(_extract_json_object(str(response)), question, now, timezone_name)


def parse_query_intent(
    question: str,
    llm_client: Any | None = None,
    use_llm_fallback: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 500,
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Ho_Chi_Minh",
) -> IntentParseResult:
    parsed = parse_query_intent_deterministic(question, now=now, timezone_name=timezone_name)
    if all(intent.kind == "UNKNOWN" for intent in parsed.query_intent.intents) and llm_client and use_llm_fallback:
        try:
            return parse_query_intent_with_llm(
                question, llm_client, temperature, max_tokens,
                now=now, timezone_name=timezone_name,
            )
        except Exception as exc:
            parsed.parser["llm_fallback_error"] = str(exc)
    return parsed
