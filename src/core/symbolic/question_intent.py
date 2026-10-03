from __future__ import annotations

import json
import re
import unicodedata as ud
from typing import Any

from src.core.symbolic.sign_schema import QuestionIntent, normalize_confidence


VALID_INTENTS = {
    "CHECK_PERMISSION",
    "QUERY_SIGN_MEANING",
    "ASK_CONDITION",
    "ASK_PENALTY",
    # Legacy labels are retained for previously stored intent payloads.
    "QUERY_PROHIBITED_SUBJECTS",
    "QUERY_TIME_CONDITION",
    "QUERY_DIRECTION_SCOPE",
    "QUERY_DISTANCE_SCOPE",
    "UNKNOWN",
}

VALID_ACTIONS = {
    "parking",
    "stopping",
    "turn_left",
    "turn_right",
    "entering",
    "speed",
    "circulation",
    "unknown",
}

VALID_VEHICLES = {
    "car",
    "motorbike",
    "truck",
    "bus",
    "bicycle",
    "pedestrian",
    "motor_vehicle",
    "unknown",
}

INTENT_PRIORITY = (
    (
        "ASK_PENALTY",
        (
            "muc phat",
            "phat bao nhieu",
            "bi phat bao nhieu",
            "bao nhieu tien",
            "tien phat",
            "xu phat",
            "muc xu phat",
            "xu phat the nao",
            "vi pham se bi gi",
        ),
    ),
    (
        "ASK_CONDITION",
        (
            "thoi gian",
            "luc nao",
            "khi nao",
            "ngay nao",
            "hieu luc",
            "dieu kien gi",
            "truong hop nao",
            "ap dung khi nao",
            "bao nhieu km h",
            "toc do toi da",
            "gioi han toc do",
        ),
    ),
    ("QUERY_DIRECTION_SCOPE", ("huong nao", "chieu nao", "re huong", "ben nao", "phia nao")),
    ("QUERY_DISTANCE_SCOPE", ("bao xa", "khoang cach", "pham vi", "bao nhieu met")),
    (
        "QUERY_PROHIBITED_SUBJECTS",
        (
            "cam nhung phuong tien nao",
            "cam phuong tien nao",
            "loai xe nao bi cam",
            "xe nao bi cam",
            "cam xe nao",
        ),
    ),
    ("QUERY_SIGN_MEANING", ("y nghia gi", "co nghia gi", "bien nay la gi", "noi dung bien")),
)

ACTION_PATTERNS = (
    (
        "parking",
        (
            "do xe",
            "dau xe",
            "dung do",
            "dung, do",
            "duoc do",
            "phep do",
            "cam do",
            "bi cam do",
            "do tai",
            "do khong",
        ),
    ),
    ("stopping", ("dung xe", "tam dung", "dung tai")),
    ("turn_left", ("re trai", "queo trai", "quay dau sang trai")),
    ("turn_right", ("re phai", "queo phai")),
    ("entering", ("di vao", "vao duong", "luu thong vao", "duoc vao", "cam vao")),
    ("speed", ("toc do", "van toc", "gioi han toc do", "chay bao nhieu")),
    ("circulation", ("luu thong", "di qua", "duoc di", "duoc chay", "tham gia giao thong")),
)

VEHICLE_PATTERNS = (
    ("bus", ("xe buyt", "xe bus", "xe khach")),
    ("truck", ("xe tai", "o to tai", "xe cho hang", "container")),
    ("motorbike", ("xe may", "mo to", "xe gan may")),
    ("bicycle", ("xe dap", "xe dap dien")),
    ("pedestrian", ("nguoi di bo", "di bo", "bo hanh")),
    ("car", ("o to", "xe hoi", "xe con", "xe oto")),
    ("motor_vehicle", ("xe co gioi", "phuong tien co gioi")),
)

TIME_CONTEXT_PATTERNS = (
    ("odd_days", ("ngay le", "hom nay la ngay le", "vao ngay le")),
    ("even_days", ("ngay chan", "hom nay la ngay chan", "vao ngay chan")),
)

LOCATION_CONTEXT_PATTERNS = (
    ("current_sign_location", ("tai day", "tai vi tri nay", "tren doan duong nay")),
    ("sign_scope", ("trong pham vi bien", "trong khu vuc nay", "tren tuyen duong nay")),
    ("direction_scope", ("theo chieu dat bien", "theo huong dat bien")),
)

PERMISSION_TERMS = (
    "duoc phep",
    "co duoc",
    "duoc khong",
    "co bi cam",
    "bi cam khong",
    "co cam",
    "duoc",
    "cam",
)


def normalize_question_text(question: str) -> str:
    text = ud.normalize("NFD", question or "").lower()
    text = "".join(char for char in text if ud.category(char) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(pattern)}(?!\w)", text)
        for pattern in patterns
    )


def extract_action(question: str) -> str:
    normalized = normalize_question_text(question)
    for action, patterns in ACTION_PATTERNS:
        if _contains_any(normalized, patterns):
            return action
    return "unknown"


def extract_vehicle(question: str) -> str:
    normalized = normalize_question_text(question)
    candidates: list[tuple[int, int, str]] = []
    for vehicle, patterns in VEHICLE_PATTERNS:
        for pattern in patterns:
            match = re.search(rf"(?<!\w){re.escape(pattern)}(?!\w)", normalized)
            if match:
                candidates.append((match.start(), -len(pattern), vehicle))
                break
    if candidates:
        return sorted(candidates)[0][2]
    return "unknown"


def extract_time_context(question: str) -> str | None:
    normalized = normalize_question_text(question)
    matches = [
        time_context
        for time_context, patterns in TIME_CONTEXT_PATTERNS
        if _contains_any(normalized, patterns)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None
    return None


def extract_location_context(question: str) -> str | None:
    normalized = normalize_question_text(question)
    for location, patterns in LOCATION_CONTEXT_PATTERNS:
        if _contains_any(normalized, patterns):
            return location
    return None


def detect_intent(question: str, action: str | None = None) -> str:
    normalized = normalize_question_text(question)
    for intent, patterns in INTENT_PRIORITY:
        if _contains_any(normalized, patterns):
            return intent

    if action == "speed" and _contains_any(
        normalized,
        ("bao nhieu", "muc nao", "gioi han", "quy dinh toc do"),
    ):
        return "ASK_CONDITION"

    if action and action != "unknown" and _contains_any(normalized, PERMISSION_TERMS):
        return "CHECK_PERMISSION"
    return "UNKNOWN"


def parse_question_intent_deterministic(question: str) -> QuestionIntent:
    action = extract_action(question)
    vehicle = extract_vehicle(question)
    time_context = extract_time_context(question)
    location_context = extract_location_context(question)
    intent = detect_intent(question, action=action)
    confidence = "low"

    if intent == "CHECK_PERMISSION" and action != "unknown":
        confidence = "high" if vehicle != "unknown" else "medium"
    elif intent in {
        "QUERY_SIGN_MEANING",
        "ASK_CONDITION",
        "ASK_PENALTY",
        "QUERY_PROHIBITED_SUBJECTS",
        "QUERY_TIME_CONDITION",
        "QUERY_DIRECTION_SCOPE",
        "QUERY_DISTANCE_SCOPE",
    }:
        confidence = "high"

    return QuestionIntent(
        intent=intent,
        action=action,
        vehicle=vehicle,
        time_context=time_context,
        location_context=location_context,
        raw_question=question,
        confidence=confidence,
        source="keyword",
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Question intent parser response must be a JSON object")
    return parsed


def _validated_intent_from_payload(
    payload: dict[str, Any],
    question: str,
    source: str,
) -> QuestionIntent:
    intent = str(payload.get("intent") or "UNKNOWN").strip().upper()
    action = str(payload.get("action") or "unknown").strip().lower()
    vehicle = str(payload.get("vehicle") or "unknown").strip().lower()

    if intent not in VALID_INTENTS:
        intent = "UNKNOWN"
    if action not in VALID_ACTIONS:
        action = "unknown"
    if vehicle not in VALID_VEHICLES:
        vehicle = "unknown"

    return QuestionIntent(
        intent=intent,
        action=action,
        vehicle=vehicle,
        time_context=payload.get("time_context"),
        location_context=payload.get("location_context"),
        raw_question=question,
        confidence=normalize_confidence(payload.get("confidence")),
        source=source,
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "intent",
                "action",
                "vehicle",
                "time_context",
                "location_context",
                "confidence",
            }
        },
    )


def parse_question_intent_with_llm(
    question: str,
    llm_client: Any,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> QuestionIntent:
    prompt = (
        "Bạn là bộ phân tích câu hỏi pháp luật giao thông Việt Nam.\n"
        "Hãy chuyển câu hỏi thành strict JSON, không markdown, không thêm chữ ngoài JSON.\n"
        "Chỉ dùng các giá trị sau:\n"
        f"intent: {sorted(VALID_INTENTS)}\n"
        f"action: {sorted(VALID_ACTIONS)}\n"
        f"vehicle: {sorted(VALID_VEHICLES)}\n"
        "Schema:\n"
        "{"
        '"intent":"CHECK_PERMISSION|QUERY_SIGN_MEANING|ASK_CONDITION|ASK_PENALTY|QUERY_PROHIBITED_SUBJECTS|QUERY_TIME_CONDITION|QUERY_DIRECTION_SCOPE|QUERY_DISTANCE_SCOPE|UNKNOWN",'
        '"action":"parking|stopping|turn_left|turn_right|entering|speed|circulation|unknown",'
        '"vehicle":"car|motorbike|truck|bus|bicycle|pedestrian|motor_vehicle|unknown",'
        '"time_context":"odd_days|even_days|null",'
        '"location_context":"current_sign_location|sign_scope|direction_scope|null",'
        '"confidence":"high|medium|low"'
        "}\n\n"
        f"Câu hỏi: {question}"
    )
    response = llm_client.generate_content(
        contents=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = _extract_json_object(str(response))
    return _validated_intent_from_payload(payload, question=question, source="llm")


def parse_question_intent(
    question: str,
    llm_client: Any | None = None,
    use_llm_fallback: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> QuestionIntent:
    deterministic = parse_question_intent_deterministic(question)
    if (
        deterministic.intent != "UNKNOWN"
        or not llm_client
        or not use_llm_fallback
    ):
        return deterministic

    try:
        return parse_question_intent_with_llm(
            question=question,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        deterministic.metadata["llm_fallback_error"] = str(exc)
        return deterministic
