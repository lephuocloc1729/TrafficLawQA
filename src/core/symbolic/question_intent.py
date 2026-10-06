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
from src.core.symbolic.vehicle_taxonomy import ROAD_USER_ACTORS


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
    "parking", "stopping", "yielding", "turn_left", "turn_right", "u_turn", "going_straight",
    "entering", "exiting", "speed", "circulation", "crossing", "overtaking",
    "reversing", "lane_change", "using_horn", "following_distance",
    "passing_obstacle", "roundabout_circulation", "reduce_speed", "unknown",
}
VALID_VEHICLES = ROAD_USER_ACTORS

INTENT_PATTERNS = (
    ("ASK_PENALTY", ("muc phat", "phat bao nhieu", "bi phat bao nhieu", "bao nhieu tien", "tien phat", "xu phat", "vi pham se bi gi")),
    ("QUERY_PROHIBITED_SUBJECTS", ("cam nhung phuong tien nao", "cam phuong tien nao", "loai xe nao bi cam", "xe nao bi cam", "cam xe nao")),
    ("QUERY_DIRECTION_SCOPE", ("huong nao", "chieu nao", "re huong", "ben nao", "phia nao")),
    ("QUERY_DISTANCE_SCOPE", ("bao xa", "khoang cach", "pham vi", "bao nhieu met")),
    ("QUERY_TIME_CONDITION", ("thoi gian nao", "thoi diem nao", "hieu luc vao luc nao", "co hieu luc khi nao")),
    ("ASK_CONDITION", ("dieu kien gi", "truong hop nao", "ap dung khi nao", "toc do toi da", "gioi han toc do", "bao nhieu km h", "co phai dung", "phai dung", "bat buoc dung", "co can dung", "phai nhuong", "co phai nhuong", "nhuong duong", "phai bam coi", "bat buoc bam coi", "co can bam coi", "phai an coi", "bat buoc an coi", "co can an coi", "phai giam toc do", "co can giam toc do", "co phai giam toc do")),
    ("QUERY_SIGN_MEANING", ("y nghia gi", "co nghia gi", "bien nay la gi", "noi dung bien")),
    ("IDENTIFY_SIGN", ("nhan dien bien", "xac dinh bien", "bien nao trong hinh", "ma bien gi")),
)
ACTION_PATTERNS = (
    ("parking", ("do xe", "dau xe", "dung do", "duoc do", "phep do", "cam do", "do tai", "do khong")),
    ("stopping", ("dung xe", "tam dung", "dung tai", "phai dung", "bat buoc dung", "co can dung", "co phai dung")),
    ("yielding", ("nhuong duong", "nhuong quyen di truoc", "phai nhuong", "co phai nhuong")),
    ("turn_left", ("re trai", "queo trai", "quay dau sang trai")),
    ("turn_right", ("re phai", "queo phai")),
    ("u_turn", ("quay dau xe", "quay dau")),
    ("going_straight", ("di thang", "chay thang")),
    ("entering", ("di vao", "vao duong", "luu thong vao", "duoc vao", "cam vao")),
    ("exiting", ("di ra", "ra khoi duong", "roi khoi duong")),
    ("reduce_speed", ("giam toc do", "chay cham lai", "giam van toc")),
    ("speed", ("toc do", "van toc", "gioi han toc do", "chay bao nhieu", "chay cham", "chay duoi", "chay thap hon")),
    ("roundabout_circulation", ("chay vong xuyen", "di vong xuyen", "chay vong theo dao", "chay vong quanh dao", "di vong quanh dao", "chay theo huong mui ten")),
    ("passing_obstacle", ("di vong", "vong ben trai", "vong ben phai", "vong qua chuong ngai vat", "di qua ben trai chuong ngai vat", "di qua ben phai chuong ngai vat", "di qua chuong ngai vat", "tranh chuong ngai vat")),
    ("crossing", ("qua duong", "sang duong", "cat ngang")),
    ("circulation", ("luu thong", "di qua", "duoc di", "duoc chay", "tham gia giao thong")),
    ("overtaking", ("vuot xe", "vuot phuong tien")),
    ("reversing", ("lui xe", "di lui", "chay lui")),
    ("lane_change", ("chuyen lan", "doi lan")),
    ("using_horn", ("bam coi", "an coi", "su dung coi", "dung coi")),
    ("following_distance", ("cu ly giua hai xe", "khoang cach giua hai xe", "giu khoang cach")),
    ("following_distance", ("cu ly giua hai xe", "khoang cach giua hai xe", "giu khoang cach")),
)
SPEED_VALUE_RE = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:km\s*/\s*h|kmh|kph)\b", re.I)
VEHICLE_HEIGHT_RE = re.compile(
    r"(?<!\w)(?:xe(?:\s+[a-z0-9]+){0,3}\s+(?:cao|co\s+chieu\s+cao)|"
    r"chieu\s+cao(?:\s+(?:cua\s+)?xe)?)\s*(?:la|=|:)?\s*"
    r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met)\b",
    re.I,
)
VEHICLE_WIDTH_RE = re.compile(
    r"(?<!\w)(?:xe(?:\s+[a-z0-9]+){0,3}\s+(?:rong|co\s+chieu\s+ngang)|"
    r"chieu\s+ngang(?:\s+(?:cua\s+)?xe)?|chieu\s+rong(?:\s+(?:cua\s+)?xe)?)"
    r"\s*(?:la|=|:)?\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met)\b",
    re.I,
)
VEHICLE_LENGTH_RE = re.compile(
    r"(?<!\w)(?:xe(?:\s+[a-z0-9]+){0,3}\s+dai|"
    r"chieu\s+dai(?:\s+(?:cua\s+)?xe)?|do\s+dai(?:\s+(?:cua\s+)?xe)?)"
    r"\s*(?:la|=|:)?\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met)\b",
    re.I,
)
VEHICLE_COMBINATION_LENGTH_RE = re.compile(
    r"(?<!\w)(?:tong\s+chieu\s+dai\s+(?:doan\s+xe|xe)|"
    r"chieu\s+dai\s+doan\s+xe|doan\s+xe\s+dai)"
    r"\s*(?:la|=|:)?\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met)\b",
    re.I,
)
SIGN_CODE_RE = re.compile(
    r"(?<![\w])((?:R[.,]?E|S[.,]?[GH]|DP|IE|P|W|R|I|S)[.,]?\s*\d{1,3}[a-z]?)(?!\w)",
    re.I,
)
VEHICLE_PATTERNS = (
    ("priority_vehicle", ("xe uu tien",)),
    ("emergency_vehicle", ("xe cuu hoa", "xe chua chay", "xe cap cuu")),
    ("taxi", ("xe oto taxi", "o to taxi", "xe taxi", "taxi")),
    ("hand_cart", ("xe nguoi keo", "xe day tay", "xe keo tay")),
    ("cargo_bicycle", ("xe dap tho", "xe tho hang", "xe dap cho hang")),
    ("motorized_tricycle", ("xe lam", "xich lo may", "xe loi may")),
    ("non_motorized_tricycle", ("xe ba banh khong dong co", "xich lo", "xe xich lo", "xe loi dap")),
    ("human_pushed_vehicle", ("xe nguoi day", "xe day nguoi")),
    ("animal_drawn_vehicle", ("xe vat nuoi keo", "xe keo vat nuoi", "xe keo bo", "xe keo ngua")),
    ("pack_animal", ("vat nuoi cho hang", "cho hang tren lung vat nuoi")),
    ("child_stroller", ("xe noi tre em", "xe day em be")),
    ("wheelchair", ("xe lan dung cho nguoi khuyet tat", "xe lan")),
    ("mobility_aid", ("phuong tien cua nguoi khuyet tat",)),
    ("tractor_unit", ("xe o to dau keo", "xe dau keo")),
    ("cong_nong_vehicle", ("xe cong nong", "cong nong")),
    ("semi_trailer", ("so mi ro mooc", "xe so mi ro mooc", "so mi romoc", "xe ban romoc")),
    ("trailer", ("romoc", "ro mooc")),
    ("motorized_tricycle", ("xe ba banh co dong co", "xe ba gac may")),
    ("electric_bicycle", ("xe dap dien",)),
    ("four_wheel_goods_vehicle", ("xe cho hang bon banh gan dong co", "xe cho hang 4 banh gan dong co")),
    ("four_wheel_passenger_vehicle", ("xe cho nguoi bon banh gan dong co", "xe cho nguoi 4 banh gan dong co")),
    ("special_use_vehicle", ("xe may chuyen dung", "xe chuyen dung")),
    ("construction_machine", ("xe may thi cong", "may thi cong")),
    ("agricultural_forestry_machine", (
        "xe may nong nghiep", "xe may lam nghiep", "may nong nghiep", "may lam nghiep",
    )),
    ("special_function_machine", ("xe may chuc nang dac biet", "may chuc nang dac biet")),
    ("defense_security_special_vehicle", (
        "xe dac chung quoc phong", "xe dac chung an ninh",
    )),
    ("tractor", ("may keo", "xe nong nghiep")),
    ("moped", ("xe gan may", "xe dap may")),
    ("bus", ("xe buyt", "xe bus", "xe khach", "o to khach")),
    ("truck", ("xe tai", "o to tai", "xe cho hang", "container")),
    ("motorcycle_or_moped", ("xe may",)),
    ("motorbike", ("mo to",)),
    ("bicycle", ("xe dap",)),
    ("pedestrian", ("nguoi di bo", "di bo", "bo hanh", "nguoi dat xe dap", "nguoi dat xe dap tho")),
    ("car", ("o to", "xe hoi", "xe con", "xe oto", "taxi")),
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
SIGN_POSITION_PHRASES = (
    "tai vi tri dat bien", "den vi tri dat bien", "ngay tai bien nay", "o vi tri dat bien",
)
HAZARDOUS_CARGO_PHRASE = "cho hang nguy hiem"


def normalize_question_text(question: str) -> str:
    text = ud.normalize("NFD", question or "").lower()
    text = "".join(char for char in text if ud.category(char) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _normalize_with_numeric_punctuation(question: str) -> str:
    text = ud.normalize("NFD", question or "").lower()
    text = "".join(char for char in text if ud.category(char) != "Mn").replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.,:=]+", " ", text)).strip()


def _measurement_readings(question: str, pattern: re.Pattern[str]) -> list[float]:
    return [
        float(match.group(1).replace(",", "."))
        for match in pattern.finditer(_normalize_with_numeric_punctuation(question))
    ]


def _vehicle_height_readings(question: str) -> list[float]:
    return _measurement_readings(question, VEHICLE_HEIGHT_RE)


def _trailer_configuration(question: str) -> str | None:
    text = normalize_question_text(question)
    if any(_contains(text, phrase) for phrase in ("so mi ro mooc", "semi trailer")):
        return "semi_trailer"
    if _contains(text, "ro mooc") and any(_contains(text, phrase) for phrase in ("keo", "keo theo")):
        return "towing_trailer"
    return None


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def extract_action(question: str) -> str:
    text = normalize_question_text(question)
    if SPEED_VALUE_RE.search(question or ""):
        return "speed"
    return next((action for action, phrases in ACTION_PATTERNS if any(_contains(text, p) for p in phrases)), "unknown")


def extract_vehicle_mentions(question: str) -> list[str]:
    text = normalize_question_text(question)
    # QCVN B.40 does not define which vehicle classes are "similar" to a
    # cong-nong vehicle. Do not classify such comparisons as the named class.
    if "tuong tu" in text and "cong nong" in text:
        return []
    if "loai xe tuong tu" in text:
        return []
    hits = []
    for vehicle, phrases in VEHICLE_PATTERNS:
        for phrase in phrases:
            match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
            if match:
                hits.append((match.start(), match.end(), -len(phrase), vehicle))
                break
    ordered: list[str] = []
    accepted_spans: list[tuple[int, int]] = []
    for start, end, _, vehicle in sorted(hits, key=lambda item: (item[0], item[2], item[1])):
        # Specific legal classes (truck, bus, special-use machine) overlap
        # generic aliases such as "ô tô" or "xe máy". Do not turn one noun
        # phrase into several conflicting actor classes; separate coordinated
        # mentions have non-overlapping spans and are retained.
        if any(start < accepted_end and accepted_start < end for accepted_start, accepted_end in accepted_spans):
            continue
        accepted_spans.append((start, end))
        if vehicle not in ordered:
            ordered.append(vehicle)
    return ordered


def extract_vehicle(question: str) -> str:
    mentions = extract_vehicle_mentions(question)
    return mentions[0] if mentions else "unknown"


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


def extract_pedestrian_location(question: str) -> str | None:
    """Extract only explicit sidewalk/road-shoulder locations; never guess from scene context."""
    text = normalize_question_text(question)
    locations = []
    for phrase, value in (("via he", "sidewalk"), ("le duong", "road_shoulder")):
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
        if not match:
            continue
        prefix = text[max(0, match.start() - 40):match.start()]
        if re.search(r"\bkhong(?:\s+\w+){0,4}\s*$", prefix):
            continue
        locations.append(value)
    return locations[0] if len(set(locations)) == 1 else None


def extract_obstacle_passing_side(question: str) -> str | None:
    """Extract an explicitly stated route side; never infer it from sign identity."""
    text = normalize_question_text(question)
    left_phrases = ("vong sang trai", "vong ben trai", "di ben trai chuong ngai vat", "qua ben trai chuong ngai vat")
    right_phrases = ("vong sang phai", "vong ben phai", "di ben phai chuong ngai vat", "qua ben phai chuong ngai vat")
    left = any(_contains(text, phrase) for phrase in left_phrases)
    right = any(_contains(text, phrase) for phrase in right_phrases)
    if _contains(text, "ben trai") and _contains(text, "ben phai"):
        return None
    if left == right:
        return None
    return "left" if left else "right"


def extract_roundabout_arrow_relation(question: str) -> bool | None:
    """Return only an explicitly stated relation to the sign's arrow."""
    text = normalize_question_text(question)
    follows = any(_contains(text, phrase) for phrase in (
        "theo huong mui ten", "cung chieu mui ten", "dung chieu mui ten",
    ))
    opposes = any(_contains(text, phrase) for phrase in (
        "nguoc huong mui ten", "nguoc chieu mui ten",
    ))
    if follows == opposes:
        return None
    return follows


def extract_hazardous_cargo_status(question: str) -> bool | None:
    """Extract cargo hazard status only from an explicit user statement."""
    text = normalize_question_text(question)
    if not _contains(text, HAZARDOUS_CARGO_PHRASE):
        return None
    phrase_start = text.find(HAZARDOUS_CARGO_PHRASE)
    prefix = text[max(0, phrase_start - 35):phrase_start]
    if re.search(r"\bkhong(?:\s+\w+){0,2}\s*$", prefix):
        return False
    return True


def _now_in_zone(now: datetime | None, timezone_name: str) -> datetime:
    timezone = ZoneInfo(timezone_name)
    if now is None:
        return datetime.now(timezone)
    return now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)


def _context_facts(question: str, intent_kinds: list[str], now: datetime | None, timezone_name: str) -> list[dict[str, Any]]:
    text = normalize_question_text(question)
    facts: list[dict[str, Any]] = []
    height_readings = _vehicle_height_readings(question)
    if height_readings:
        distinct_values = set(height_readings)
        unambiguous = len(distinct_values) == 1
        facts.append({
            "fact_id": "query-vehicle-height",
            "predicate": "vehicle_height_m",
            "value": next(iter(distinct_values)) if unambiguous else None,
            "status": "provided" if unambiguous else "ambiguous",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high" if unambiguous else "low",
            "metadata": {
                "unit": "m",
                "extraction": "explicit_vehicle_height_phrase",
                "conflicting_values": sorted(distinct_values) if not unambiguous else [],
                "scope_note": "user-stated vehicle height; not inferred from vehicle class",
            },
        })
    for predicate, pattern in (
        ("vehicle_width_m", VEHICLE_WIDTH_RE),
        ("vehicle_length_m", VEHICLE_LENGTH_RE),
        ("vehicle_combination_length_m", VEHICLE_COMBINATION_LENGTH_RE),
    ):
        readings = _measurement_readings(question, pattern)
        if not readings:
            continue
        distinct_values = set(readings)
        unambiguous = len(distinct_values) == 1
        facts.append({
            "fact_id": f"query-{predicate.replace('_', '-')}",
            "predicate": predicate,
            "value": next(iter(distinct_values)) if unambiguous else None,
            "status": "provided" if unambiguous else "ambiguous",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high" if unambiguous else "low",
            "metadata": {
                "unit": "m",
                "extraction": "explicit_vehicle_dimension_phrase",
                "conflicting_values": sorted(distinct_values) if not unambiguous else [],
                "scope_note": "user-stated vehicle dimension; not inferred from vehicle class",
            },
        })
    trailer_configuration = _trailer_configuration(question)
    if trailer_configuration:
        facts.append({
            "fact_id": "query-vehicle-trailer-configuration",
            "predicate": "vehicle_trailer_configuration",
            "value": trailer_configuration,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
            "metadata": {"extraction": "explicit_trailer_configuration_phrase"},
        })
    parity = extract_time_context(question)
    if parity:
        facts.append({
            "fact_id": "query-time-parity",
            "predicate": "calendar_parity",
            "value": parity,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        })

    mention = TIME_MENTION_RE.search(text) if not parity else None
    if mention:
        if mention.group(0) in {"bay gio", "hien tai", "luc nay"}:
            current = _now_in_zone(now, timezone_name)
            facts.append({
                "fact_id": "query-observed-at",
                "predicate": "observed_at",
                "value": current.isoformat(),
                "status": "resolved",
                "source": "system_clock",
                "raw_text": question,
                "confidence": "high",
                "metadata": {"resolution": "user_referred_to_now", "timezone": timezone_name},
            })
        else:
            facts.append({
                "fact_id": "query-time-reference",
                "predicate": "temporal_reference",
                "value": None,
                "status": "needs_resolution",
                "source": "user_query",
                "raw_text": question,
                "confidence": "low",
                "metadata": {"unresolved_expression": mention.group(0)},
            })

    if not parity and not mention and "CHECK_PERMISSION" in intent_kinds:
        current = _now_in_zone(now, timezone_name)
        facts.append({
            "fact_id": "query-observed-at",
            "predicate": "observed_at",
            "value": current.isoformat(),
            "status": "resolved",
            "source": "system_clock",
            "raw_text": None,
            "confidence": "high",
            "metadata": {"resolution": "default_current", "timezone": timezone_name},
        })

    pedestrian_location = extract_pedestrian_location(question)
    if pedestrian_location:
        facts.append({
            "fact_id": "query-pedestrian-location",
            "predicate": "pedestrian_location",
            "value": pedestrian_location,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        })
    obstacle_side = extract_obstacle_passing_side(question)
    if obstacle_side:
        facts.append({
            "fact_id": "query-obstacle-passing-side",
            "predicate": "intended_obstacle_passing_side",
            "value": obstacle_side,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        })
    arrow_relation = extract_roundabout_arrow_relation(question)
    if arrow_relation is not None:
        facts.append({
            "fact_id": "query-roundabout-arrow-relation",
            "predicate": "intended_roundabout_route_follows_arrow",
            "value": arrow_relation,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        })
    if any(_contains(text, phrase) for phrase in SIGN_POSITION_PHRASES):
        facts.append({
            "fact_id": "query-vehicle-at-sign-position",
            "predicate": "vehicle_at_r309_sign_position",
            "value": True,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        })
    hazardous_cargo = extract_hazardous_cargo_status(question)
    if hazardous_cargo is not None:
        facts.append({
            "fact_id": "query-hazardous-cargo-status",
            "predicate": "hazardous_cargo_status",
            "value": hazardous_cargo,
            "status": "provided",
            "source": "user_query",
            "raw_text": question,
            "confidence": "high",
        })
    return facts


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


def _requested_answer(kind: str, action: str, question: str) -> dict[str, Any]:
    result = _answer_spec(kind)
    match = SPEED_VALUE_RE.search(question or "") if kind == "CHECK_PERMISSION" and action == "speed" else None
    if match:
        number = match.group(1).replace(",", ".")
        result["requested_speed_kmh"] = float(number) if "." in number else int(number)
    height_readings = _vehicle_height_readings(question)
    if kind == "CHECK_PERMISSION" and height_readings and len(set(height_readings)) == 1:
        result["requested_vehicle_height_m"] = height_readings[0]
    for key, pattern in (
        ("requested_vehicle_width_m", VEHICLE_WIDTH_RE),
        ("requested_vehicle_length_m", VEHICLE_LENGTH_RE),
        ("requested_vehicle_combination_length_m", VEHICLE_COMBINATION_LENGTH_RE),
    ):
        readings = _measurement_readings(question, pattern)
        if readings and len(set(readings)) == 1 and kind == "CHECK_PERMISSION":
            result[key] = readings[0]
    return result


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
    explicit_sign_codes = list(dict.fromkeys(match.group(1).strip() for match in SIGN_CODE_RE.finditer(question or "")))
    target = (
        IntentTarget(kind="sign_code", sign_codes=explicit_sign_codes)
        if explicit_sign_codes
        else IntentTarget(kind="scene" if location else "unresolved")
    )
    intents = [
        Intent(
            intent_id=f"intent-{index + 1}",
            kind=kind,
            target=target,
            action=action,
            actor=actor,
            requested_answer=_requested_answer(kind, action_value, question),
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
        actor_status = str(actor_obj.get("status") or "not_stated")
        if actor_value not in VALID_VEHICLES:
            # Keep new vehicle labels observable rather than silently erasing them.
            actor_value = re.sub(r"[^a-z0-9]+", "_", normalize_question_text(actor_value)).strip("_") or "unknown"
            actor_status = "needs_resolution" if actor_value != "unknown" else "not_stated"
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
            actor=IntentSlot(actor_value, actor_obj.get("raw_text"), actor_status, "user_query"),
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
    vehicle_enum = "|".join(sorted(VALID_VEHICLES))
    prompt = (
        "Phân tích câu hỏi giao thông Việt Nam thành strict JSON, không thêm văn xuôi. "
        "Một câu hỏi có thể có nhiều intents. Intent chỉ mô tả yêu cầu; dữ kiện thời gian/vị trí "
        "đưa vào context_facts kèm source=user_query. Không suy đoán dữ kiện ảnh.\n"
        'Schema: {"schema_version":"2.0","intents":[{"intent_id":"intent-1",'
        '"kind":"CHECK_PERMISSION|QUERY_SIGN_MEANING|ASK_CONDITION|ASK_PENALTY|'
        'QUERY_PROHIBITED_SUBJECTS|QUERY_TIME_CONDITION|QUERY_DIRECTION_SCOPE|'
        'QUERY_DISTANCE_SCOPE|IDENTIFY_SIGN|UNKNOWN","target":{"kind":"scene|'
        'sign_observation|sign_code|unresolved","observation_ids":[],"sign_codes":[]},'
        '"action":{"value":"parking|stopping|yielding|turn_left|turn_right|u_turn|going_straight|entering|exiting|speed|reduce_speed|circulation|crossing|overtaking|reversing|lane_change|using_horn|following_distance|passing_obstacle|roundabout_circulation|unknown",'
        '"raw_text":null,"status":"provided|not_stated"},"actor":{"value":"'
        + vehicle_enum + '",'
        '"raw_text":null,"status":"provided|not_stated"},'
        '"requested_answer":{"kind":"...","include_legal_basis":true,"requested_speed_kmh":null},"confidence":"high|medium|low",'
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
