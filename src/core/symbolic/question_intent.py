from __future__ import annotations

import json
import re
import unicodedata as ud
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.core.symbolic.intent_schema import (
    Intent,
    IntentParseResult,
    IntentSlot,
    IntentTarget,
    QueryIntent,
    VALID_INTENT_KINDS,
    normalize_intent_kind,
)
from src.core.symbolic.sign_schema import normalize_confidence


VALID_INTENTS = VALID_INTENT_KINDS
VALID_ACTIONS = {
    "parking", "stopping", "yielding", "turn_left", "turn_right", "u_turn", "going_straight",
    "entering", "exiting", "speed", "circulation", "crossing", "overtaking",
    "reversing", "lane_change", "using_horn", "following_distance",
    "passing_obstacle", "roundabout_circulation", "reduce_speed", "unknown",
}
VALID_VEHICLES = frozenset({
    "motorcycle", "moped", "motorcycle_or_moped", "automobile", "passenger_car", "bus", "truck",
    "tractor_unit", "other_motor_vehicle", "non_motor_vehicle", "pedestrian", "unknown",
})
ACTOR_VALUE_ALIASES = {
    "motorbike": "motorcycle", "motorcycle": "motorcycle",
    "moped": "moped", "motorcycle_or_moped": "motorcycle_or_moped",
    "car": "automobile", "taxi": "automobile",
    "four_wheel_passenger_vehicle": "passenger_car",
    "four_wheel_goods_vehicle": "truck", "truck": "truck", "bus": "bus",
    "tractor_unit": "tractor_unit",
    "motor_vehicle": "other_motor_vehicle", "motorized_tricycle": "other_motor_vehicle",
    "trailer": "other_motor_vehicle", "semi_trailer": "other_motor_vehicle",
    "tractor_towed_trailer": "other_motor_vehicle", "priority_vehicle": "other_motor_vehicle",
    "emergency_vehicle": "other_motor_vehicle",
    "vehicle": "unknown",
    "bicycle": "non_motor_vehicle", "bicycle_machine": "non_motor_vehicle",
    "electric_bicycle": "non_motor_vehicle", "cargo_bicycle": "non_motor_vehicle",
    "non_motor_vehicle": "non_motor_vehicle", "non_motorized_tricycle": "non_motor_vehicle",
    "pedicab": "non_motor_vehicle", "animal_drawn_vehicle": "non_motor_vehicle",
    "hand_cart": "non_motor_vehicle", "human_pushed_vehicle": "non_motor_vehicle",
    "pack_animal": "non_motor_vehicle", "child_stroller": "non_motor_vehicle",
    "wheelchair": "non_motor_vehicle", "mobility_aid": "non_motor_vehicle",
    "special_use_machine": "other_motor_vehicle", "special_use_vehicle": "other_motor_vehicle",
    "construction_machine": "other_motor_vehicle",
    "agricultural_forestry_machine": "other_motor_vehicle", "tractor": "other_motor_vehicle",
    "cong_nong_vehicle": "other_motor_vehicle", "special_function_machine": "other_motor_vehicle",
    "defense_security_special_vehicle": "other_motor_vehicle",
    "pedestrian": "pedestrian",
    "unknown": "unknown",
    "similar_motor_vehicle": "unknown",
    "similar_non_motor_vehicle": "unknown",
}


def normalize_intent_actor(value: str | None) -> str:
    actor = str(value or "unknown").strip().lower()
    if actor in ACTOR_VALUE_ALIASES:
        return ACTOR_VALUE_ALIASES[actor]
    if actor in VALID_VEHICLES:
        return actor
    return "unknown"


def classify_intent_actor(question: str, detailed_actor: str | None = None) -> str:
    """Map wording/detail to the smaller, user-facing actor vocabulary."""
    text = normalize_question_text(question)
    if re.search(r"(?<!\w)(?:xe con|o to con)(?!\w)", text):
        return "passenger_car"
    return normalize_intent_actor(detailed_actor or extract_vehicle(question))

INTENT_PATTERNS = (
    ("UNKNOWN", (
        "muc phat", "phat bao nhieu", "bi phat bao nhieu", "bao nhieu tien", "tien phat",
        "xu phat", "vi pham se bi gi", "phat the nao", "bi phat the nao",
    )),
    ("INFORMATION", (
        "cam nhung phuong tien nao", "cam phuong tien nao", "loai xe nao bi cam", "xe nao bi cam", "cam xe nao",
        "phuong tien nao", "loai phuong tien nao", "ap dung cho loai phuong tien nao",
        "huong nao", "chieu nao", "re huong", "ben nao", "phia nao",
        "bao xa", "khoang cach", "pham vi", "bao nhieu met",
        "thoi gian nao", "thoi diem nao", "hieu luc vao luc nao", "co hieu luc khi nao",
        "toc do toi da", "gioi han toc do", "bao nhieu km h", "toc do bao nhieu", "van toc bao nhieu",
        "cao toc nao", "cao toc gi", "ten cao toc", "ten duong cao toc", "cao toc nay la gi", "ma cao toc",
        "ten duong nao", "bien chi duong", "di den dau",
        "dieu kien gi", "truong hop nao", "ap dung khi nao",
    )),
    ("INFORMATION", (
        "y nghia gi", "y nghia cua bien", "co nghia gi", "bien nay la gi", "bien nay la bien gi",
        "bien gi vay", "bien gi", "noi dung bien", "noi dung cua bien", "nhan dien bien", "nhan dien giup bien",
        "xac dinh bien", "bien nao trong hinh", "bien nao", "ma bien gi", "ma bien bao nay",
        "thong bao dieu gi", "noi len dieu gi", "loai bien bao nao", "ten goi cua bien",
        "thuoc loai bien nao", "bao hieu dieu gi", "hien thi bien bao nao",
    )),
)
INFORMATION_SCOPE_PHRASES = (
    "cam nhung phuong tien nao", "cam phuong tien nao", "loai xe nao bi cam", "xe nao bi cam", "cam xe nao",
    "phuong tien nao", "loai phuong tien nao", "ap dung cho loai phuong tien nao",
    "huong nao", "chieu nao", "re huong", "ben nao", "phia nao",
    "bao xa", "khoang cach", "pham vi", "bao nhieu met",
    "con cach", "cach ha noi", "cach sai gon", "cach tphcm",
    "thoi gian nao", "thoi diem nao", "hieu luc vao luc nao", "co hieu luc khi nao",
    "toc do toi da", "gioi han toc do", "bao nhieu km h", "toc do bao nhieu", "van toc bao nhieu",
)
ACTION_PATTERNS = (
    ("parking", ("do xe", "dau xe", "dung do", "duoc do", "phep do", "cam do", "do tai", "do khong")),
    ("stopping", ("dung xe", "tam dung", "dung tai", "dung don", "phai dung", "bat buoc dung", "co can dung", "co phai dung")),
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
    ("tricycle", ("xe ba banh",)),
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
MANDATORY_ACTION_TERMS = (
    "co phai dung", "phai dung", "bat buoc dung", "co can dung",
    "co phai nhuong", "phai nhuong", "co can nhuong",
    "phai bam coi", "bat buoc bam coi", "co can bam coi", "phai an coi",
    "bat buoc an coi", "co can an coi", "phai giam toc do", "co can giam toc do",
    "co phai giam toc do",
)
PERMISSION_TERMS = (
    "duoc phep", "co duoc", "duoc khong", "co bi cam khong", "bi cam khong",
    *MANDATORY_ACTION_TERMS,
)
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
    left_phrases = (
        "vong sang trai", "vong ben trai", "di ben trai chuong ngai vat",
        "qua ben trai chuong ngai vat", "chuong ngai vat ben trai",
        "chuong ngai vat ve ben trai", "tranh chuong ngai vat ben trai",
        "tranh chuong ngai vat ve ben trai",
    )
    right_phrases = (
        "vong sang phai", "vong ben phai", "di ben phai chuong ngai vat",
        "qua ben phai chuong ngai vat", "chuong ngai vat ben phai",
        "chuong ngai vat ve ben phai", "tranh chuong ngai vat ben phai",
        "tranh chuong ngai vat ve ben phai",
    )
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


def _resolve_query_datetime(
    normalized_question: str,
    now: datetime,
) -> tuple[datetime | None, str | None]:
    """Resolve explicit clock/date wording against the current zoned time."""
    if any(phrase in normalized_question for phrase in ("bay gio", "hien tai", "luc nay")):
        return now, "user_referred_to_now"

    target_date = now.date()
    date_expression: str | None = None
    if "ngay mai" in normalized_question:
        target_date += timedelta(days=1)
        date_expression = "tomorrow"
    elif "hom qua" in normalized_question:
        target_date -= timedelta(days=1)
        date_expression = "yesterday"
    elif "hom nay" in normalized_question:
        date_expression = "today"
    else:
        explicit_date = re.search(
            r"\b(\d{1,2})(?:[/-]|\s+)(\d{1,2})(?:(?:[/-]|\s+)(\d{2,4}))?\b",
            normalized_question,
        )
        if explicit_date:
            day, month = int(explicit_date.group(1)), int(explicit_date.group(2))
            year_text = explicit_date.group(3)
            year = int(year_text) if year_text else now.year
            if year < 100:
                year += 2000
            try:
                target_date = datetime(year, month, day).date()
                date_expression = explicit_date.group(0)
            except ValueError:
                return None, None

    clock_match = re.search(
        r"\b(\d{1,2})\s*(?:gio|h)(?:\s*(\d{1,2})\s*(?:phut|p))?\b",
        normalized_question,
    )
    if clock_match:
        hour = int(clock_match.group(1))
        minute = int(clock_match.group(2) or 0)
        if hour > 23 or minute > 59:
            return None, None
        resolved_time = time(hour, minute, tzinfo=now.tzinfo)
        return datetime.combine(target_date, resolved_time), "explicit_query_date_time"

    if date_expression:
        return datetime.combine(target_date, now.timetz()), "explicit_query_date"
    return None, None


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
        current = _now_in_zone(now, timezone_name)
        resolved_at, resolution = _resolve_query_datetime(text, current)
        if resolved_at is not None:
            facts.append({
                "fact_id": "query-observed-at",
                "predicate": "observed_at",
                "value": resolved_at.isoformat(),
                "status": "resolved",
                "source": "user_query",
                "raw_text": question,
                "confidence": "high",
                "metadata": {"resolution": resolution, "timezone": timezone_name},
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
    if "CHECK_PERMISSION" in intent_kinds and not any(
        fact.get("predicate") == "calendar_parity" for fact in facts
    ):
        observed = next((
            fact for fact in reversed(facts)
            if fact.get("predicate") == "observed_at" and fact.get("status") == "resolved"
        ), None)
        current = _now_in_zone(now, timezone_name)
        parity_date = current.date()
        if observed and isinstance(observed.get("value"), str):
            try:
                parity_date = datetime.fromisoformat(observed["value"]).date()
            except ValueError:
                pass
        facts.append({
            "fact_id": "query-default-calendar-parity",
            "predicate": "calendar_parity",
            "value": "odd_days" if parity_date.day % 2 else "even_days",
            "status": "derived",
            "source": "system_clock" if observed is None else "derived",
            "raw_text": None,
            "confidence": "high",
            "metadata": {
                "resolution": "default_current_date" if observed is None else "derived_from_observed_at",
                "timezone": timezone_name,
            },
        })
    return facts


def _intent_kinds(question: str, action: str) -> list[str]:
    text = normalize_question_text(question)
    found = [kind for kind, phrases in INTENT_PATTERNS if any(_contains(text, p) for p in phrases)]
    if "UNKNOWN" not in found and re.search(r"\bvi pham\b.*\bbi gi\b", text):
        found.append("UNKNOWN")
    asks_for_scope_information = any(_contains(text, p) for p in INFORMATION_SCOPE_PHRASES)
    asks_yes_no_permission = (
        "duoc khong" in text
        or ("co duoc" in text and "khong" in text)
        or "co bi cam khong" in text
        or "bi cam khong" in text
    )
    asks_for_mandatory_action = any(_contains(text, p) for p in MANDATORY_ACTION_TERMS)
    asks_for_penalty = any(
        _contains(text, phrase)
        for phrase in (
            "muc phat", "phat bao nhieu", "bi phat bao nhieu", "bao nhieu tien", "tien phat",
            "xu phat", "vi pham se bi gi", "phat the nao", "bi phat the nao",
        )
    ) or bool(re.search(r"\bvi pham\b.*\bbi gi\b", text))
    if (
        action != "unknown"
        and any(_contains(text, p) for p in PERMISSION_TERMS)
        and (not asks_for_scope_information or asks_for_mandatory_action or asks_yes_no_permission)
        and (not asks_for_penalty or asks_yes_no_permission)
    ):
        found.append("CHECK_PERMISSION")
    if not found:
        found = ["UNKNOWN"]
    return list(dict.fromkeys(found))


def _information_type(question: str) -> str | None:
    """Return a known hint when available; unmatched information stays untyped."""
    text = normalize_question_text(question)
    if any(term in text for term in (
        "y nghia gi", "y nghia cua bien", "co nghia gi", "bien nay la gi", "bien nay la bien gi",
        "bien gi vay", "bien gi", "noi dung bien", "noi dung cua bien", "nhan dien bien",
        "nhan dien giup bien", "xac dinh bien", "bien nao trong hinh", "bien nao", "ma bien gi",
        "ma bien bao nay", "thong bao dieu gi", "noi len dieu gi", "loai bien bao nao", "ten goi cua bien",
        "thuoc loai bien nao", "bao hieu dieu gi", "hien thi bien bao nao",
    )):
        return "sign_meaning"
    if any(term in text for term in (
        "cam nhung phuong tien nao", "cam phuong tien nao", "loai xe nao bi cam",
        "xe nao bi cam", "cam xe nao", "ap dung cho loai phuong tien nao",
    )):
        return "prohibited_subjects"
    if any(term in text for term in ("thoi gian", "luc nao", "ngay nao", "hieu luc khi nao")):
        return "time_condition"
    if any(term in text for term in ("huong nao", "chieu nao", "ben nao", "phia nao", "re huong")):
        return "direction_scope"
    if any(term in text for term in ("toc do", "van toc", "km h")):
        return "speed"
    if any(term in text for term in (
        "ten cao toc", "cao toc nao", "cao toc gi", "cao toc nay la gi", "ma cao toc",
        "ten duong cao toc", "ten duong nao",
    )):
        return "road_identity"
    if "bao xa" in text and any(term in text for term in ("con cach", "cach ha noi", "cach sai gon", "cach tphcm", "cach tp hcm")):
        return "destination_distance"
    if any(term in text for term in ("khoang cach", "pham vi", "bao xa", "bao nhieu met")):
        return "distance_scope"
    return None


def _answer_spec(kind: str) -> dict[str, Any]:
    answer_types = {
        "CHECK_PERMISSION": "decision_with_explanation",
        "INFORMATION": "traffic_sign_information",
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
    detailed_actor = extract_vehicle(question)
    vehicle_value = classify_intent_actor(question, detailed_actor)
    kinds = _intent_kinds(question, action_value)
    actor_status = "provided" if detailed_actor != "unknown" else "not_stated"
    if detailed_actor == "unknown" and "CHECK_PERMISSION" in kinds:
        # Permission questions without an explicit vehicle use the project's
        # requested default actor. Information questions remain unresolved.
        vehicle_value = "car"
        actor_status = "defaulted"
    location = extract_location_context(question)
    action = IntentSlot(action_value, question if action_value != "unknown" else None,
                        "provided" if action_value != "unknown" else "not_stated")
    actor = IntentSlot(vehicle_value, question if detailed_actor != "unknown" else None,
                       actor_status)
    explicit_sign_codes = list(dict.fromkeys(match.group(1).strip() for match in SIGN_CODE_RE.finditer(question or "")))
    normalized_question = normalize_question_text(question)
    refers_to_scene = location is not None or any(phrase in normalized_question for phrase in (
        "bien nay", "bien bao nay", "bien bao trong anh", "bien trong anh",
        "anh nay", "hinh nay", "trong hinh", "trong anh",
    ))
    target = (
        IntentTarget(kind="sign_code", sign_codes=explicit_sign_codes)
        if explicit_sign_codes
        else IntentTarget(kind="scene" if refers_to_scene else "unresolved")
    )
    intents = [
        Intent(
            intent_id=f"intent-{index + 1}",
            kind=kind,
            target=target,
            action=action,
            actor=actor,
            requested_answer=_requested_answer(kind, action_value, question),
            information_type=_information_type(question) if kind == "INFORMATION" else None,
        )
        for index, kind in enumerate(kinds)
    ]
    return IntentParseResult(
        schema_version="3.0",
        query_intent=QueryIntent(
            schema_version="3.0",
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
        raw_kind = str(item.get("kind") or "UNKNOWN").strip().upper()
        kind = normalize_intent_kind(item.get("kind"))
        action_payload = item.get("action")
        actor_payload = item.get("actor")
        action_obj = action_payload if isinstance(action_payload, dict) else {}
        actor_obj = actor_payload if isinstance(actor_payload, dict) else {}
        action_value = str(action_obj.get("value") if action_obj else action_payload or "unknown").lower()
        raw_actor_value = str(actor_obj.get("value") if actor_obj else actor_payload or "unknown").lower()
        actor_value = normalize_intent_actor(raw_actor_value)
        if action_value not in VALID_ACTIONS:
            action_value = "unknown"
        actor_status = str(actor_obj.get("status") or ("provided" if raw_actor_value != "unknown" else "not_stated"))
        actor_raw_text = actor_obj.get("raw_text") or (question if raw_actor_value != "unknown" else None)
        if actor_value == "unknown" and raw_actor_value != "unknown":
            actor_raw_text = actor_raw_text or raw_actor_value
            actor_status = "needs_resolution"
        target_payload = item.get("target")
        target_obj = target_payload if isinstance(target_payload, dict) else {}
        target_kind = str(target_obj.get("kind") if target_obj else target_payload or "unresolved")
        if target_kind not in {"scene", "sign_observation", "sign_code", "unresolved"}:
            target_kind = "unresolved"
        explicit_sign_codes = list(dict.fromkeys(
            match.group(1).strip() for match in SIGN_CODE_RE.finditer(question or "")
        ))
        if explicit_sign_codes and target_kind in {"scene", "sign_code"}:
            target_kind = "sign_code"
        target = IntentTarget(
            kind=target_kind,
            observation_ids=[str(x) for x in target_obj.get("observation_ids", []) if isinstance(x, str)],
            sign_codes=[str(x) for x in target_obj.get("sign_codes", []) if isinstance(x, str)] or (
                explicit_sign_codes
            ),
        )
        requested = item.get("requested_answer") if isinstance(item.get("requested_answer"), dict) else _answer_spec(kind)
        information_type = str(item.get("information_type") or "").strip().lower() or None
        if raw_kind in {"QUERY_SIGN", "QUERY_SIGN_MEANING", "IDENTIFY_SIGN"} and not information_type:
            information_type = "sign_meaning"
        parsed_intents.append(Intent(
            intent_id=str(item.get("intent_id") or f"intent-{index + 1}"),
            kind=kind,
            target=target,
            action=IntentSlot(action_value, action_obj.get("raw_text"), str(action_obj.get("status") or "not_stated")),
            actor=IntentSlot(actor_value, actor_raw_text, actor_status),
            requested_answer=requested,
            information_type=information_type,
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
        schema_version="3.0",
        query_intent=QueryIntent(
            schema_version="3.0",
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
        "Bạn là bộ phân loại ý định cho hệ thống hỏi đáp biển báo và luật giao thông Việt Nam. "
        "Hãy phân loại theo ý nghĩa câu hỏi, không yêu cầu câu hỏi phải chứa một cụm từ cố định. "
        "Một câu hỏi có thể có nhiều intents nếu thực sự hỏi nhiều việc.\n"
        "Định nghĩa nhãn:\n"
        "- CHECK_PERMISSION: hỏi một hành động có được phép/bị cấm hay có bắt buộc phải làm không; "
        "bao gồm cách hỏi khẩu ngữ hoặc diễn đạt gián tiếp.\n"
        "- INFORMATION: hỏi thông tin thể hiện bởi biển hoặc dữ liệu nhận diện, gồm nhận diện biển, ký hiệu/mã, "
        "ý nghĩa/thông điệp của biển, đối tượng xe, thời gian hiệu lực, "
        "hướng, tốc độ, khoảng cách, mã/tên đường và các thông tin cụ thể khác. information_type chỉ là gợi ý tùy chọn: "
        "nếu không khớp các giá trị liệt kê, vẫn chọn INFORMATION và bỏ qua trường information_type để LLM tự trả lời "
        "từ câu hỏi cùng context. Với câu hỏi nhận diện/ý nghĩa biển, đặt information_type=sign_meaning.\n"
        "- UNKNOWN: bao gồm câu hỏi ngoài phạm vi giao thông/biển báo, không thể xác định ý định, hoặc hỏi tiền phạt, "
        "mức xử phạt hay chế tài cho vi phạm. Câu hỏi lạ cách "
        "diễn đạt nhưng vẫn rõ nghĩa không được gán UNKNOWN chỉ vì không khớp từ khóa.\n"
        "Chỉ thêm trường liên quan đến kind: CHECK_PERMISSION dùng action, actor; INFORMATION có thể có "
        "information_type nếu khớp một giá trị đã liệt kê; nếu không, hãy bỏ trường đó. "
        "target chỉ thêm khi câu hỏi nhắm tới biển/cảnh cụ thể. "
        "Với cách gọi chung 'xe máy', dùng actor=motorcycle_or_moped, không tự chọn mô tô hay xe gắn máy.\n"
        "Intent mô tả điều người dùng muốn biết; không suy đoán dữ kiện từ ảnh. Dữ kiện thời gian/vị trí "
        "được nêu rõ đưa vào context_facts với source=user_query. Trả về JSON hợp lệ, không thêm văn xuôi.\n"
        'Schema: {"schema_version":"3.0","intents":[{"kind":"CHECK_PERMISSION|INFORMATION|UNKNOWN",'
        '"target":"scene|sign_observation|sign_code|unresolved",'
        '"action":"parking|stopping|yielding|turn_left|turn_right|u_turn|going_straight|entering|exiting|speed|reduce_speed|circulation|crossing|overtaking|reversing|lane_change|using_horn|following_distance|passing_obstacle|roundabout_circulation|unknown",'
        '"actor":"'
        + vehicle_enum + '","information_type":"sign_meaning|prohibited_subjects|time_condition|direction_scope|speed|distance_scope|destination_distance|road_identity",'
        '"requested_answer":{}}],'
        '"context_facts":[],"unresolved_items":[],"confidence":"high|medium|low"}\n'
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
    use_llm: bool | None = None,
) -> IntentParseResult:
    parsed = parse_query_intent_deterministic(question, now=now, timezone_name=timezone_name)
    # Keep the old option as a compatibility alias; the LLM is now the primary
    # classifier whenever it is enabled and a client is available.
    llm_enabled = use_llm_fallback if use_llm is None else use_llm
    if llm_client and llm_enabled:
        try:
            llm_parsed = parse_query_intent_with_llm(
                question, llm_client, temperature, max_tokens,
                now=now, timezone_name=timezone_name,
            )
            # Intent classification belongs to the LLM. Keep deterministic
            # extraction only as a source of structured slots the rule engine needs.
            _fill_missing_permission_slots(llm_parsed, parsed)
            return llm_parsed
        except Exception as exc:
            parsed.query_intent.parser["llm_classification_error"] = str(exc)
    return parsed


def _fill_missing_permission_slots(
    llm_result: IntentParseResult,
    deterministic_result: IntentParseResult,
) -> None:
    """Use deterministic extraction to fill slots, without overriding LLM intent labels."""
    fallback = next(iter(deterministic_result.query_intent.intents), None)
    if fallback is None:
        return
    for intent in llm_result.query_intent.intents:
        if intent.kind != "CHECK_PERMISSION":
            continue
        if intent.action.value == "unknown" and fallback.action.value != "unknown":
            intent.action = fallback.action
        if fallback.actor.status == "defaulted":
            intent.actor = fallback.actor
        elif intent.actor.value == "unknown" and fallback.actor.status == "provided":
            intent.actor = fallback.actor
        elif intent.actor.value != "unknown" and not intent.actor.raw_text:
            intent.actor.raw_text = deterministic_result.query_intent.raw_query
            intent.actor.status = "provided"
        requested = _requested_answer("CHECK_PERMISSION", fallback.action.value, deterministic_result.query_intent.raw_query)
        for key, value in requested.items():
            intent.requested_answer.setdefault(key, value)
