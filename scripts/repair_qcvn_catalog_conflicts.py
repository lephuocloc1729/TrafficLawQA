"""Apply reviewed-source extraction corrections to the QCVN draft catalog.

This script does not approve entries for automatic decisions. It records the
published wording, source references, and conflicting figure labels so the
team can review them without silently treating a typo as law.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
ANOMALY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_catalog_extraction_anomalies.json"
UNCODED_FIGURES_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_uncoded_figure_review.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
SOURCE_SHA256 = "b6b99d0a0aeb13fe79072831857ece15bb254d8fb7817e80e4a2739a0d2145ce"

R413_MEANINGS = {
    "R.413i": "Kết thúc làn đường dành cho xe ô tô khách.",
    "R.413j": "Kết thúc làn đường dành cho xe ô tô con.",
    "R.413k": "Kết thúc làn đường dành cho xe ô tô tải.",
    "R.413l": "Kết thúc làn đường dành cho xe máy.",
    "R.413m": "Kết thúc làn đường dành cho xe buýt.",
    "R.413n": "Kết thúc làn đường dành cho xe ô tô.",
    "R.413o": "Kết thúc làn đường dành cho xe máy và xe đạp.",
    "R.413p": "Kết thúc làn đường dành cho xe đạp.",
}
R413_EXCERPTS = {
    "R.413i": 'a) Khi kết thúc làn đường dành cho xe ô tô khách đi lại, đặt biển số R.413i " Kết thúc làn đường dành cho xe ô tô khách".',
    "R.413j": 'b) Khi kết thúc làn đường làn đường dành cho xe ô tô con đi lại, đặt biển số R.413j "Kết thúc làn đường dành cho xe ô tô con".',
    "R.413k": 'c) Khi kết thúc làn đường dành cho xe ô tô tải đi lại, đặt biển số R.413k "Kết thúc làn đường dành cho xe ô tô tải".',
    "R.413l": 'd) Khi kết thúc làn đường dành cho xe máy đi lại, đặt biển số R.413l "Kết thúc làn đường dành cho xe máy".',
    "R.413m": 'e) Khi kết thúc làn đường dành riêng cho xe buýt đi lại, đặt biển số R.413m "Kết thúc làn đường dành cho xe buýt".',
    "R.413n": 'f) Khi kết thúc làn đường dành cho xe ô tô đi lại, đặt biển số R.413n "Kết thúc làn đường dành cho xe ô tô".',
    "R.413o": 'g) Khi kết thúc làn đường dành cho xe máy và xe đạp đi lại, đặt biển số R.413o "Kết thúc làn đường dành cho xe máy và xe đạp".',
    "R.413p": 'h) Khi kết thúc làn đường dành cho đạp đi lại, đặt biển số R.413p "Kết thúc làn đường dành cho xe đạp".',
}
IE_FAMILY_CODES = {
    "IE.450", "IE.451", "IE.453", "IE.455", "IE.456", "IE.457", "IE.459",
    "IE.460", "IE.461", "IE.463", "IE.464", "IE.465", "IE.467", "IE.468", "IE.472",
    "R.302", "R.413", "R.415", "W.201",
}
IE_NAMES = {
    "IE.450a": "Khoảng cách đến nút giao với đường dẫn vào đường cao tốc",
    "IE.450b": "Sơ đồ đến đường dẫn vào đường cao tốc",
    "IE.451a": "Khoảng cách đến lối vào đường cao tốc",
    "IE.451b": "Lối vào đường cao tốc",
    "IE.452": "Bắt đầu đường cao tốc",
    "IE.453a": "Khoảng cách đến điểm kết thúc đường cao tốc",
    "IE.453b": "Điểm kết thúc đường cao tốc",
    "IE.453c": "Hết đường cao tốc",
    "IE.454": "Khoảng cách đến lối ra phía trước",
    "IE.455a": "Khoảng cách đến các lối ra tiếp theo",
    "IE.455b": "Khoảng cách đến các lối ra tiếp theo",
    "IE.456a": "Khoảng cách đến trạm dừng nghỉ",
    "IE.456b": "Hướng rẽ vào trạm dừng nghỉ",
    "IE.456c": "Lối vào trạm dừng nghỉ",
    "IE.457a": "Khoảng cách đến nơi đỗ xe trong trạm dừng nghỉ",
    "IE.457b": "Hướng rẽ đến bãi đỗ xe trong trạm dừng nghỉ",
    "IE.458": "Khoảng cách đến các trạm dừng nghỉ tiếp theo",
    "IE.459a": "Khoảng cách đến khu tham quan, du lịch",
    "IE.459b": "Hướng rẽ vào khu tham quan, du lịch",
    "IE.460a": "Cột lý trình kilômét",
    "IE.460b": "Cột lý trình kilômét",
    "IE.460c": "Cột lý trình kilômét",
    "IE.461a": "Khoảng cách đến khu dịch vụ công cộng, giải trí",
    "IE.461b": "Hướng rẽ đến khu dịch vụ công cộng, giải trí",
    "IE.461c": "Hướng rẽ vào khu dịch vụ công cộng",
    "IE.461d": "Hướng rẽ vào khu giải trí",
    "IE.462": "Tần số trên sóng radio",
    "IE.463a": "Khoảng cách đến công trình kiểm soát tải trọng xe",
    "IE.463b": "Hướng rẽ vào công trình kiểm soát tải trọng xe",
    "IE.463c": "Lối vào công trình kiểm soát tải trọng xe",
    "IE.464a": "Địa điểm và phương hướng ra một chiều",
    "IE.464b": "Hướng ra tại lối ra",
    "IE.465a": "Địa điểm và khoảng cách",
    "IE.465b": "Địa điểm và khoảng cách",
    "IE.466": "Sơ đồ lối ra một chiều",
    "IE.467a": "Vị trí nhập làn",
    "IE.467b": "Khoảng cách đến vị trí nhập làn",
    "IE.468a": "Chướng ngại vật phía trước – đi bên trái",
    "IE.468b": "Chướng ngại vật phía trước – đi một trong hai hướng",
    "IE.468c": "Chướng ngại vật phía trước – đi bên phải",
    "IE.469": "Tiêu phản quang chỉ dẫn hướng rẽ",
    "IE.470": "Số điện thoại khẩn cấp",
    "IE.471": "Giữ khoảng cách lái xe an toàn",
    "IE.472a": "Khoảng cách đến trạm thu phí",
    "IE.472b": "Trạm thu phí",
    "IE.472c": "Thu phí không dừng",
    "IE.472d": "Rút thẻ",
    "IE.472e": "Thu phí",
    "IE.473": "Giảm tốc độ",
    "IE.474": "Lối ra đường cao tốc",
}
IE_MEANINGS = {
    "IE.450a": ("Chỉ dẫn khoảng cách đến nút giao với đường dẫn vào đường cao tốc.", "47.1.1", [44, 45]),
    "IE.450b": ("Chỉ dẫn sơ đồ đến đường dẫn vào đường cao tốc.", "47.1.2", [45]),
    "IE.451a": ("Chỉ dẫn khoảng cách đến lối vào đường cao tốc.", "47.2.1", [45]),
    "IE.451b": ("Chỉ dẫn lối vào đường cao tốc.", "47.2.2", [45]),
    "IE.452": ("Chỉ dẫn bắt đầu đường cao tốc; biển thể hiện tên, ký hiệu đường cao tốc và tốc độ tối đa, tối thiểu.", "47.3", [46]),
    "IE.453a": ("Chỉ dẫn khoảng cách đến điểm kết thúc đường cao tốc.", "47.4.1", [46]),
    "IE.453b": ("Chỉ dẫn điểm kết thúc đường cao tốc.", "47.4.1", [46]),
    "IE.454": ("Chỉ dẫn khoảng cách đến lối ra phía trước.", "47.5", [47]),
    "IE.455a": ("Chỉ dẫn khoảng cách đến các lối ra tiếp theo.", "47.6", [48]),
    "IE.455b": ("Chỉ dẫn khoảng cách đến các lối ra tiếp theo.", "47.6", [48]),
    "IE.456a": ("Thông báo khoảng cách đến trạm dừng nghỉ.", "47.7.2", [49]),
    "IE.456b": ("Chỉ dẫn hướng rẽ vào trạm dừng nghỉ.", "47.7.2", [49]),
    "IE.456c": ("Chỉ dẫn lối vào trạm dừng nghỉ.", "47.7.2", [49]),
    "IE.457a": ("Chỉ dẫn khoảng cách đến nơi đỗ xe trong trạm dừng nghỉ.", "47.8", [49]),
    "IE.457b": ("Chỉ dẫn hướng rẽ đến vị trí bãi đỗ xe trong trạm dừng nghỉ.", "47.8", [49]),
    "IE.458": ("Chỉ dẫn khoảng cách đến các trạm dừng nghỉ tiếp theo.", "47.9", [50]),
    "IE.459a": ("Thông báo khoảng cách đến khu tham quan, du lịch.", "47.10.2", [51]),
    "IE.459b": ("Chỉ dẫn hướng rẽ vào khu tham quan, du lịch.", "47.10.2", [51]),
    "IE.460a": ("Chỉ dẫn cột lý trình kilômét trên đường cao tốc.", "47.11", [51, 52]),
    "IE.460b": ("Chỉ dẫn cột lý trình kilômét trên đường cao tốc.", "47.11", [51, 52]),
    "IE.460c": ("Chỉ dẫn cột lý trình kilômét trên đường cao tốc.", "47.11", [51, 52]),
    "IE.461a": ("Chỉ dẫn khoảng cách đến khu dịch vụ công cộng hoặc giải trí.", "47.12", [52]),
    "IE.461b": ("Chỉ dẫn hướng rẽ đến khu dịch vụ công cộng hoặc giải trí.", "47.12", [52]),
    "IE.461c": ("Chỉ dẫn hướng rẽ vào khu dịch vụ công cộng.", "47.12", [53]),
    "IE.461d": ("Chỉ dẫn hướng rẽ vào khu giải trí.", "47.12", [53]),
    "IE.462": ("Chỉ dẫn tần số sóng radio phát thông tin thời tiết và thông tin tiện ích liên quan đến giao thông trên đường cao tốc.", "47.13", [53]),
    "IE.463a": ("Chỉ dẫn khoảng cách đến công trình kiểm soát tải trọng xe.", "47.14", [53]),
    "IE.463b": ("Chỉ dẫn hướng rẽ vào nơi đặt công trình kiểm soát tải trọng xe.", "47.14", [53]),
    "IE.463c": ("Chỉ dẫn lối vào công trình kiểm soát tải trọng xe.", "47.14", [53, 54]),
    "IE.464a": ("Chỉ dẫn địa điểm nơi đến và làn xe đi ra theo sơ đồ lối ra một chiều.", "47.15", [54]),
    "IE.464b": ("Chỉ dẫn địa điểm và hướng ra tại lối ra.", "47.15", [48, 54, 55]),
    "IE.465a": ("Chỉ dẫn tên địa điểm và khoảng cách đến địa điểm; biển có thể gồm hai hoặc ba dòng thông tin.", "47.16", [55]),
    "IE.465b": ("Chỉ dẫn tên địa điểm và khoảng cách đến địa điểm; biển có thể gồm hai hoặc ba dòng thông tin.", "47.16", [55]),
    "IE.466": ("Chỉ dẫn sơ đồ lối ra một chiều tại các nút giao trên đường cao tốc.", "47.17", [56]),
    "IE.467a": ("Chỉ dẫn vị trí nhập làn.", "47.18", [56]),
    "IE.467b": ("Chỉ dẫn phía trước sắp đến vị trí nhập làn.", "47.18", [56]),
    "IE.468a": ("Chỉ dẫn đi qua chướng ngại vật phía trước theo hướng bên trái.", "47.19", [56, 57]),
    "IE.468b": ("Chỉ dẫn có thể đi qua chướng ngại vật phía trước theo cả hai hướng.", "47.19", [56, 57]),
    "IE.468c": ("Chỉ dẫn đi qua chướng ngại vật phía trước theo hướng bên phải.", "47.19", [56, 57]),
    "IE.469": ("Tiêu phản quang chỉ dẫn hướng rẽ để nhắc người điều khiển phương tiện chuẩn bị đổi hướng khi sắp vào đường cong nguy hiểm có bán kính cong nhỏ.", "47.20", [57]),
    "IE.470": ("Chỉ dẫn số điện thoại khẩn cấp để người điều khiển phương tiện liên hệ khi khẩn cấp, xảy ra sự cố hoặc tai nạn trên đường cao tốc.", "47.21", [58]),
    "IE.471": ("Thông báo cho người điều khiển phương tiện biết để giữ khoảng cách lái xe an toàn khi lưu thông trên đường cao tốc.", "47.22", [58]),
    "IE.472a": ("Thông báo khoảng cách phù hợp phía trước đến trạm thu phí.", "47.23", [59]),
    "IE.472b": ("Thông báo phía trước chuẩn bị đến trạm thu phí.", "47.23", [59]),
    "IE.472c": ("Chỉ dẫn thu phí không dừng.", "47.23; Phụ lục P.24(c)", [59, 400]),
    "IE.472d": ("Chỉ dẫn rút thẻ.", "47.23; Phụ lục P.24(d)", [59, 400]),
    "IE.472e": ("Chỉ dẫn thu phí.", "47.23; Phụ lục P.24(e)", [59, 401]),
    "IE.473": ("Cảnh báo người điều khiển phương tiện chú ý lái xe và giảm tốc độ tại khu vực có nguy cơ mất an toàn giao thông.", "47.24", [59]),
    "IE.474": ("Chỉ dẫn lối ra đường cao tốc.", "47.25", [60]),
}
FIGURE_CODE_LINKS = {
    "qcvn41_2024_figure_p46_10": ["IE.452"],
    "qcvn41_2024_figure_p53_23": ["IE.461a", "IE.461b", "IE.461c", "IE.461d"],
    "qcvn41_2024_figure_p105_108": ["W.219"],
    "qcvn41_2024_figure_p117_140": ["R.302a", "R.302b", "R.302c"],
    "qcvn41_2024_figure_p127_156": ["R.415a", "R.415b"],
    "qcvn41_2024_figure_p128_159": ["R.E.9a", "R.E.9b", "R.E.9c", "R.E.9d"],
    "qcvn41_2024_figure_p129_160": ["R.E.10a", "R.E.10b", "R.E.10c", "R.E.10d"],
    "qcvn41_2024_figure_p129_161": ["R.E.11a", "R.E.11b"],
    "qcvn41_2024_figure_p142_195": ["I.433b", "I.433c", "I.433d"],
    "qcvn41_2024_figure_p155_235": ["S.510a", "S.510b"],
    "qcvn41_2024_figure_p156_236": ["S.G.7", "S.G.8"],
    "qcvn41_2024_figure_p156_238": ["S.G.11a", "S.G.11c"],
    "qcvn41_2024_figure_p157_239": ["S.G.12a", "S.G.12b"],
}
TECHNICAL_FIGURES = {
    "qcvn41_2024_figure_p13_1": "sign_shape_and_size_diagram",
    "qcvn41_2024_figure_p18_2": "sign_mounting_structure_diagram",
    "qcvn41_2024_figure_p19_3": "combined_sign_installation_example",
    "qcvn41_2024_figure_p40_4": "road_number_and_route_symbol_key",
    "qcvn41_2024_figure_p43_5": "service_symbol_key_used_on_signs",
    "qcvn41_2024_figure_p44_6": "expressway_sign_mounting_structure_diagram",
    "qcvn41_2024_figure_p207_331": "letter_geometry_specification",
    "qcvn41_2024_figure_p208_332": "letter_geometry_specification",
    "qcvn41_2024_figure_p209_333": "letter_geometry_specification",
    "qcvn41_2024_figure_p212_334": "letter_geometry_specification",
    "qcvn41_2024_figure_p213_335": "letter_geometry_specification",
}


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    anomalies = json.loads(ANOMALY_PATH.read_text(encoding="utf-8"))
    figure_index = json.loads(UNCODED_FIGURES_PATH.read_text(encoding="utf-8"))
    catalog["items"] = [item for item in catalog["items"] if item.get("official_code") != "I.453c"]
    catalog["items"] = [item for item in catalog["items"] if item.get("official_code") not in IE_FAMILY_CODES]

    bridge = next(item for item in catalog["items"] if item.get("official_code") == "I.439")
    bridge["official_name"] = "Tên cầu"
    bridge["legal_meaning"] = "Biển cung cấp thông tin về cầu; trên biển ghi tên cầu, lý trình và tên đường."
    bridge["legal_basis"] = [{
        "document": "QCVN 41:2024/BGTVT",
        "edition": "2024",
        "section_or_figure": "Phụ lục E, mục E.42 và Hình E.42 (PDF page 144)",
        "source_url": SOURCE_URL,
        "accessed_at": "2026-10-05",
        "verification_status": "verified",
    }]
    bridge["source_evidence"] = {
        "official_pdf_sha256": SOURCE_SHA256,
        "occurrence_pages": [144],
        "source_text_excerpts": [
            'a) Để chỉ các thông tin về một cầu, đặt biển số I.439 "Tên cầu".',
            'c) Trên biển "Tên cầu" ghi: Tên cầu, lý trình và tên đường.',
        ],
    }
    bridge["review_status"] = "unreviewed_extraction"
    bridge["eligible_for_automatic_decision"] = False
    bridge["issues"] = list(dict.fromkeys([
        issue for issue in bridge.get("issues", []) if issue != "no_extracted_legal_definition"
    ] + ["team_legal_review_required"]))
    bridge["review_notes"] = (
        "Installation conditions (bridge length/location) are placement rules, not included in this sign-meaning field."
    )

    ie453c = next(item for item in catalog["items"] if item.get("official_code") == "IE.453c")
    ie453c["legal_meaning"] = "Biển chỉ dẫn hết đường cao tốc."
    ie453c["legal_basis"] = [{
        "document": "QCVN 41:2024/BGTVT",
        "edition": "2024",
        "section_or_figure": "Mục 47.4.2, Hình 10b (PDF page 47)",
        "source_url": SOURCE_URL,
        "accessed_at": "2026-10-05",
        "verification_status": "verified",
    }]
    ie453c["source_evidence"] = {
        "official_pdf_sha256": SOURCE_SHA256,
        "occurrence_pages": [47, 353],
        "source_text_excerpts": [
            '47.4.2. Biển số IE.453c "Hết đường cao tốc". Để chỉ dẫn hết đường cao tốc, '
            'đặt biển số IE.453c “Hết đường cao tốc” (Hình 10b).'
        ],
    }
    ie453c["issues"] = list(dict.fromkeys([
        *ie453c.get("issues", []),
        "appendix_figure_subcaption_mislabels_IE_453c_as_I_453c",
        "team_legal_review_required",
    ]))
    ie453c["review_status"] = "unreviewed_extraction"
    ie453c["eligible_for_automatic_decision"] = False
    ie453c["review_notes"] = (
        "Meaning comes from §47.4.2. Appendix P.4 contains design specifications, not the sign's legal meaning. "
        "Figure 10b subcaption says I.453c, while §47.4.2 and the figure caption say IE.453c; team review required."
    )

    for code, (meaning, section, pages) in IE_MEANINGS.items():
        item = next(row for row in catalog["items"] if row.get("official_code") == code)
        item["official_name"] = IE_NAMES[code]
        item["legal_meaning"] = meaning
        item["legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": f"Mục {section} (PDF page{'s' if len(pages) > 1 else ''} {', '.join(map(str, pages))})",
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
        }]
        item["source_evidence"] = {
            "official_pdf_sha256": SOURCE_SHA256,
            "occurrence_pages": pages,
            "source_text_excerpts": [],
            "source_section_summary": (
                "The legal meaning above is a concise summary, not a verbatim quotation. "
                f"Consult QCVN 41:2024/BGTVT §{section} at the cited PDF page(s)."
            ),
        }
        item["review_status"] = "unreviewed_extraction"
        item["eligible_for_automatic_decision"] = False
        item["issues"] = list(dict.fromkeys([
            issue for issue in item.get("issues", [])
            if issue not in {"no_extracted_legal_definition", "missing_official_name_extraction"}
        ] + ["section_47_meaning_reconciled_from_official_text", "team_legal_review_required"]))
        item["review_notes"] = (
            "Meaning is a concise extraction from the cited operative section, not the technical-specification appendix. "
            "Some variants share only a family-level description in §47; verify variant-specific visual meaning and cross-references during team review."
        )

    for code, meaning in R413_MEANINGS.items():
        item = next(row for row in catalog["items"] if row.get("official_code") == code)
        item["legal_meaning"] = meaning
        item["legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": "Phụ lục D, mục D.15 và Hình D.17 (PDF pages 125–126)",
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
        }]
        item["source_evidence"] = {
            "official_pdf_sha256": SOURCE_SHA256,
            "occurrence_pages": [125, 126],
            "source_text_excerpts": [R413_EXCERPTS[code]],
        }
        item["issues"] = list(dict.fromkeys([
            *item.get("issues", []),
            "official_prose_and_figure_subcaptions_conflict_on_R412_vs_R413_code",
            "team_legal_review_required",
        ]))
        item["review_status"] = "unreviewed_extraction"
        item["eligible_for_automatic_decision"] = False
        item["review_notes"] = (
            "Appendix D.15 prose defines this R.413 variant, but figure subcaptions label variants i–o as R.412; "
            "Hình D.17 caption and variant p use R.413. Preserve the prose code with explicit conflict pending team review."
        )

    for code in ("S.H.3a", "S.H.3b", "S.H.3c"):
        item = next(row for row in catalog["items"] if row.get("official_code") == code)
        item["official_name"] = None
        item["legal_meaning"] = None
        item["legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": "Phụ lục F, mục F.3 và Hình F.3b (PDF page 152)",
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "unverified",
        }]
        item["source_evidence"] = {
            "official_pdf_sha256": SOURCE_SHA256,
            "occurrence_pages": [152],
            "source_text_excerpts": [
                "Trên các tuyến đường đối ngoại và các tuyến đường có nhiều người đi lại có thể sử dụng biển phụ số S.H,3a; S.H,3b; S.H,3c (Biển số H,3 (a,b,c) theo GMS)."
            ],
            "source_section_summary": (
                "QCVN identifies these as optional GMS supplementary signs but does not state each variant's legal meaning in text. "
                "Do not infer an effect from the image or code; obtain team review or a currently applicable authoritative GMS source."
            ),
        }
        item["review_status"] = "unreviewed_extraction"
        item["eligible_for_automatic_decision"] = False
        item["issues"] = list(dict.fromkeys([
            issue for issue in item.get("issues", []) if issue != "no_extracted_legal_definition"
        ] + ["variant_specific_legal_meaning_not_textually_defined_in_QCVN", "team_legal_review_required"]))
        item["review_notes"] = (
            "The source names the GMS family and permitted route contexts only; the individual variant effects remain unverified."
        )

    catalog["items_count"] = len(catalog["items"])
    anomaly_items = anomalies["items"]
    known = {row.get("candidate_code") for row in anomaly_items}
    additions = [
        {
            "candidate_code": "I.453c",
            "candidate_name": "Hết đường cao tốc",
            "source_pdf_pages": [47],
            "extracted_context": "Hình 10b subcaption labels the sign I.453c; §47.4.2 and the figure caption label it IE.453c.",
            "official_cross_check": "The official PDF has an internal code conflict: the operative §47.4.2 text and Hình 10b caption use IE.453c, while the subfigure label uses I.453c. The catalog keeps IE.453c and preserves I.453c as a conflicting printed label, not a separate canonical sign.",
            "disposition": "excluded_from_canonical_catalog_as_conflicting_figure_sublabel;retain_as_source_conflict_pending_team_review",
            "status": "confirmed_official_text_code_conflict",
        },
        {
            "candidate_code": "R.413i–R.413o",
            "candidate_name": "Kết thúc làn đường dành riêng cho từng loại xe hoặc nhóm xe",
            "source_pdf_pages": [125, 126],
            "extracted_context": "Appendix D.15 prose defines R.413i–R.413o; figure subcaptions a–g label the same variants R.412i–R.412o. Figure D.17 caption is R.413; subcaption h is R.413p.",
            "official_cross_check": "This is an internal code conflict in the published QCVN. Catalog entries retain the explicit D.15 prose codes and meanings as unreviewed candidates; no rule may use them until team review resolves the mismatch.",
            "disposition": "retain_prose_defined_R413_variants_as_unreviewed_candidates;block_automatic_decision_pending_team_review",
            "status": "confirmed_official_text_code_conflict",
        },
    ]
    anomaly_items.extend(row for row in additions if row["candidate_code"] not in known)
    for figure in figure_index["items"]:
        codes = FIGURE_CODE_LINKS.get(figure["internal_review_id"])
        if codes:
            figure["review_status"] = "linked_to_catalog_signs"
            figure["classification"] = "official_figure_of_existing_coded_signs"
            figure["related_official_codes"] = codes
            figure["review_notes"] = (
                "The adjacent operative QCVN text explicitly defines these codes and the figure illustrates them. "
                "No new official code is created from the figure caption."
            )
        elif figure["internal_review_id"] in TECHNICAL_FIGURES:
            figure["review_status"] = "technical_diagram"
            figure["classification"] = TECHNICAL_FIGURES[figure["internal_review_id"]]
            figure["related_official_codes"] = []
            figure["review_notes"] = (
                "This figure provides design, installation, symbol, or typography guidance; it is not an independent sign. "
                "Do not add a canonical sign code from this caption."
            )
        elif 158 <= figure["pdf_page"] <= 199 and figure["official_figure_caption"].startswith("Hình G."):
            figure["review_status"] = "out_of_scope"
            figure["classification"] = "road_marking_not_traffic_sign"
            figure["related_official_codes"] = []
            figure["review_notes"] = (
                "Appendix G figures depict road markings. The requested engine scope is traffic signs and supplementary signs, "
                "so these are intentionally not rule inputs."
            )
        elif figure["official_figure_caption"].startswith("Hình A.1"):
            figure["review_status"] = "out_of_scope"
            figure["classification"] = "traffic_signal_not_traffic_sign"
            figure["related_official_codes"] = []
            figure["review_notes"] = "Appendix A depicts traffic lights/signals, not traffic signs or supplementary signs."
        elif figure["official_figure_caption"].startswith("Hình 34a") or figure["official_figure_caption"].startswith("Hình 34b"):
            figure["review_status"] = "out_of_scope"
            figure["classification"] = "reflective_delineator_not_traffic_sign"
            figure["related_official_codes"] = []
            figure["review_notes"] = "QCVN classifies these as reflective delineators, not coded traffic signs."
        elif figure["official_figure_caption"].startswith("Hình B.27a"):
            figure["review_status"] = "technical_diagram"
            figure["classification"] = "speed_limit_design_chart"
            figure["related_official_codes"] = []
            figure["review_notes"] = "A chart for determining a maximum speed limit, not a sign."
    figure_index["classified_existing_sign_figure_count"] = sum(
        item.get("review_status") == "linked_to_catalog_signs" for item in figure_index["items"]
    )
    figure_index["out_of_scope_figure_count"] = sum(
        item.get("review_status") == "out_of_scope" for item in figure_index["items"]
    )
    figure_index["technical_diagram_count"] = sum(
        item.get("review_status") == "technical_diagram" for item in figure_index["items"]
    )
    figure_index["unresolved_figure_count"] = sum(
        item.get("review_status") == "unclassified_illustration" for item in figure_index["items"]
    )
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ANOMALY_PATH.write_text(json.dumps(anomalies, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    UNCODED_FIGURES_PATH.write_text(json.dumps(figure_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
