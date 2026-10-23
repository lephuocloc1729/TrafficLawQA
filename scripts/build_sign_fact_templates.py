"""Build fact templates for every class in the reviewed VTSDB100 mapping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deps.detection.yolo26s import _AMBIGUOUS_CLASSES, _EXACT_CLASSES, _FAMILY_CLASSES
from src.core.symbolic.sign_catalog import load_default_sign_catalog
from src.core.symbolic.sign_normalizer import normalize_sign_code


OUTPUT = ROOT / "src/core/symbolic/catalogs/vtsdb100_fact_templates.json"

PROHIBITED_ACTORS = {
    "P.101": ("vehicle",),
    "P.102": ("vehicle",),
    "P.103a": ("motor_vehicle",),
    "P.103b": ("motor_vehicle",),
    "P.103c": ("motor_vehicle",),
    "P.104": ("motorcycle", "moped"),
    "P.106a": ("truck", "tractor", "tractor_unit", "special_use_vehicle"),
    "P.106b": ("truck", "tractor", "tractor_unit", "special_use_vehicle"),
    "P.106c": ("vehicle_carrying_dangerous_goods",),
    "P.107": ("bus", "truck", "tractor", "tractor_unit", "special_use_vehicle"),
    "P.108": ("motor_vehicle", "motorized_tricycle", "tractor", "special_use_vehicle"),
    "P.110a": ("bicycle",),
    "P.112": ("pedestrian",),
    "P.115": ("vehicle",),
    "P.116": ("vehicle",),
    "P.117": ("vehicle",),
    "P.118": ("vehicle",),
    "P.123a": ("vehicle",),
    "P.123b": ("vehicle",),
    "P.124a": ("vehicle",),
    "P.124b": ("car", "motorized_tricycle"),
    "P.124c": ("vehicle",),
    "P.125": ("motor_vehicle",),
    "P.128": ("vehicle",),
    "P.130": ("motor_vehicle",),
    "P.131a": ("motor_vehicle",),
    "P.131b": ("motor_vehicle",),
    "P.131c": ("motor_vehicle",),
    "P.136": ("vehicle",),
    "P.137": ("vehicle",),
}

# These scopes refine the actor set itself. Vehicle types are otherwise
# represented once, in prohibited_actor.value; do not repeat them on the
# paired prohibited_action fact.
PROHIBITED_ACTOR_SCOPES = {
    "P.103a": {"excluded_vehicle_types": ["motorcycle", "moped"]},
    "P.103b": {"excluded_vehicle_types": ["motorcycle", "moped"]},
    "P.103c": {"excluded_vehicle_types": ["motorcycle", "moped"]},
    "P.125": {"excluded_vehicle_types": ["motorcycle", "moped"]},
}

ACTION_SCOPE = {
    "P.101": ("entering",),
    "P.102": ("entering",),
    "P.103a": ("entering",),
    "P.107": ("entering",),
    "P.104": ("entering",),
    "P.106a": ("entering",),
    "P.106b": ("entering",),
    "P.106c": ("entering",),
    "P.108": ("entering",),
    "P.110a": ("entering",),
    "P.112": ("entering",),
    "P.115": ("entering",),
    "P.116": ("entering",),
    "P.117": ("entering",),
    "P.118": ("entering",),
    "P.103b": ("turn_right",),
    "P.103c": ("turn_left",),
    "P.123a": ("turn_left",),
    "P.123b": ("turn_right",),
    "P.124a": ("u_turn",),
    "P.124b": ("u_turn",),
    "P.124c": ("turn_left", "u_turn"),
    "P.125": ("overtake",),
    "P.128": ("honk",),
    "P.130": ("stop", "park"),
    "P.131a": ("park",),
    "P.131b": ("park",),
    "P.131c": ("park",),
    "P.136": ("go_straight",),
    "P.137": ("turn_left", "turn_right"),
    "P.129": ("stop",),
    "R.301a": ("go_straight",),
    "R.301d": ("turn_right",),
    "R.301e": ("turn_left",),
    "R.302a": ("passing_obstacle",),
    "R.303": ("roundabout_circulation",),
    "W.224": ("reduce_speed", "yielding"),
    "W.227": ("reduce_speed",),
    "I.442": ("reduce_speed",),
}

OCR_SLOTS = {
    "P.106b": ("maximum_payload_mass", "t"),
    "P.115": ("maximum_vehicle_mass", "t"),
    "P.116": ("maximum_axle_load", "t"),
    "P.117": ("maximum_vehicle_height", "m"),
    "P.118": ("maximum_vehicle_width", "m"),
    "P.119": ("maximum_vehicle_length", "m"),
    "P.120": ("maximum_combination_length", "m"),
    "P.121": ("minimum_distance", "m"),
    "P.127": ("maximum_speed", "km/h"),
    "R.306": ("minimum_speed", "km/h"),
    "W.239b": ("maximum_vehicle_height", "m"),
}

# Keep scopes only where they narrow the fact to a meaningful subset of
# vehicles. A universal vehicle scope adds no information.
FACT_SCOPES = {
    ("P.106b", "maximum_payload_mass"): {"vehicle_types": ["truck"]},
    ("P.121", "minimum_distance"): {
        "vehicle_types": ["car", "bus", "truck", "tractor_unit", "four_wheel_passenger_vehicle", "four_wheel_goods_vehicle"]
    },
    ("P.127", "maximum_speed"): {"vehicle_types": ["motor_vehicle"]},
    ("R.306", "minimum_speed"): {"vehicle_types": ["motor_vehicle"]},
}

ACTION_FACT_SCOPES = {
    "P.129": {"vehicle_types": ["vehicle"]},
    "R.301a": {"vehicle_types": ["vehicle"]},
    "R.301d": {"vehicle_types": ["vehicle"]},
    "R.301e": {"vehicle_types": ["vehicle"]},
    "R.302a": {"vehicle_types": ["vehicle"]},
    "R.303": {"vehicle_types": ["vehicle"]},
    "W.224": {"vehicle_types": ["vehicle"]},
    "W.227": {"vehicle_types": ["vehicle"]},
    "I.442": {"vehicle_types": ["motor_vehicle"]},
}

ROAD_VEHICLE_SCOPE = {
    "R.403a": ("automobile",),
    "I.434a": ("bus",),
}

# Plain-language meanings for the detector's exact classes. Keep these keyed
# by class rather than sign code because several detector classes share a code
# but show different values, directions, or variants of the sign.
SIGN_MEANINGS = {
    "no_entry": "Cấm mọi phương tiện đi vào từ phía đặt biển. Xe ưu tiên được đi theo quy định; người đi bộ vẫn có thể đi trên vỉa hè hoặc lề đường.",
    "no_cars": "Cấm ô tô đi vào đoạn đường này, trừ xe được ưu tiên theo quy định.",
    "no_cars_turn_right": "Cấm ô tô rẽ phải tại nút giao phía trước. Biển này không cấm xe máy hai bánh và xe gắn máy rẽ phải; xe ưu tiên được áp dụng theo quy định.",
    "no_cars_turn_left": "Cấm ô tô rẽ trái tại nút giao phía trước. Biển này không cấm xe máy hai bánh và xe gắn máy rẽ trái; xe ưu tiên được áp dụng theo quy định.",
    "no_motorcycles": "Cấm xe máy đi vào đoạn đường này, trừ xe được ưu tiên theo quy định. Người dắt xe máy vẫn được đi.",
    "no_trucks": "Cấm xe tải đi vào đoạn đường này; biển cũng áp dụng với máy kéo và xe máy chuyên dùng. Xe ưu tiên được đi theo quy định.",
    "no_trucks_over_2.5t": "Cấm xe tải có khối lượng chuyên chở vượt quá 2,5 tấn. Biển cũng áp dụng với máy kéo và xe máy chuyên dùng theo quy định.",
    "no_pedestrians": "Cấm người đi bộ đi qua đoạn đường này.",
    "height_limit": "Xe cao hơn mức ghi trên biển không được đi qua.",
    "no_turn_left": "Cấm rẽ trái tại nút giao phía trước; biển không cấm quay đầu xe.",
    "no_turn_right": "Cấm rẽ phải tại nút giao phía trước; biển không cấm quay đầu xe.",
    "no_u_turn": "Cấm quay đầu xe tại vị trí đặt biển.",
    "no_turn_left_and_u_turn": "Cấm cả rẽ trái và quay đầu xe tại nút giao phía trước.",
    "no_honking": "Cấm sử dụng còi trong đoạn đường có hiệu lực của biển.",
    "no_stopping_and_parking": "Cấm dừng xe và đỗ xe ở phía đường có đặt biển, trừ xe ưu tiên theo quy định.",
    "no_parking": "Cấm đỗ xe ở phía đường có đặt biển, trừ xe ưu tiên theo quy định.",
    "no_parking_odd_days": "Cấm đỗ xe vào ngày lẻ trong tháng ở phía đường có đặt biển, trừ xe ưu tiên theo quy định.",
    "no_parking_even_days": "Cấm đỗ xe vào ngày chẵn trong tháng ở phía đường có đặt biển, trừ xe ưu tiên theo quy định.",
    "no_straight_ahead": "Cấm đi thẳng qua nút giao phía trước, trừ xe ưu tiên theo quy định. Biển phụ có thể giới hạn lệnh cấm theo giờ hoặc loại xe.",
    "no_turn_left_and_right": "Cấm rẽ trái và rẽ phải tại nút giao phía trước, trừ xe ưu tiên theo quy định. Biển phụ có thể giới hạn lệnh cấm theo giờ hoặc loại xe.",
    "no_cars_u_turn": "Cấm ô tô và xe ba bánh có động cơ quay đầu tại vị trí đặt biển.",
    "no_buses_and_trucks": "Cấm xe khách và xe tải đi vào đoạn đường này; biển cũng áp dụng với máy kéo và xe máy chuyên dùng. Xe ưu tiên được đi theo quy định.",
    "no_trailers": "Cấm xe kéo theo rơ-moóc đi vào, kể cả xe máy, máy kéo và xe khách. Biển không áp dụng với ô tô đầu kéo kéo sơ-mi rơ-moóc và xe ưu tiên theo quy định.",
    "no_vehicles_carrying_dangerous_goods": "Cấm xe chở hàng nguy hiểm đi vào đoạn đường này.",
    "no_bicycles": "Cấm xe đạp đi vào đoạn đường này. Người dắt xe đạp vẫn được đi.",
    "no_slow_moving_vehicles": "Theo phân loại riêng của bộ dữ liệu này, biển cấm xe ba bánh có động cơ và xe ba bánh thô sơ, kể cả xe lam.",
    "no_motorcycles_and_slow_vehicles": "Theo phân loại riêng của bộ dữ liệu này, biển cấm xe máy và các loại xe ba bánh có động cơ hoặc thô sơ.",
    "give_way": "Báo trước sắp giao nhau với đường ưu tiên. Xe trên đường đang đi phải giảm tốc độ và nhường đường cho xe trên đường ưu tiên.",
    "width_limit": "Xe rộng hơn mức ghi trên biển không được đi qua.",
    "weight_limit": "Cấm xe có tổng khối lượng của xe, người và hàng vượt quá mức ghi trên biển đi qua, kể cả xe ưu tiên.",
    "axle_weight_limit": "Cấm xe có tải trọng trên bất kỳ trục nào vượt quá mức ghi trên biển đi qua, kể cả xe ưu tiên.",
    "no_overtaking": "Cấm các xe cơ giới vượt nhau trong đoạn đường có hiệu lực của biển.",
    "min_distance_between_vehicles": "Ô tô phải giữ khoảng cách với xe phía trước ít nhất bằng mức ghi trên biển.",
    "end_no_overtaking": "Báo hết đoạn đường cấm vượt; từ đây, xe cơ giới được vượt khi đáp ứng quy định an toàn.",
    "end_all_prohibitions": "Báo các lệnh cấm trên đoạn đường vừa qua đã hết hiệu lực.",
    "road_closed": "Cấm mọi phương tiện đi lại theo cả hai chiều, trừ xe ưu tiên theo quy định.",
    "pedestrian_crossing": "Báo trước nơi người đi bộ sang đường. Giảm tốc độ, nhường đường và chỉ đi tiếp khi không gây nguy hiểm cho người đi bộ.",
    "children_crossing": "Báo trước khu vực có trẻ em thường sang đường hoặc tụ tập gần đường. Hãy giảm tốc độ và quan sát kỹ.",
    "road_works": "Báo trước đoạn đường đang thi công, có thể có công nhân và máy móc trên mặt đường. Giảm tốc độ và đi cẩn thận.",
    "two_way_traffic": "Báo trước đoạn đường phía trước có xe chạy theo cả hai chiều, thường do sửa đường hoặc có chướng ngại vật.",
    "railway_crossing_with_gates": "Báo trước giao cắt với đường sắt không có rào chắn và không có người điều khiển giao thông. Quan sát kỹ trước khi qua.",
    "go_straight": "Tại nút giao, các phương tiện phải đi thẳng theo hướng mũi tên.",
    "turn_left_only": "Tại nút giao, các phương tiện phải rẽ trái theo hướng mũi tên.",
    "turn_right_only": "Tại nút giao, các phương tiện phải rẽ phải theo hướng mũi tên.",
    "roundabout": "Báo nơi giao nhau có vòng xuyến; các phương tiện phải đi theo chiều vòng quanh đảo an toàn.",
    "main_road": "Bắt đầu đoạn đường ưu tiên. Xe từ đường nhánh phải nhường đường cho xe đang đi trên đường này.",
    "end_main_road": "Báo hết đoạn đường ưu tiên.",
    "hospital": "Chỉ dẫn gần đó có bệnh viện, bệnh xá hoặc cơ sở điều trị.",
    "bus_stop": "Chỉ dẫn điểm dừng hoặc bến xe buýt để đón, trả khách. Biển phụ có thể ghi tuyến và lộ trình.",
    "market": "Báo trước khu vực gần chợ. Xe cơ giới cần quan sát kỹ và giảm tốc độ.",
    "checkpoint": "Báo trạm kiểm tra phía trước. Phương tiện vận tải phải dừng để làm thủ tục kiểm tra theo quy định.",
    "one_way_road": "Chỉ dẫn đường một chiều. Biển này được đặt sau nút giao để xác nhận chiều lưu thông trên đường.",
    "u_turn_area": "Chỉ dẫn vị trí được phép quay đầu xe.",
    "clearance_height": "Báo chiều cao thông xe thực tế bên dưới công trình phía trước. Xe cao hơn mức ghi trên biển không được đi qua.",
    "slow_down": "Nhắc người lái xe giảm tốc độ trước đoạn đường cần đi chậm.",
    "accident_area": "Cảnh báo đoạn đường phía trước thường xảy ra tai nạn. Hãy chú ý và điều khiển xe cẩn thận.",
    "end_dual_carriageway": "Báo trước đoạn đường đôi sắp kết thúc; phía trước, hai chiều xe chạy không còn được ngăn cách bằng dải phân cách cứng.",
    "tunnel": "Chỉ dẫn phía trước có hầm chui.",
    "end_tunnel": "Chỉ dẫn sắp hết hầm chui.",
    "road_reserved_for_cars": "Chỉ dẫn đường dành cho các loại ô tô.",
    "expressway": "Chỉ dẫn bắt đầu đường cao tốc.",
    "reserved_for_disabled": "Chỉ dẫn chỗ đỗ xe dành cho người khuyết tật.",
    "residential_area": "Báo bắt đầu khu đông dân cư; biển có hiệu lực đến biển hết khu đông dân cư. Trong khu vực này, tốc độ tối đa mặc định phụ thuộc vào loại xe và loại đường: ô tô con, ô tô tải, ô tô chở người và xe máy trên 50 cm³ được chạy tối đa 60 km/h trên đường đôi hoặc đường một chiều có từ hai làn xe, và 50 km/h trên đường hai chiều hoặc đường một chiều có một làn xe. Xe gắn máy dưới 50 cm³, xe máy điện và xe máy chuyên dùng tối đa 40 km/h trên các loại đường này. Các giới hạn tốc độ ghi trên biển tại đoạn đường vẫn phải được tuân thủ.",
    "end_residential_area": "Báo hết khu đông dân cư; từ đây không còn áp dụng các quy định riêng về lưu thông trong khu đông dân cư.",
    "parking_area": "Chỉ dẫn nơi được phép đỗ xe hoặc bãi đỗ xe.",
    "speed_limit_40": "Tốc độ tối đa cho phép là 40 km/h.",
    "speed_limit_50": "Tốc độ tối đa cho phép là 50 km/h.",
    "speed_limit_60": "Tốc độ tối đa cho phép là 60 km/h.",
    "speed_limit_70": "Tốc độ tối đa cho phép là 70 km/h.",
    "speed_limit_80": "Tốc độ tối đa cho phép là 80 km/h.",
    "speed_limit_90": "Tốc độ tối đa cho phép là 90 km/h.",
    "speed_limit_100": "Tốc độ tối đa cho phép là 100 km/h.",
    "speed_limit_120": "Tốc độ tối đa cho phép là 120 km/h.",
    "min_speed_limit_30": "Tốc độ tối thiểu phải duy trì là 30 km/h.",
    "min_speed_limit_60": "Tốc độ tối thiểu phải duy trì là 60 km/h.",
    "end_min_speed_limit_30": "Báo hết đoạn đường áp dụng tốc độ tối thiểu.",
    "steep_descent_5_percent": "Cảnh báo dốc xuống nguy hiểm phía trước; độ dốc được ghi là 5%. Hãy giảm tốc độ và kiểm soát xe.",
    "steep_ascent_5_percent": "Cảnh báo dốc lên nguy hiểm phía trước; độ dốc được ghi là 5%.",
    "dangerous_curve_left": "Cảnh báo khúc cua nguy hiểm vòng sang trái phía trước.",
    "dangerous_curve_right": "Cảnh báo khúc cua nguy hiểm vòng sang phải phía trước.",
    "cliff_ahead": "Cảnh báo phía trước có kè, vực sâu hoặc sông suối sát đường; cần chú ý nguy cơ xe rơi khỏi đường.",
    "road_narrows": "Cảnh báo mặt đường phía trước bị thu hẹp ở cả hai bên.",
    "crossroad": "Cảnh báo sắp đến nơi giao nhau cùng mức với các đường ngang cấp, không có đường ưu tiên.",
    "intersection_with_non_priority_road": "Cảnh báo sắp giao nhau với đường nhánh không ưu tiên. Xe trên đường nhánh phải nhường đường.",
    "road_bump": "Cảnh báo phía trước có gồ giảm tốc hoặc đoạn mặt đường lồi lõm. Giảm tốc độ để đi qua an toàn.",
    "pedestrian_crossing_zone": "Chỉ dẫn vị trí dành cho người đi bộ sang ngang đường.",
    "pedestrian_overpass": "Chỉ dẫn lối xuống hầm chui dành cho người đi bộ.",
    "multiple_curves": "Cảnh báo phía trước có từ hai khúc cua liên tiếp; khúc cua đầu tiên vòng sang trái.",
    "intersection_with_non_priority_road_right": "Cảnh báo sắp giao nhau với đường nhánh không ưu tiên từ bên phải.",
    "t_junction": "Cảnh báo sắp đến ngã ba dạng chữ T; đường phía trước không tiếp tục đi thẳng.",
    "t_junction_right": "Cảnh báo sắp đến ngã ba dạng chữ T, có nhánh rẽ sang phải.",
    "t_junction_left": "Cảnh báo sắp đến ngã ba dạng chữ T, có nhánh rẽ sang trái.",
    "intersection_with_non_priority_road_left": "Cảnh báo sắp giao nhau với đường nhánh không ưu tiên từ bên trái.",
    "dead_end": "Chỉ dẫn phía trước là đường cụt, không có lối đi tiếp.",
    "underpass": "Cảnh báo phía trước có cổng hoặc vòm thấp có thể hạn chế xe cao đi qua.",
    "road_narrows_left": "Cảnh báo mặt đường phía trước bị thu hẹp từ bên trái.",
    "road_narrows_right": "Cảnh báo mặt đường phía trước bị thu hẹp từ bên phải.",
    "keep_right": "Chỉ dẫn đi vòng qua chướng ngại vật ở bên phải.",
    "other_prohibition_signs": "Đây là nhóm biển cấm chưa xác định được loại cụ thể; cần nhận diện rõ hình biển để giải thích chính xác lệnh cấm.",
    "overhead_power_lines": "Biển thuộc nhóm cảnh báo hoặc chỉ dẫn về chướng ngại phía trên đường, như đường dây điện; cần đọc trị số tĩnh không nếu biển có ghi chiều cao."
}


def _fact(predicate: str, value, **extra) -> dict:
    return {"predicate": predicate, "value": value, **extra}


def _template(class_label: str, sign_code: str | None, attributes: dict, catalog, identity_status: str) -> dict:
    result = {
        "class_label": class_label,
        "sign_code": sign_code,
        "identity_status": identity_status,
        "facts": [],
    }
    if not sign_code or identity_status != "identified":
        meaning = SIGN_MEANINGS.get(class_label)
        if meaning:
            result["facts"].append(_fact(
                "sign_meaning", meaning, status="derived", confidence="low",
                source_status="unresolved_class_description",
            ))
        result["template_status"] = "unresolved_identity"
        return result

    entry = catalog.lookup(sign_code)
    if entry:
        meaning = SIGN_MEANINGS.get(class_label, entry.legal_meaning or entry.official_name)
        result["facts"].append(_fact(
            "sign_meaning",
            meaning,
            status="derived",
            confidence="medium" if entry.status == "verified" else "low",
            source_status=entry.status,
        ))

    actor_types = PROHIBITED_ACTORS.get(sign_code)
    if actor_types:
        value = list(actor_types)
        conditions = {}
        if sign_code in {
            "P.101", "P.102", "P.103a", "P.103b", "P.103c", "P.104", "P.106a", "P.107", "P.108",
            "P.123a", "P.123b", "P.124a", "P.124b", "P.124c", "P.130", "P.131a", "P.131b", "P.131c",
            "P.136", "P.137",
        }:
            conditions["exceptions"] = ["priority_vehicle"]
        if sign_code == "P.102":
            conditions["direction"] = "opposite_sign"
        if sign_code == "P.106b":
            conditions["truck"] = {
                "predicate": "maximum_payload_mass",
                "operator": ">",
                "limit": "sign_value",
            }
        if sign_code == "P.108":
            conditions["trailer_configuration"] = {
                "required": "towing_trailer",
                "excluded_configurations": ["semi_trailer"],
            }
        actor_scope = PROHIBITED_ACTOR_SCOPES.get(sign_code)
        result["facts"].append(_fact(
            "prohibited_actor", value, status="derived", confidence="medium",
            **({"scope": actor_scope} if actor_scope else {}),
            **({"conditions": conditions} if conditions else {}),
        ))

    actions = ACTION_SCOPE.get(sign_code)
    if actions:
        action_predicate = (
            "mandatory_action" if sign_code in {"P.129", "R.301a", "R.301d", "R.301e", "R.302a", "R.303", "W.224", "W.227"}
            else "recommended_action" if sign_code == "I.442"
            else "prohibited_action"
        )
        # For prohibitions, prohibited_actor is the single source of vehicle
        # applicability. Do not duplicate that list on prohibited_action.
        action_scope = ACTION_FACT_SCOPES.get(sign_code) if action_predicate != "prohibited_action" else None
        result["facts"].append(_fact(
            action_predicate, list(actions), status="derived", confidence="medium",
            **({"scope": action_scope} if action_scope else {}),
        ))

    road_vehicle_types = ROAD_VEHICLE_SCOPE.get(sign_code)
    if road_vehicle_types:
        result["facts"].append(_fact(
            "vehicle_scope", list(road_vehicle_types), status="derived", confidence="medium",
        ))

    if sign_code in {"P.131b", "P.131c"}:
        result["facts"].append(_fact(
            "calendar_parity", "odd" if sign_code == "P.131b" else "even",
            status="derived", confidence="medium",
        ))

    slot = OCR_SLOTS.get(sign_code)
    if slot:
        predicate, unit = slot
        value = attributes.get({
            "maximum_payload_mass": "payload_limit_t",
            "maximum_vehicle_mass": "total_weight_limit_t",
            "maximum_axle_load": "axle_load_limit_t",
            "maximum_vehicle_height": "height_limit_m",
            "maximum_vehicle_width": "width_limit_m",
            "maximum_vehicle_length": "length_limit_m",
            "maximum_combination_length": "combination_length_limit_m",
            "minimum_distance": "minimum_following_distance_m",
            "maximum_speed": "speed_limit_kmh",
            "minimum_speed": "min_speed_kmh",
        }[predicate])
        extra = {"scope": FACT_SCOPES[(sign_code, predicate)]} if (sign_code, predicate) in FACT_SCOPES else {}
        result["facts"].append(_fact(
            predicate, value, unit=unit,
            status="derived" if value is not None else "not_observed",
            confidence="high" if value is not None else "low",
            fill_from="ocr_or_detector_attribute",
            **extra,
        ))
    return result


def main() -> None:
    catalog = load_default_sign_catalog()
    classes = {}
    for label, (code, attributes, _section) in _EXACT_CLASSES.items():
        classes[label] = _template(label, normalize_sign_code(code), attributes, catalog, "identified")
    for label in _AMBIGUOUS_CLASSES:
        classes[label] = _template(label, None, {}, catalog, "unidentified")
    for label, (family, candidates, _section) in _FAMILY_CLASSES.items():
        item = _template(label, None, {}, catalog, "ambiguous")
        item["sign_family"] = family
        item["candidate_codes"] = candidates
        classes[label] = item

    missing_meanings = sorted(set(classes) - set(SIGN_MEANINGS))
    if missing_meanings:
        raise ValueError(f"Missing plain-language sign meanings for classes: {missing_meanings}")

    payload = {
        "schema_version": "1.0",
        "source": "VTSDB100 reviewed class mapping and QCVN 41:2024/BGTVT sign catalog",
        "class_count": len(classes),
        "classes": classes,
        "supplementary_templates": {
            "ocr_time_panel": [
                _fact("active_time", None, status="not_observed", confidence="low", fill_from="ocr")
            ]
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(classes)} sign class templates to {OUTPUT}")


if __name__ == "__main__":
    main()
