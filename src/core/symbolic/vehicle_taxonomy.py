"""Shared road-user and vehicle taxonomy grounded in current Vietnamese law.

Primary classification source: Law 36/2024/QH15, Article 34(1)-(5), official
Government Gazette PDF, accessed 2026-10-05:
https://congbao.chinhphu.vn/tai-ve-van-ban-so-36-2024-qh15-42555-51422?format=pdf

The law's categories are not interchangeable: xe cơ giới (motor_vehicle),
xe thô sơ (non_motor_vehicle), and xe máy chuyên dùng (special_use_machine)
are kept separate. "Similar vehicle" is represented as a class requiring a
resolved legal subtype; this module does not infer similarity from appearance
or everyday wording. Taxi and priority/emergency status are actor qualifiers,
not standalone statutory vehicle classes.
"""

MOTOR_VEHICLES = frozenset({
    # Law 36/2024/QH15 Article 34(1): automobiles, trailers/semitrailers,
    # motorized four-wheel passenger/goods vehicles, motorcycles, mopeds,
    # and similar motor vehicles.
    "car", "truck", "bus", "motorbike", "motorcycle", "moped", "motorcycle_or_moped",
    "tractor_unit", "motorized_tricycle", "four_wheel_passenger_vehicle",
    "four_wheel_goods_vehicle", "motor_vehicle", "trailer", "semi_trailer", "taxi",
    "similar_motor_vehicle",
})

NON_MOTOR_VEHICLES = frozenset({
    # Article 34(2): bicycles and assisted/electric bicycles, cycle rickshaws,
    # wheelchairs for persons with disabilities, animal-drawn vehicles, and
    # legally similar non-motor vehicles.
    "bicycle", "electric_bicycle", "bicycle_machine", "pedicab", "wheelchair",
    "animal_drawn_vehicle", "cargo_bicycle", "non_motorized_tricycle",
    "non_motor_vehicle", "similar_non_motor_vehicle",
})

SPECIAL_USE_MACHINES = frozenset({
    # Article 34(3): construction machines, agricultural/forestry machines,
    # tractors and tractor-towed trailers, special-function machines, and
    # defense/security special-purpose vehicles.
    "tractor", "special_use_vehicle", "special_use_machine", "construction_machine",
    "agricultural_forestry_machine", "tractor_towed_trailer",
    "special_function_machine", "defense_security_special_vehicle",
})

# Compatibility/ordinary-language actor types not themselves named as classes
# in Article 34. They remain parseable for clarification, but legal grouping is
# not inferred unless an explicit rule or reviewed mapping supports it.
OTHER_VEHICLE_ACTORS = frozenset({
    "taxi", "hand_cart", "human_pushed_vehicle", "pack_animal",
    "mobility_aid", "child_stroller", "cong_nong_vehicle",
    "priority_vehicle", "emergency_vehicle", "tricycle",
})

VEHICLES = frozenset({
    *MOTOR_VEHICLES,
    *NON_MOTOR_VEHICLES,
    *SPECIAL_USE_MACHINES,
    *OTHER_VEHICLE_ACTORS,
    "vehicle",
})

ROAD_USER_ACTORS = frozenset({
    *VEHICLES,
    "pedestrian",
    # Actor status, not vehicle class: rule predicates must establish when
    # priority/emergency treatment legally applies.
    "priority_vehicle", "emergency_vehicle",
    "unknown",
})

ACTOR_CLASS_CANDIDATES = {
    # An unqualified "xe ba bánh" does not say whether it has a motor.
    # Keep both subtypes so rules can state explicitly which kinds they cover.
    "tricycle": ("motorized_tricycle", "non_motorized_tricycle"),
    # Everyday "xe máy" does not reliably distinguish the statutory xe mô tô
    # from xe gắn máy (moped); retain both until a subtype is stated.
    "motorcycle_or_moped": ("motorbike", "moped"),
}


def vehicle_is_covered(vehicle: str, target_vehicle: str) -> bool:
    """Return only explicit membership/hierarchy matches; unknown is never covered."""
    vehicle = (vehicle or "unknown").strip().lower()
    target_vehicle = (target_vehicle or "unknown").strip().lower()
    if target_vehicle == "unknown" or vehicle == "unknown":
        return False
    if vehicle == target_vehicle:
        return True
    candidates = ACTOR_CLASS_CANDIDATES.get(vehicle)
    if candidates:
        # A composite actor is covered only when every plausible subtype is.
        return all(vehicle_is_covered(candidate, target_vehicle) for candidate in candidates)
    if target_vehicle == "motor_vehicle":
        return vehicle in MOTOR_VEHICLES
    if target_vehicle == "non_motor_vehicle":
        return vehicle in NON_MOTOR_VEHICLES
    if target_vehicle in {"special_use_machine", "special_use_vehicle"}:
        return vehicle in SPECIAL_USE_MACHINES
    if target_vehicle == "vehicle":
        return vehicle in VEHICLES
    if target_vehicle == "car":
        return vehicle in {"car", "taxi"}
    return vehicle == target_vehicle
