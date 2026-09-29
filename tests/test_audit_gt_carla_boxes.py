import math
from types import SimpleNamespace

import numpy as np

from examples import audit_gt_carla_boxes as audit


def box(carla_id, center, center_z=0.0, heading=(1.0, 0.0), length=4.0, width=2.0,
        height=2.0):
    return audit.WorldBox(carla_id, center, center_z, heading, length, width, height)


def test_exact_oriented_box_overlap_is_contact_at_zero_margin():
    assert audit.boxes_contact(box(1, (0.0, 0.0)), box(2, (3.5, 0.0)))


def test_separated_oriented_boxes_are_not_contact_at_zero_margin():
    assert not audit.boxes_contact(box(1, (0.0, 0.0)), box(2, (4.01, 0.0)))


def test_polygon_clearance_is_symmetric():
    a = audit.oriented_footprint((0.0, 0.0), (1.0, 0.0), 4.0, 2.0)
    b = audit.oriented_footprint((5.0, 0.5), (2 ** -0.5, 2 ** -0.5), 3.0, 1.0)
    assert math.isclose(audit.polygon_clearance(a, b), audit.polygon_clearance(b, a))


def test_bev_overlap_without_vertical_overlap_is_not_3d_contact():
    assert not audit.boxes_contact(box(1, (0.0, 0.0), center_z=0.0),
                                  box(2, (0.0, 0.0), center_z=3.0))


def test_rotated_boxes_use_oriented_geometry():
    diagonal = (2 ** -0.5, 2 ** -0.5)
    assert audit.boxes_contact(box(1, (0.0, 0.0)), box(2, (3.0, 0.0), heading=diagonal))


def test_raw_box_uses_its_own_dimensions_not_a_class_default():
    raw = SimpleNamespace(
        x=1.0, y=2.0, z=3.0, yaw=0.0, length=9.0, width=0.5, height=7.0,
    )
    result = audit.raw_box_to_world(raw, np.eye(4), 99)
    assert result.length == 9.0
    assert result.width == 0.5
    assert result.height == 7.0
    assert result.center == (1.0, -2.0)
    assert result.center_z == 3.0


def test_self_id_is_translated_to_the_observer_carla_id():
    scenario = SimpleNamespace(meta=SimpleNamespace(agent_id_of=lambda agent: 1234))
    assert audit.resolve_carla_id(SimpleNamespace(obj_id=audit.SELF_ID), scenario, "ego_vehicle") == 1234
    assert audit.resolve_carla_id(SimpleNamespace(obj_id=audit.SELF_ID), SimpleNamespace(
        meta=SimpleNamespace(agent_id_of=lambda agent: None)), "infrastructure") is None


def test_interval_is_open_on_start_and_closed_on_end():
    assert not audit.in_interval(5.0, 5.0, 6.0)
    assert audit.in_interval(5.1, 5.0, 6.0)
    assert audit.in_interval(6.0, 5.0, 6.0)


def test_target_pair_hit_requires_the_exact_carla_pair():
    contacts = {frozenset((10, 20)), frozenset((10, 30))}
    assert audit.target_pair_hit(contacts, [20, 10])
    assert not audit.target_pair_hit(contacts, [10, 40])


def test_scenario_selection_key_includes_dataset_split():
    val = SimpleNamespace(split="val", scenario_type="type", scenario="same")
    train = SimpleNamespace(split="train", scenario_type="type", scenario="same")
    scenarios = {audit.scenario_key(s): s for s in (train, val)}
    row = {"dataset_split": "val", "scenario_type": "type", "scenario": "same"}
    assert scenarios[audit.scenario_row_key(row)] is val


def test_duplicate_carla_ids_are_merged_not_counted_as_two_actors(monkeypatch):
    obj = SimpleNamespace(
        obj_id=77, x=0.0, y=0.0, z=0.0, yaw=0.0,
        length=4.0, width=2.0, height=2.0,
    )
    scenario = SimpleNamespace(
        meta=SimpleNamespace(agent_id_of=lambda agent: None),
        agents={
            "ego_vehicle": SimpleNamespace(frames=[1], calib_paths={1: "a"}, label_paths={1: "a"}),
            "other_vehicle": SimpleNamespace(frames=[1], calib_paths={1: "b"}, label_paths={1: "b"}),
        },
    )
    monkeypatch.setattr(audit, "load_calib", lambda path: {
        "ego_to_world": np.eye(4), "lidar_to_ego": np.eye(4),
    })
    monkeypatch.setattr(audit, "parse_label_file", lambda path: SimpleNamespace(objects=[obj]))
    boxes, diagnostics = audit.merge_frame_boxes(scenario, 1)
    assert set(boxes) == {77}
    assert diagnostics["duplicate_carla_id_observations_merged"] == 1
