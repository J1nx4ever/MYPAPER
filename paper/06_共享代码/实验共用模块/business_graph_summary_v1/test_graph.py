"""Semantic ground-truth tests for factual graph construction."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from .graph import (build_graph, NODE_NAMES, RELATION_NAMES, GROUP_NAMES,
                    EDGE_FEATURE_NAMES, QUANTITY_NAMES, QUANTITY_SCALES,
                    ROAD_ENDPOINTS, PHYSICAL_NODE_TO_TOKEN)
from .test_support import graph_state, business_state, tensor, put_truck


def relations(graph, relation, sample=0):
    return [r for r in graph.records(sample) if r['relation'] == relation]


def pairs(graph, relation, sample=0):
    return {(r['source'], r['destination']) for r in relations(graph, relation, sample)}


def quantity(graph, node, name, sample=0):
    return float(graph.raw_quantities[sample, node, QUANTITY_NAMES.index(name)])


def test_road_ground_truth_direction_attributes_no_duplicate_reverse_and_node_layout():
    s = graph_state()
    roads = s[1439:1519].reshape(16, 5)
    roads[0, 2:5] = [.7, .6, 1]
    g = build_graph(tensor(s))
    assert len(NODE_NAMES) == 19 and NODE_NAMES[18] == 'Global'
    assert len(GROUP_NAMES) == 4
    edges = relations(g, 'road')
    assert len(edges) == 16
    assert [(r['source'], r['destination']) for r in edges] == [
        (PHYSICAL_NODE_TO_TOKEN[a], PHYSICAL_NODE_TO_TOKEN[b]) for a, b in ROAD_ENDPOINTS]
    assert edges[0]['attributes']['road_length_10000m'] == pytest.approx(.7)
    assert edges[0]['attributes']['road_closed'] == 1
    assert edges[6]['attributes']['road_closed'] == 0  # separate directed physical road
    assert g.candidate_dump_ids.tolist() == [[13, 13, 13, 14]]
    assert not any('candidate' in name for name in RELATION_NAMES)


def test_factual_commitment_targets_and_reverse_messages_are_exact():
    g = build_graph(tensor(business_state()))
    assert pairs(g, 'reserved_load') == {(1, 9), (2, 9)}
    assert pairs(g, 'reserved_load_reverse') == {(9, 1), (9, 2)}
    assert pairs(g, 'loaded_dump') == {(3, 13), (4, 14)}
    assert pairs(g, 'loaded_dump_reverse') == {(13, 3), (14, 4)}
    assert pairs(g, 'compatible_material') == {(9, 13), (10, 13), (11, 13), (12, 14)}
    assert pairs(g, 'compatible_material_reverse') == {(13, 9), (13, 10), (13, 11), (14, 12)}
    assert quantity(g, 9, 'reserved_truck_count') == pytest.approx(2)
    assert quantity(g, 9, 'reserved_tons') == pytest.approx(210)
    assert quantity(g, 13, 'inbound_loaded_tons') == pytest.approx(140)
    assert quantity(g, 13, 'inbound_known_metal_tons') == pytest.approx(42)
    assert quantity(g, 14, 'inbound_loaded_tons') == pytest.approx(150)


def test_candidates_do_not_modify_factual_graph_or_commitment_totals():
    s = business_state()
    changed = s.copy()
    changed[19:23] = [0, 1, 0, 1]
    changed[23:27] = [0, 50 / 193, 0, 25 / 193]
    first, second = build_graph(tensor(s)), build_graph(tensor(changed))
    for name in ('edge_index', 'edge_type', 'edge_group', 'edge_attr', 'quantities', 'candidate_dump_ids'):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_ego_and_nonpositive_reservation_cannot_form_load_commitment():
    s = graph_state()
    t = s[27:1107].reshape(9, 120)
    t[0, 20], t[0, 26] = 80 / 193, 1
    t[1, 26] = 1  # stale target but no actual reserved tons
    assert not relations(build_graph(tensor(s)), 'reserved_load')


def test_missing_references_never_alias_zero_or_get_guessed():
    s = put_truck(graph_state(), tid=1, loaded=False, ore=False, q=100, target=3)
    t = s[27:1107].reshape(9, 120)
    t[1, 26:30] = 0
    t[1, 32:41] = 0
    t[1, 45:61] = 0
    s = put_truck(s, tid=2, loaded=True, ore=False, q=100)
    s[27 + 240 + 30:27 + 240 + 32] = 0
    g = build_graph(tensor(s))
    assert not relations(g, 'reserved_load')
    assert not relations(g, 'loaded_dump')
    assert all(r['source'] != 1 for r in relations(g, 'truck_at_node'))
    assert not relations(g, 'truck_segment_from')
    assert all(r['attributes']['road_occupancy_9'] == 0 for r in relations(g, 'road'))


def test_in_transit_location_uses_two_endpoint_roles_without_false_arrival():
    s = business_state()
    t = s[27:1107].reshape(9, 120)
    t[2, 32 + 5] = 1  # segment takes precedence over potentially stale node identity
    g = build_graph(tensor(s))
    assert (2, 16) in pairs(g, 'truck_segment_from')  # road8: node3 -> Load0(node5)
    assert (2, 9) in pairs(g, 'truck_segment_to')
    assert all(r['source'] != 2 for r in relations(g, 'truck_at_node'))
    edge = next(r for r in relations(g, 'truck_segment_to') if r['source'] == 2)
    assert edge['attributes']['segment_remaining_10000m'] == pytest.approx(.04)
    assert edge['attributes']['segment_remaining_valid'] == 1
    assert relations(g, 'road')[8]['attributes']['road_occupancy_9'] == pytest.approx(1 / 9)
    assert quantity(g, 9, 'incoming_segment_truck_count') == pytest.approx(1)


def test_count_mass_not_duplicated_by_membership_service_multiset_retains_instances():
    s = business_state()
    load = s[1107:1295].reshape(4, 47)
    load[0, 20:23] = np.asarray([2, 3, 2]) / 9
    load[0, 29:31] = np.asarray([2, 3]) / 9
    load[0, 38:41] = np.asarray([2, 2, 3]) / 9
    g = build_graph(tensor(s))
    assert quantity(g, 9, 'reserved_truck_count') == pytest.approx(2)
    assert quantity(g, 9, 'reserved_tons') == pytest.approx(210)
    assert quantity(g, 9, 'queue_unique_count') == pytest.approx(2)
    assert quantity(g, 9, 'queue_instance_count') == pytest.approx(3)
    assert quantity(g, 9, 'service_unique_count') == pytest.approx(2)
    assert quantity(g, 9, 'service_instance_count') == pytest.approx(3)
    assert len(relations(g, 'service')) == 3


def test_ordered_lists_keep_positions_but_service_permutation_is_exactly_invariant():
    s = business_state()
    load = s[1107:1295].reshape(4, 47)
    load[0, 20:22] = np.asarray([2, 3]) / 9
    load[0, 38:41] = np.asarray([3, 2, 2]) / 9
    g = build_graph(tensor(s))
    rows = relations(g, 'queue')
    assert [r['destination'] for r in rows] == [1, 2]
    assert [r['attributes']['position_9'] for r in rows] == pytest.approx([1 / 9, 2 / 9])
    changed = s.copy()
    changed[1107 + 38:1107 + 41] = np.asarray([2, 3, 2]) / 9
    permuted = build_graph(tensor(changed))
    assert torch.equal(g.edge_index, permuted.edge_index)
    assert torch.equal(g.edge_attr, permuted.edge_attr)
    assert torch.equal(g.quantities, permuted.quantities)
    changed[1107 + 20:1107 + 22] = np.asarray([3, 2]) / 9
    reordered = build_graph(tensor(changed))
    assert not torch.equal(g.edge_index, reordered.edge_index)
    assert torch.equal(g.quantities, reordered.quantities)


def test_unknown_loaded_grade_separate_from_known_zero_grade():
    s = business_state()
    t = s[27:1107].reshape(9, 120)
    t[4, 21:23] = [0, .8]  # unknown waste grade payload must be gated
    g = build_graph(tensor(s))
    edge = next(r for r in relations(g, 'loaded_dump') if r['source'] == 4)
    assert edge['attributes']['payload_grade'] == 0
    assert edge['attributes']['payload_grade_known'] == 0
    assert quantity(g, 14, 'inbound_loaded_tons') == pytest.approx(150)
    assert quantity(g, 14, 'inbound_known_grade_tons') == 0
    t[4, 21:23] = [1, 0]
    known = build_graph(tensor(s))
    assert quantity(known, 14, 'inbound_known_grade_tons') == pytest.approx(150)
    assert quantity(known, 14, 'inbound_known_metal_tons') == 0


def test_batch_offsets_isolation_terminal_ego_and_validation_failures():
    a, b = business_state(), graph_state(terminal=True)
    g = build_graph(tensor([a, b]))
    assert g.batch_size == 2 and g.ego_ids.tolist() == [0, -1]
    assert g.records(0) == build_graph(tensor(a)).records()
    assert g.records(1) == build_graph(tensor(b)).records()
    assert torch.equal(g.quantities[1], build_graph(tensor(b)).quantities[0])
    malformed = g.edge_index.clone()
    malformed[1, 0] += 19
    with pytest.raises(ValueError, match='cross-sample'):
        replace(g, edge_index=malformed).validate()
    bad = a.copy()
    bad[1439] = 2 / 9
    with pytest.raises(ValueError, match='frozen production map'):
        build_graph(tensor(bad))
    assert len(QUANTITY_SCALES) == len(QUANTITY_NAMES)
    assert g.edge_attr.shape[1] == len(EDGE_FEATURE_NAMES)


def test_capacity_wait_and_admission_have_separate_types_and_truthful_order():
    s = business_state()
    dump = s[1295:1391].reshape(2, 48)
    dump[1, 10:12] = [1, 5 / 9]
    g = build_graph(tensor(s))
    assert pairs(g, 'capacity_wait') == {(14, 4)}
    assert pairs(g, 'admission') == {(14, 4)}
    assert quantity(g, 14, 'capacity_wait_unique_count') == pytest.approx(1)
    assert quantity(g, 14, 'admission_count') == pytest.approx(1)
    assert quantity(g, 14, 'inbound_loaded_tons') == pytest.approx(150)


def test_two_hop_business_channels_connect_loaded_pressure_to_candidates():
    g = build_graph(tensor(business_state()))
    # Loaded Truck3 -> ore Dump0 -> ore candidate Load1: a factual two-hop path.
    assert (3, 13) in pairs(g, 'loaded_dump')
    assert (13, 10) in pairs(g, 'compatible_material_reverse')
    # Empty committed Truck2 is directly visible to its actual Load0.
    assert (2, 9) in pairs(g, 'reserved_load')
    # No candidate writes a hypothetical reservation from Ego0.
    assert not any(r['source'] == 0 for r in relations(g, 'reserved_load'))


def test_loaded_payload_wins_over_stale_reservation_without_double_counting():
    s = business_state()
    t = s[27:1107].reshape(9, 120)
    t[3, 20], t[3, 26] = 140 / 193, 1
    g = build_graph(tensor(s))
    assert pairs(g, 'reserved_load') == {(1, 9), (2, 9)}
    assert quantity(g, 9, 'reserved_tons') == pytest.approx(210)
    assert quantity(g, 13, 'inbound_loaded_tons') == pytest.approx(140)


def test_continuous_edge_and_quantity_paths_retain_input_gradient():
    states = tensor(business_state(), requires_grad=True)
    g = build_graph(states)
    assert g.edge_attr.requires_grad and g.quantities.requires_grad
    loss = g.edge_attr.sum() + g.quantities.sum()
    loss.backward()
    for index in (1439 + 2, 1439 + 3, 27 + 120 + 20, 27 + 240 + 61,
                  27 + 360 + 19, 27 + 360 + 22):
        assert states.grad[0, index].isfinite() and states.grad[0, index].abs() > 0
    # Discrete target identity is never treated as a differentiable location.
    assert states.grad[0, 27 + 120 + 26:27 + 120 + 30].abs().sum() == 0


@pytest.mark.parametrize('kind', ['load', 'dump'])
def test_observed_target_must_agree_with_observed_material(kind):
    state = business_state()
    trucks = state[27:1107].reshape(9, 120)
    if kind == 'load':
        trucks[1, 26:30] = [0, 0, 0, 1]  # waste Load3, while reserved material is ore
    else:
        trucks[3, 30:32] = [0, 1]  # waste Dump1, while loaded material is ore
    with pytest.raises(ValueError, match='material differs'):
        build_graph(tensor(state))
