"""Retrieve an observed M6 BG128 dispatch case with prospective grade pressure."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from _shared import GROUPS, edge_attention_heads, iter_rows, write_json


DIRECT_GROUPS = ('Transport', 'Commitment')
HORIZON_MIN = 25
PHYSICAL_FIELDS = ('training_seed', 'shift_id', 'run_index', 'seed', 'scenario', 'decision_index', 'sim_time_min',
                   'truck_id', 'truck_type', 'truck_payload_t', 'grade_roll_pp', 'grade_min_pp',
                   'grade_margin_pp', 'grade_expected_pp', 'crusher_queue', 'remaining_min',
                   'ore_t', 'waste_t', 'critic_value')
CASE_FIELDS = (*PHYSICAL_FIELDS, 'shovels', 'action', 'action_mask', 'action_probs', 'baseline_actions',
               'ego_node', 'edges', 'edge_attn_l1', 'edge_attn_l2', 'action_outcomes', 'gamma',
               'ore_unloads')


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)


def action_index(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def classify(row, far_queue=1):
    if (row.get('policy') != 'BG128' or str(row.get('scenario')).casefold() not in ('swapped', 'nominal')
            or str(row.get('dispatch_kind')).upper() != 'LOAD'):
        return 'outside_scope', None, None

    selected = row.get('action')
    greedy = (row.get('baseline_actions') or {}).get('Greedy')
    if not action_index(selected) or not action_index(greedy):
        return 'missing_compared_action', None, None
    if selected == greedy:
        return 'actions_agree', None, None

    margin, expected, minimum = (row.get(key) for key in ('grade_margin_pp', 'grade_expected_pp', 'grade_min_pp'))
    if not all(finite(value) for value in (margin, expected, minimum)):
        return 'missing_grade_state', None, None
    if not (margin > 0 and expected < minimum):
        return 'outside_grade_pattern', None, None

    shovels = {shovel.get('action'): shovel for shovel in row.get('shovels') or []}
    near, far = shovels.get(greedy), shovels.get(selected)
    if near is None or far is None:
        return 'missing_candidate', near, far
    if (not all(finite(shovel.get(key)) for shovel in (near, far)
                for key in ('distance_km', 'grade_pp', 'queue'))
            or not all(isinstance(shovel.get('is_high_grade'), bool) for shovel in (near, far))):
        return 'missing_candidate_physics', near, far
    if (near['distance_km'] >= far['distance_km'] or near['grade_pp'] >= far['grade_pp']
            or near['queue'] != 0 or far['queue'] != far_queue or near.get('is_waste') is True
            or far.get('is_waste') is True or near['is_high_grade'] or not far['is_high_grade']):
        return 'outside_candidate_pattern', near, far

    mask = row.get('action_mask') or []
    if max(selected, greedy) >= len(mask):
        return 'missing_action_mask', near, far
    if not (bool(mask[selected]) and bool(mask[greedy])):
        return 'invalid_candidate_action', near, far
    return 'matches', near, far


def physical_node(shovel):
    if shovel.get('node_id') is not None:
        return str(shovel['node_id'])
    # BG128's four Load actions map to contiguous graph tokens Load0–Load3.
    site = shovel.get('site_id')
    if isinstance(site, str) and site.casefold() in ('load0', 'load1', 'load2', 'load3'):
        return str(9 + int(site[-1]))
    action = shovel.get('action')
    if action_index(action) and action < 4:
        return str(9 + action)
    return None


def attention_record(candidate, site, group, route, graph_captured, *, edges=(), paths=()):
    samples = [edge['heads'] for edge in edges] + [path['head_products'] for path in paths]
    available = [sample for sample in samples if sample is not None]
    if available:
        status = 'observed'
    elif samples or not graph_captured:
        status = 'not_captured'
    else:
        status = 'not_in_graph'
    return dict(candidate=candidate, site_id=site, group=group, mean=float(np.mean(available))
                if available else None, n_edges=len(edges), n_paths=len(paths),
                status=status, route=route, edges=list(edges), paths=list(paths))


def candidate_attention(row, near, far):
    """Find factual incoming edges and two-hop attention paths to the decision truck."""
    edges = row.get('edges')
    ego = row.get('ego_node') if row.get('ego_node') is not None else row.get('truck_id')
    weights = {layer: edge_attention_heads(row, layer) for layer in (1, 2)}
    early, late = weights[1], weights[2]
    if early is not None and late is not None and early.shape != late.shape:
        raise ValueError('Both graph layers must record the same attention heads per factual edge')

    by_source, by_destination, to_ego = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, edge in enumerate(edges or []):
        src, dst = str(edge['src']), str(edge['dst'])
        by_source[src].append(index)
        by_destination[dst].append(index)
        if dst == str(ego):
            to_ego[src].append(index)

    def edge_record(index, layer):
        edge = edges[index]
        heads = weights[layer]
        return dict(edge_index=index, layer=layer, src=edge['src'], dst=edge['dst'],
                    group=str(edge['group']).title(), relation=edge.get('relation'),
                    heads=heads[index].tolist() if heads is not None else None)

    def path_record(first, second):
        product = (early[first] * late[second]).tolist() if early is not None and late is not None else None
        return dict(relay_node=edges[first]['dst'], layer1=edge_record(first, 1),
                    layer2=edge_record(second, 2), head_products=product,
                    mean=float(np.mean(product)) if product is not None else None)

    requested, relayed, displayed = [], [], []
    for label, shovel in (('near', near), ('far', far)):
        node = physical_node(shovel)
        captured = edges is not None and ego is not None and node is not None
        direct, via_shovel, via_relay = defaultdict(list), defaultdict(list), defaultdict(list)
        for second in to_ego[node]:
            direct[str(edges[second]['group']).title()].append(edge_record(second, 2))
            for first in by_destination[node]:
                group = str(edges[first]['group']).title()
                if group not in DIRECT_GROUPS:
                    via_shovel[group].append(path_record(first, second))
        for first in by_source[node]:
            group = str(edges[first]['group']).title()
            for second in to_ego[str(edges[first]['dst'])]:
                via_relay[group].append(path_record(first, second))
        for group in GROUPS:
            original = attention_record(
                label, shovel.get('site_id'), group,
                'direct' if group in DIRECT_GROUPS else 'via_shovel', captured,
                edges=direct[group] if group in DIRECT_GROUPS else (),
                paths=via_shovel[group] if group not in DIRECT_GROUPS else (),
            )
            alternate = attention_record(label, shovel.get('site_id'), group,
                                         'candidate_via_relay', captured, paths=via_relay[group])
            requested.append(original)
            relayed.append(alternate)
            displayed.append(alternate if original['status'] != 'observed'
                             and alternate['status'] == 'observed' else original)
    return displayed, requested, relayed


def action_panel(row):
    mask = row.get('action_mask') or []
    probabilities = row.get('action_probs') or []
    if probabilities and len(probabilities) != len(mask):
        raise ValueError('action_probs and action_mask lengths differ for the selected case')

    shovels = {shovel.get('action'): shovel for shovel in row.get('shovels') or []}
    outcomes = {outcome.get('action'): outcome for outcome in row.get('action_outcomes') or []}
    greedy = row['baseline_actions']['Greedy']
    state_value = row.get('critic_value')
    actions = []
    for action, legal in enumerate(mask):
        if not legal:
            if probabilities and probabilities[action] != 0:
                raise ValueError('Invalid action has nonzero probability in the selected case')
            continue
        outcome = outcomes.get(action, {})
        immediate = outcome.get('immediate_reward')
        next_value = outcome.get('next_critic_value')
        discounted_next = outcome.get('discounted_next_critic_value')
        gamma = outcome.get('gamma', row.get('gamma'))
        if not finite(discounted_next) and finite(next_value) and finite(gamma) and 0 <= gamma <= 1:
            discounted_next = gamma * next_value
        delta = discounted_next - state_value if finite(discounted_next) and finite(state_value) else None
        probability = probabilities[action] if probabilities else None
        if probabilities and (not finite(probability) or not 0 <= probability <= 1):
            raise ValueError('Invalid action probability in the selected case')
        actions.append(dict(action=action, site_id=shovels.get(action, {}).get('site_id'),
                            probability=probability, greedy=action == greedy, executed=action == row['action'],
                            immediate_reward=immediate if finite(immediate) else None,
                            next_critic_value=next_value if finite(next_value) else None,
                            discounted_next_critic_value=discounted_next if finite(discounted_next) else None,
                            discounted_value_delta=delta, outcome_source=outcome.get('source')))
    return actions


def followup_panel(row, shift):
    time = row['sim_time_min']
    followup = []
    event_index = row.get('decision_index')
    for sample_time, sample_index, grade in sorted(shift, key=lambda sample: (sample[0], sample[1] or 0)):
        offset = sample_time - time
        if offset == 0 and event_index is not None and sample_index is not None and sample_index < event_index:
            continue
        if 0 <= offset <= HORIZON_MIN:
            followup.append(dict(offset_min=offset, grade_roll_pp=grade))
    return followup


def arrival_panel(row):
    identity = (row.get('truck_id'), row.get('decision_index'))
    if any(part is None for part in identity):
        return []
    arrivals = []
    for event in row.get('ore_unloads') or []:
        if (event.get('truck_id'), event.get('dispatch_decision_index')) != identity:
            continue
        time = event.get('sim_time_min')
        if finite(time):
            offset = time - row['sim_time_min']
            if offset >= 0:
                arrivals.append(dict(event, offset_min=offset))
    return sorted(arrivals, key=lambda event: event['offset_min'])


def retrieve(rows, far_queue=1):
    traces, diagnostics = defaultdict(list), defaultdict(int)
    last_decision, earliest_candidates = {}, {}
    input_rows = 0
    for input_rows, row in enumerate(rows, start=1):
        reason, near, far = classify(row, far_queue)
        diagnostics[reason] += 1
        if row.get('policy') != 'BG128' or str(row.get('scenario')).casefold() not in ('swapped', 'nominal'):
            continue
        time = row.get('sim_time_min')
        if not finite(time):
            diagnostics['missing_sim_time'] += 1
            continue
        key = row.get('training_seed'), row['shift_id']
        index = row.get('decision_index')
        grade = row.get('grade_roll_pp')
        last_decision[key] = max(last_decision.get(key, time), time)
        if finite(grade):
            traces[key].append((time, index, grade))
        if reason == 'matches':
            stamp = time, index or 0
            if key not in earliest_candidates or stamp < earliest_candidates[key][0]:
                compact = {field: row.get(field) for field in CASE_FIELDS}
                earliest_candidates[key] = stamp, compact, near, far

    base = {'schema': 'bg128-interpretability-case-v3', 'status': 'no_case', 'case': None,
            'criteria': {'far_queue': far_queue, 'strict_far_queue': 1},
            'diagnostics': dict(diagnostics), 'input_rows': input_rows, 'candidates': diagnostics.get('matches', 0)}
    if not earliest_candidates:
        return base

    def rank(key):
        stamp = earliest_candidates[key][0]
        covers_horizon = last_decision[key] >= stamp[0] + HORIZON_MIN
        return not covers_horizon, str(key[0]), str(key[1]), stamp

    key = min(earliest_candidates, key=rank)
    _, row, near, far = earliest_candidates[key]
    attention, requested_attention, relayed_attention = candidate_attention(row, near, far)
    actions = action_panel(row)
    followup = followup_panel(row, traces[key])
    linked_unloads = arrival_panel(row)
    arrivals = [event for event in linked_unloads if event['offset_min'] <= HORIZON_MIN]
    base.update(status='ok', case={
        'near': near,
        'far': far,
        'shovels': row.get('shovels') or [],
        'physical': {field: row.get(field) for field in PHYSICAL_FIELDS},
        'attention': attention,
        'attention_requested': requested_attention,
        'attention_candidate_to_ego': relayed_attention,
        'actions': actions,
        'followup': followup,
        'followup_horizon_min': HORIZON_MIN,
        'shift_log_covers_horizon': not rank(key)[0],
        'arrival_verified': bool(arrivals),
        'arrivals': arrivals,
        'linked_unloads_after_horizon': [event for event in linked_unloads
                                         if event['offset_min'] > HORIZON_MIN],
        'selection': 'Among matching cases, prefer shifts that log through the 25-minute horizon, then sort by '
                     'training seed, shift identity, time, and decision index.',
        'interpretation': {
            'attention': 'For Transport/Commitment read factual Layer 2 shovel-to-truck coefficients. '
                         'For Service/Material read actual paths with a Layer 1 incoming edge into the '
                         'shovel and a Layer 2 shovel-to-truck edge. When a requested route is absent, '
                         'show a factual two-hop candidate-to-truck route through another relay when '
                         'available, classified by its Layer 1 group. Each two-hop path score averages '
                         'products of corresponding recorded head coefficients across the two layers. '
                         'The graph mixes heads and groups, adds residuals and applies normalization. '
                         'Path scores are descriptive and do not measure an output attribution or '
                         'support significance claims from a single case.',
            'critic': 'The frozen BG128 Critic in this archive estimates joint cost from the current state. '
                      'Action-specific estimates require recorded branch outcomes and are null when unobserved.',
            'followup': 'Observed same-shift rolling grade at decision times. An arrival is verified only '
                        'when the recorded unload links this truck to this exact dispatch decision.',
        },
    })
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Decision rows exported by 01_collect_rollouts.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--far-queue', type=int, choices=(0, 1), default=1,
                        help='Required farther-shovel queue size. Use 0 for an explicitly alternate case.')
    args = parser.parse_args()
    write_json(args.output, retrieve(iter_rows(args.input), far_queue=args.far_queue))


if __name__ == '__main__':
    main()
