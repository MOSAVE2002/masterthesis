"""Input-aware deduplication and cost/buffer/structure coverage selection."""
import json
import numpy as np


def input_signature(candidate):
    row = candidate['row']
    return (tuple(tuple(round(float(v), 7) for v in x) for x in json.loads(row['gnn_node_features'])),
            tuple(json.loads(row['operation_job_indices'])),
            tuple(sorted(map(tuple, json.loads(row['gnn_active_edges'])))))


def unique_input_candidates(candidates):
    result, seen = [], set()
    for candidate in candidates:
        key = input_signature(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def select_buffer_candidates(candidates, count):
    """mix462 at count=12: four cost/buffer quadrants, six buffer, two structure."""
    pool = unique_input_candidates(candidates)
    count = int(count)
    if count <= 0 or len(pool) < count:
        raise ValueError(f"Need {count} distinct input graphs, got {len(pool)}.")
    y = np.array([c['local_job_repair_buffers'] for c in pool], dtype=float)
    cost = np.array([c['nominal_schedule_cost'] for c in pool], dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(cost).all():
        raise ValueError("Candidate costs and buffers must be finite.")
    assignments = np.array([[m for o, m in c['structure'][0]] for c in pool])
    assignment_distance = (assignments[:, None, :] != assignments[None, :, :]).mean(axis=2)
    edges = [set(c['structure'][1]) for c in pool]
    distances = (assignment_distance + np.array([
        [len(a ^ b) / max(1, len(a | b)) for b in edges] for a in edges])) / 2
    mean = y.mean(axis=1)
    ch, bh = cost > np.median(cost), mean > np.median(mean)
    q = np.quantile(y, [1/3, 2/3], axis=0)
    bins = (y > q[0]).astype(int) + (y > q[1]).astype(int)
    onehot = np.eye(3)[bins]
    z = (y - y.min(axis=0)) / np.maximum(np.ptp(y, axis=0), 1e-8)
    chosen, roles = [], {}
    def add(i, role):
        chosen.append(int(i)); roles[int(i)] = role
    def novelty(i):
        return float(distances[i, chosen].min()) if chosen else 0.
    cz = (cost - cost.min()) / max(float(np.ptp(cost)), 1e-8)
    bz = (mean - mean.min()) / max(float(np.ptp(mean)), 1e-8)
    for c, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        if len(chosen) >= min(4, count): break
        available = [i for i in range(len(pool)) if ch[i] == c and bh[i] == b and i not in chosen]
        if available:
            add(max(available, key=lambda i: (-abs(cz[i] - (.75 if c else .25))
                - abs(bz[i] - (.75 if b else .25)), novelty(i), -i)), f'cost_{c}_buffer_{b}')
    buffer_limit = max(len(chosen), count - max(1, round(count/6)))
    while len(chosen) < buffer_limit:
        counts = onehot[chosen].sum(axis=0); target = (len(chosen) + 1) / 3
        add(max((i for i in range(len(pool)) if i not in chosen), key=lambda i: (
            -float(((counts + onehot[i] - target)**2).sum()),
            float(((z[i] - z[chosen])**2).mean(axis=1).min()), -i)), 'buffer_coverage')
    while len(chosen) < count:
        add(max((i for i in range(len(pool)) if i not in chosen),
                key=lambda i: (novelty(i), -cost[i], -i)), 'structure')
    return [{'candidate': pool[i], 'category': roles[i]} for i in sorted(chosen)]
