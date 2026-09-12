"""Convert v9 local-buffer CSVs without changing schedules or target values.

Run with --source OLD_DATASET --destination NEW_DATASET. Existing output is
never overwritten. Reconstruction uses the recorded due dates and lifetime,
not an assumed mapping between machine profiles and repair rates.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from helper.local_buffer import LABEL_METHOD, TARGET_COLUMN, operation_expectation
from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA, local_buffer_node_features,
    reliability_node_feature_names,
)
from helper.time_units import normalize_time_unit

LEGACY_FEATURES = [
    'nominal_start_over_horizon', 'nominal_completion_over_horizon',
    'processing_time_over_weibull_alpha', 'repair_rate_times_weibull_alpha_over_30',
    'weibull_beta_over_5', 'nominal_midpoint_over_horizon',
    'mean_machine_lifetime_over_horizon',
]
SPLITS = {'train': ('training', 'graphs_training.csv'),
          'valid': ('valid', 'graphs_valid.csv'), 'test': ('test', 'graphs_test.csv')}


def convert_row(row):
    if json.loads(row['gnn_feature_names']) != LEGACY_FEATURES:
        raise ValueError('Expected the exact seven-feature v9 schema.')
    due_dates = json.loads(row['training_due_dates'])
    horizon = max(map(float, due_dates.values()))
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Missing or invalid recorded normalization horizon.')
    vectors = json.loads(row['gnn_node_features'])
    membership = json.loads(row['operation_job_indices'])
    targets = json.loads(row[TARGET_COLUMN])
    if len(vectors) != len(membership) or not vectors:
        raise ValueError('Inconsistent operation membership.')
    contributions = [[] for _ in targets]
    features = []
    for x, job in zip(vectors, membership):
        if len(x) != 7 or not all(math.isfinite(v) for v in x):
            raise ValueError('Invalid legacy feature vector.')
        if not math.isclose(x[5], (x[0] + x[1]) / 2, abs_tol=1e-9):
            raise ValueError('Inconsistent legacy midpoint.')
        beta = x[4] * 5.0
        alpha = x[6] * horizon / math.gamma(1 + 1 / beta)
        rate = x[3] * 30.0 / alpha
        t = x[5] * horizon
        features.append(local_buffer_node_features(t, alpha, beta, rate))
        if not isinstance(job, int) or not 0 <= job < len(targets):
            raise ValueError('Invalid job index.')
        contributions[job].append(operation_expectation(t, alpha, beta, rate, order=256))
    error = max(abs(math.fsum(values) - y) for values, y in zip(contributions, targets))
    if not math.isfinite(error) or error > 1e-6:
        raise ValueError(f'Reconstructed labels differ by {error}; migration rejected.')
    converted = dict(row)
    converted['gnn_feature_names'] = json.dumps(reliability_node_feature_names(), separators=(',', ':'))
    converted['gnn_node_features'] = json.dumps(features, separators=(',', ':'))
    return converted, error


def migrate_dataset(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or (destination.exists() and any(destination.iterdir())):
        raise ValueError('Choose a new, empty destination; existing data is preserved.')
    summary = json.loads((source / 'generation_summary.json').read_text())
    if (summary.get('status') != 'completed'
            or summary.get('label', {}).get('label_method') != LABEL_METHOD
            or summary.get('label', {}).get('target_column') != TARGET_COLUMN
            or summary.get('graph', {}).get('feature_names') != LEGACY_FEATURES):
        raise ValueError('Source is not a completed deterministic v9 local-buffer dataset.')
    normalize_time_unit(summary, require_metadata=True)
    updated = copy.deepcopy(summary)
    updated['schema_version'] = max(14, int(updated['schema_version']))
    updated['status'] = 'running'
    updated['output_directory'] = str(destination)
    updated['graph']['graph_schema'] = RELIABILITY_GNN_GRAPH_SCHEMA
    updated['graph']['feature_names'] = reliability_node_feature_names()
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / 'generation_summary.json'
    summary_path.write_text(json.dumps(updated, indent=2))
    audit, names = {}, {}
    for split, (directory, filename) in SPLITS.items():
        path = source / directory / filename
        target = destination / directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        count, max_error, names[split] = 0, 0., set()
        with path.open(newline='') as f, target.open('w', newline='') as out:
            reader = csv.DictReader(f)
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                converted, error = convert_row(row)
                writer.writerow(converted)
                count += 1
                max_error = max(max_error, error)
                names[split].add(row['instance_name'])
        audit[split] = {'graphs': count, 'instances': len(names[split]),
                        'maximum_reconstructed_label_error': max_error,
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'converted_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    if any(names[a] & names[b] for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]):
        raise ValueError('Instance splits overlap; destination remains marked incomplete.')
    updated.update(status='completed', feature_migration={
        'source': str(source), 'source_graph_schema': summary['graph']['graph_schema'],
        'schedules_edges_membership_and_labels_unchanged': True, 'splits': audit,
    })
    summary_path.write_text(json.dumps(updated, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_dataset(args.source, args.destination), indent=2))
