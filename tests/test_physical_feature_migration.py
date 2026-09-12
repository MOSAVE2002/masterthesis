import importlib
import json
import math

import pytest
import torch
from torch_geometric.data import Data
from helper.local_buffer import TARGET_COLUMN, operation_expectation
from helper.sequence_setup import local_buffer_node_features

migration = importlib.import_module('04_GraphNeuralNetworks.models.migrate_local_buffer_features')
training = importlib.import_module('04_GraphNeuralNetworks.models.model_training_FJSP_GNN')


def legacy_row(horizon=100):
    # Deliberately neither of the two default profiles.
    t, a, b, rate, duration = 40., 150., 2.5, .07, 10.
    return {'training_due_dates': json.dumps({'1': horizon}),
            'gnn_feature_names': json.dumps(migration.LEGACY_FEATURES),
            'gnn_node_features': json.dumps([[(t-duration/2)/horizon,
                (t+duration/2)/horizon, duration/a, rate*a/30, b/5,
                t/horizon, a*math.gamma(1+1/b)/horizon]]),
            'operation_job_indices': '[0]',
            TARGET_COLUMN: json.dumps([operation_expectation(t, a, b, rate)])}


def test_conversion_preserves_label_and_is_invariant_to_due_date_horizon():
    converted = []
    for horizon in [80., 100., 300.]:
        old = legacy_row(horizon)
        new, error = migration.convert_row(old)
        assert new[TARGET_COLUMN] == old[TARGET_COLUMN]
        assert error < 1e-10
        converted.append(json.loads(new['gnn_node_features'])[0])
    for x in converted:
        assert x == pytest.approx([40/150, .07*150/10, 2.5/5, 1/(60*.07)])


def test_migration_rejects_labels_for_another_schedule():
    row = legacy_row();row[TARGET_COLUMN] = '[1000]'
    with pytest.raises(ValueError, match='Reconstructed labels'):
        migration.convert_row(row)


def test_absolute_repair_scale_is_retained():
    # Scaling all times by two preserves the first three inputs, but doubles
    # the expectation and the fourth input; no hidden output-unit ambiguity.
    x = local_buffer_node_features(40, 150, 2.5, .07)
    y = local_buffer_node_features(80, 300, 2.5, .035)
    assert x[:3] == y[:3]
    assert y[3] == 2*x[3]
    assert operation_expectation(80, 300, 2.5, .035) == pytest.approx(2*operation_expectation(40, 150, 2.5, .07))


def test_cached_zero_edge_ablation_does_not_modify_original_graph():
    g = Data(x=torch.ones(2, 4), edge_index=torch.tensor([[0], [1]]),
             job_edge_index=torch.tensor([[0], [1]]), job_y=torch.tensor([1.]),
             job_membership=torch.zeros(2, dtype=torch.long), num_jobs_tensor=torch.tensor([1]))
    loader = training.CachedGraphBatches([g], 1)
    model = training.FJSPGraphSAGE(4, 4, 2, 'sage')
    before = training._evaluate(model, loader)
    training._evaluate(model, loader, zero_edges=True)
    assert loader.batches[0].edge_index.shape == (2, 1)
    assert loader.batches[0].job_edge_index.shape == (2, 1)
    assert training._evaluate(model, loader) == before


def test_training_can_keep_test_set_unopened(tmp_path, monkeypatch):
    from helper.sequence_setup import reliability_node_feature_names, ReliabilityGraphConfig
    from helper.local_buffer import LABEL_METHOD
    calls=[]
    graph=Data(x=torch.ones(2,4),edge_index=torch.tensor([[0],[1]]),
        job_edge_index=torch.tensor([[0],[1]]),job_y=torch.tensor([2.]),
        job_membership=torch.zeros(2,dtype=torch.long),num_jobs_tensor=torch.tensor([1]))
    train_path,valid_path=tmp_path/'train.csv',tmp_path/'valid.csv'
    train_path.write_text('optimization_run\n0\n')
    valid_path.write_text('optimization_run\n0\n')
    monkeypatch.setattr(training,'_validate_dataset_context',lambda *a,**k:None)
    def load(path):
        calls.append(path)
        return [graph.clone()],reliability_node_feature_names(),ReliabilityGraphConfig(),LABEL_METHOD
    monkeypatch.setattr(training,'load_graphs',load)
    _,meta=training.train_from_file(train_path,valid_path,None,epochs=2,convolution='job',
        loss_name='mse',cache_batches=True,device='cpu',model_dir=tmp_path/'weights')
    result=json.loads(meta.read_text())
    assert calls==[train_path,valid_path]
    assert result['test_evaluated'] is False
    assert result['test_metrics'] is None
    assert result['num_test_graphs']==0
    assert result['zero_edge_split']=='valid'
