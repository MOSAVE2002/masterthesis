# Standalone validation helpers

The helpers in this directory are not called by `main.py` and do not modify
the pipeline configuration or generated result files.

Validate every GNN model selected in `config.json` on the first configured
in-distribution test instance:

```bash
python3 tests/validate_gnn_embedding.py
```

Validate only one architecture or use another generated instance:

```bash
python3 tests/validate_gnn_embedding.py \
  --convolution sage \
  --instance i3_k3_o3-5_15 \
  --tolerance 1e-5
```
