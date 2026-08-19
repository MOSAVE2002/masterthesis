# Standalone post-solve evaluation

The evaluator reconstructs the schedules stored in existing solution text
files, labels them with fresh Monte-Carlo replications, and writes comparison
tables. Both direct execution and the final phase of `main.py` require
`workflow.evaluate` to be `true` in `config.json`.

Evaluate every current nonlinear and GNN solution with 10,000 replications:

```bash
python3 06_Evaluation/evaluate_solutions.py
```

Run it through the configured workflow, independently of whether solving is
enabled in the same invocation:

```json
"workflow": {
  "solve": false,
  "evaluate": true
}
```

The evaluation settings are configured separately:

```json
"evaluation": {
  "solutions_directory": "02_data/fjsp_solutions",
  "instances_directory": "02_data/fjsp_instances",
  "output_directory": "06_Evaluation/results",
  "replications": 10000,
  "random_seed": 900042,
  "wilson_confidence": 0.95
}
```

Evaluate only selected files or use fewer replications for a smoke test:

```bash
python3 06_Evaluation/evaluate_solutions.py \
  02_data/fjsp_solutions/gurobi_gnn/fixed_candidate/\
sage_sum_global_add_layers1_hidden4/\
solution_i3_k3_o3-5_15_gurobi_gnn_layers1_hidden4_seed42.txt \
  --replications 500
```

The result tables are generated only when the evaluation phase runs. In the
pipeline this requires:

```json
"workflow": {
  "evaluate": true
}
```

The default outputs are:

- `06_Evaluation/results/result_table.csv`
- `06_Evaluation/results/result_table.tex`
- `06_Evaluation/results/result_table.pdf`
- `06_Evaluation/results/job_comparison.csv`

## GNN data and calibration diagnostics

When `evaluation.dataset_diagnostics.enabled` is true,
`evaluate_gnn_dataset.py` additionally reads the generated training,
validation and test CSV files. It evaluates label coverage around the service
threshold and compares the data with the constant training-mean baseline.

Generated files:

- `06_Evaluation/results/gnn_dataset_distribution.csv`
- `06_Evaluation/results/gnn_constant_baseline.csv`
- `06_Evaluation/results/gnn_label_histograms.pdf`
- `06_Evaluation/results/gnn_label_histograms.png`
- `06_Evaluation/results/gnn_dataset_diagnostics.json`

When `evaluation.prediction_diagnostics.enabled` is true,
`evaluate_gnn_predictions.py` compares the probabilities embedded in solved
GNN schedules with the independent Monte-Carlo probabilities from
`job_comparison.csv`.

Generated files:

- `06_Evaluation/results/gnn_prediction_metrics.csv`
- `06_Evaluation/results/gnn_calibration.csv`
- `06_Evaluation/results/gnn_calibration.pdf`
- `06_Evaluation/results/gnn_calibration.png`
- `06_Evaluation/results/gnn_prediction_diagnostics.json`

Both diagnostics are part of the normal evaluation workflow and run there
only when:

```json
"workflow": {
  "evaluate": true
}
```

They can also be started directly. An explicit script call always runs,
independently of `workflow.evaluate`:

```bash
python3 06_Evaluation/evaluate_gnn_dataset.py
python3 06_Evaluation/evaluate_gnn_predictions.py
```

The dataset evaluator deliberately rejects CSV files whose stored service
level differs from the active configuration. Regenerate old datasets before
running diagnostics after changing `service_level`.

## Nonlinear probability-band analysis

Compare the nonlinear Markov lower bound with independent Monte-Carlo
probabilities for several small instances:

```bash
python3 06_Evaluation/analyze_nonlinear_probability_bands.py \
  --instances 5 --replications 1000 --time-limit 10
```

Explicit instances can be selected by repeating `--instance-name`. The script
runs independently of the workflow flags and writes the raw observations,
band summary, JSON metadata, PDF, and PNG to `06_Evaluation/results`.

`mc_point_feasible` evaluates the Monte-Carlo point estimate against the
configured service target. `wilson_feasible` additionally reports whether the
one-sided Wilson lower confidence bound reaches that target.
`bonferroni_wilson_feasible` uses the per-job Bonferroni correction required
for a joint confidence statement over all jobs of one schedule. Both Wilson
values are diagnostic only and are not fed back into either optimization
model.

## Two-stage candidate-generation analysis

Evaluate the configured `nonlinear_evaluated` Fix-and-Optimize pipeline on
small instances without changing the existing GNN dataset:

```bash
python3 06_Evaluation/analyze_two_stage_candidate_generation.py \
  --instances 5 --samples-per-instance 15
```

The script generates candidates with the linear Gurobi solution pool,
left-shifts each fixed job/machine predecessor graph to its canonical
earliest-start timing,
evaluates the nonlinear midpoint/Weibull/Markov expression on every fixed
schedule, selects candidates using pilot Monte Carlo probabilities, and uses
the configured final Monte Carlo replication count for the labels. Raw and
summary CSV files, JSON metadata, PDF, and PNG are written to
`06_Evaluation/results`.
