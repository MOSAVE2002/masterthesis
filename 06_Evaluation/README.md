# Standalone post-solve evaluation

Die numerische Analyse ist in den normalen Evaluate-Workflow integriert.

## Controlled base-model versus nonlinear-model comparison

Calibrate a nominal makespan for each physical instance, derive the common
due dates as `ceil((1 + offset) * nominal_makespan)`, solve both formulations,
and evaluate both schedules with common-random-number Monte Carlo simulation:

```bash
python3 06_Evaluation/compare_base_nonlinear.py \
  --instances 4 \
  --num-jobs 3 \
  --num-machines 3 \
  --replications 2000
```

The default offsets are read from `training.data_generation.adaptive_due_dates`
and currently equal 0.00, 0.15, 0.30 and 0.45. The controlled comparison
still uses common makespan-relative due dates; the GNN training generator uses
the same offsets for its job-specific, makespan-calibrated TWK factors. The
analysis writes raw schedule comparisons, per-job observations, offset
summaries, repair-buffer calibration summaries and JSON metadata to
`06_Evaluation/results`.

This is a post-solve robustness experiment, not a service-constrained
optimization. A job is successful in one Monte-Carlo replication when its
simulated completion time does not exceed its due date. Repeated simulations
estimate the on-time probability of every job; the result tables then compare
the minimum and mean job probabilities of each schedule across the tested
parameter settings. No simulated probability is returned to the solver.

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
sage_sum_global_add_layers2_hidden32/\
solution_i3_k3_o3-5_15_gurobi_gnn_layers2_hidden32_seed42.txt \
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
- `06_Evaluation/results/operation_comparison.csv`

`result_table.csv` enthält zusätzlich den simulierten Makespan, dessen Anstieg
gegenüber dem Plan, simulierte Gesamtverspätung und Gesamtkosten, gemeinsame
Termintreue aller Jobs sowie aggregierte Right-Shift-Kennzahlen. Quantile werden
innerhalb jeder Monte-Carlo-Auswertung berechnet. `job_comparison.csv` ergänzt
jobbezogene Verzögerungs- und Verspätungsquantile; `operation_comparison.csv`
enthält die Startverschiebungen jeder einzelnen Operation.
Die simulierten Gesamtkosten werden für jede Wiederholung als konstante
Bearbeitungskosten plus Betriebskostensatz mal simuliertem Makespan plus
Verspätungskostensatz mal simulierter Gesamtverspätung neu berechnet. Dadurch
bleiben Abhängigkeiten zwischen Makespan und Verspätung auch in den
Kostenquantilen erhalten.

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

## Chapter-7 numerical analysis

When `evaluation.numerical_analysis.enabled` is true, the normal evaluation
workflow automatically summarizes solver behavior, model sizes, surrogate
quality and Monte-Carlo robustness. It also writes runtime ECDF and performance
profile data and plots below `06_Evaluation/results/numerical_analysis`.

Every newly solved model writes a `*_solver_progress.csv` file next to its
solution text. The callback trajectory contains solver time, incumbent
(primal bound), best bound (dual bound), relative MIP gap, explored nodes,
solution count, event type and objective sense. Changes of incumbent, bound or
solution count are retained, an additional point is retained at the first
MIP/MIPNODE callback after 0.25 seconds without a change, and the final solver state is
always added. Existing solution files remain
readable but cannot retroactively provide this trajectory.

The numerical-analysis step combines and compares these trajectories in:

- `solver_progress.csv` (all raw trajectory rows with instance/model metadata)
- `solver_progress_summary.csv` (common time grid by model/architecture)
- `solver_progress_summary_by_size_due.csv` (additional split by tier, size and due-date condition)
- `solver_gap_over_time.pdf` and `.png` (median relative gap with IQR)
- `solver_incumbent_over_time.pdf` and `.png` (incumbent and target-gap rates)
- `solver_incumbent_improvement_over_time.pdf` and `.png` (normalized improvement from the first incumbent)

The median gap is calculated only for runs that already have a finite gap.
Therefore it must be interpreted together with the displayed incumbent and
target-gap rates. Raw objective values are not averaged across heterogeneous
instances; the incumbent comparison uses the percentage improvement within
each individual run.

Run the complete post-solve evaluation and summary with:

```bash
.venv/bin/python main.py --workflow evaluate
```

With `evaluation.numerical_analysis.strict: true`, completeness problems are
reported after the available diagnostic tables have been written. This now
also checks that every evaluated solver run has a trajectory sidecar.
