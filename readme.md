# Szenariofreies stochastisches FJSP mit Weibull-MINLP und GNN

Die aktive Pipeline verwendet wieder die nichtlineare Ausfallwahrscheinlichkeit
am Mittelpunkt jeder Operation. Für Maschine (k) gelten eine
Weibull-Ausfallzeit und eine exponentielle Reparaturzeit:

\[
f_k(x)=\frac{\beta_k}{\alpha_k}
\left(\frac{x}{\alpha_k}\right)^{\beta_k-1}
e^{-(x/\alpha_k)^{\beta_k}},
\qquad R_k\sim\operatorname{Exp}(\lambda_k).
\]

Mit

\[
t_i=\frac{S_i+C_i}{2}
\]

lautet die verwendete Wahrscheinlichkeit, dass Maschine (k) zum
Auswertezeitpunkt wegen eines Ausfalls noch in Reparatur ist:

\[
Pd_{ik}(t_i)=\int_0^{t_i} f_k(x)e^{-\lambda_k(t_i-x)}\,dx.
\]

## Referenzmodell

`gurobi_nonlinear` bettet das Integral über eine 12-Punkt-Gauss-Legendre-
Quadratur direkt in Gurobi ein. Die Quadraturgrenzen hängen von der
Entscheidungsvariablen (t_i) ab. Zusammen mit Maschinenzuordnung und
Reihenfolge entsteht damit wieder ein nichtkonvexes nichtlineares MIP
(MINLP, `NonConvex=2`). Es gibt keinen Renewal-Callback und keine Lazy Cuts.

Für Operation (i) wird die erwartete Ausfallverzögerung durch

\[
\Delta_i=\sum_k \frac{Pd_{ik}(t_i)Y_{ik}}{\lambda_k}
\]

approximiert. Für jeden Job wird die jobspezifische Markov-Untergrenze

\[
P_u^{LB}=\max\left(0,
1-\frac{\sum_{i\in\mathcal O_u}\Delta_i}
{dd_u-C_{i_u^*}}\right)
\]

gebildet und unmittelbar gefordert:

\[
P_u^{LB}\ge\alpha_u\qquad\forall u\in\mathcal J.
\]

In Gurobi wird die Definition durch eine nichtkonvexe quadratische Gleichung
mit einer expliziten Wahrscheinlichkeitsvariablen abgebildet. Zusammen mit den
Weibull-Integralen bleibt das Referenzmodell ein MINLP. `service_scope` steht
für diese Pipeline auf `"job"`.

## GNN-Ersatzmodell

`gurobi_gnn` sagt für jeden Job eine eigene Pünktlichkeits-Untergrenze
\(\widehat P_u^{LB}\) voraus. Die Knotenfeatures sind:

1. nominaler Start und nominale Fertigstellung,
2. Bearbeitungszeit relativ zum Weibull-Skalenparameter,
3. Reparaturrate und Weibull-Formparameter.

Der normalisierte Due-Date-Slack wird nicht als separates GNN-Feature
verwendet. Da alle Jobs einer Instanz dieselbe Due Date besitzen und diese als
Normalisierungshorizont dient, ist er mit
\((d-C_i)/d=1-C_i/d\) exakt durch die normalisierte Fertigstellungszeit
bestimmt. Die Slack-Hilfsvariable des nichtlinearen Referenzmodells bleibt
davon unberührt; sie bildet dort den zeitlichen Puffer
\(s_u=d_u-C_u\) in der Wahrscheinlichkeitsgleichung ab.

Die Graphkanten werden durch

\[
U_{ijk}=1
\iff i\text{ ist unmittelbarer Vorgänger von }j\text{ auf Maschine }k
\]

aktiviert. Zusätzlich enthält der Graph feste Jobpräzedenzkanten. Ein
jobspezifisches Pooling fasst die Operationsknoten jedes Jobs zusammen. Für
jeden Job gilt im eingebetteten Modell

\[
\widehat P_u^{LB}\ge\alpha_u.
\]

Das GNN-Modell ist ein MILP, während das Referenzmodell das schwierige MINLP
bleibt.

## Trainingsdaten

Fix-and-Optimize erzeugt Maschinenzuordnungen und unmittelbare
Maschinenfolgen. Im Modus `nonlinear_evaluated` wird jede feste Struktur vor
der nichtlinearen Auswertung und Simulation kanonisch auf ihren frühestmöglichen
Schedule nach links geschoben. Jeder Schedule-Graph wird in genau einer
CSV-Zeile gespeichert.
Diese Zeile enthält unter anderem

```text
job_ids=[1,2,3,...]
job_ontime_probabilities=[P_1^MC,P_2^MC,P_3^MC,...]
operation_job_indices=[...]
```

Damit kann ein Graph beliebig viele Jobs und ebenso viele unterschiedliche
Joblabels enthalten. `job_probability_label_method` dokumentiert die
Monte-Carlo-Pünktlichkeitsanteile; die nichtlinearen Markov-Untergrenzen werden
getrennt in `nonlinear_job_ontime_probability_lbs` gespeichert.

Die Daten werden unter `02_data/gnn_dataset` geschrieben. Die aktiven Modelle
liegen unter
`04_GraphNeuralNetworks/trained_gnn_models/job_ontime_v3`.

## Ausführung

### Voraussetzungen

Die Pipeline wurde mit Python 3.13.1 und den in `requirements.txt` exakt
festgehaltenen Paketversionen getestet. Für die Optimierungsphasen wird
zusätzlich eine gültige Gurobi-Lizenz benötigt.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Die Installation kann vor dem ersten Pipeline-Lauf geprüft werden:

```bash
python3 -c "import gurobipy, numpy, torch, torch_geometric"
python3 -m pip check
```

### Pipeline starten

```bash
python3 main.py
```

`config.json` steuert die fünf Phasen `create_instances`,
`generate_training_data`, `train_gnn`, `solve` und `evaluate`. Die optionale
Auswertung simuliert die gespeicherten Solver-Schedules unabhängig nach und
schreibt die Vergleichstabellen nach `06_Evaluation/results`. Die Instanzgrößen für
Training und die in-distribution Evaluation werden zentral über
`instances.generation.num_jobs` und `num_machines` festgelegt.

## Zentrale Dateien

```text
03_Gurobi/build_fjsp_with_nonlinear.py
03_Gurobi/build_fjsp_with_gnn.py
04_GraphNeuralNetworks/models/generate_fix_and_optimize_training_data.py
04_GraphNeuralNetworks/models/model_training_FJSP_GNN.py
helper/stochastic_fjsp.py
helper/sequence_setup.py
helper/gurobi_solution_writer.py
config.json
```
