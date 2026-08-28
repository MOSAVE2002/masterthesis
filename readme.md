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

## Wirtschaftliche Zielfunktion

Die produktiven Modelle minimieren gemeinsam

\[
\sum_{i,k} c_k p_{ik}Y_{ik}
+c_{Halle}\,C_{\max}.
\]

Der erste Term erfasst die Maschinenkosten während der Bearbeitung. Der zweite
Term bildet die Hallenbetriebskosten bis zum Makespan ab. Die Kostenrate wird
über `objective.facility_cost_per_time` konfiguriert. Dieselbe Zielfunktion
wird im nichtlinearen Modell, im GNN-Modell und bei der Erzeugung der
Kandidatenschedules verwendet.

## Maschinenprofile

Die Instanzgenerierung verwendet zwei bewusst unterschiedliche
Maschinenarten:

- `old`: niedrige Kosten, geringere Geschwindigkeit, Weibull-Formparameter
  \(\beta=3\) und schwächere Zuverlässigkeits- und Reparaturparameter,
- `new`: höhere Kosten, höhere Geschwindigkeit, größere Weibull-Skala,
  Weibull-Formparameter \(\beta=2\) und schnellere Reparatur.

Kosten, Geschwindigkeit, Weibull-Skala, Weibull-Form und Reparaturrate sind
innerhalb jedes Profils über alle Instanzen konstant; der Parameter-Jitter ist
deaktiviert. Jede Operation kann auf Maschinen aus mindestens zwei
Profilklassen ausgeführt werden. Ihre Bearbeitungszeit entsteht
aus einer Basiszeit, dem Geschwindigkeitsfaktor des Profils und einem kleinen
Operationsrauschen. Die
Profile werden unter `instances.generation.machine_profiles` konfiguriert und
bleiben bei der Monte-Carlo-Erzeugung der Trainingslabels unverändert.

## Referenzmodell

`gurobi_nonlinear` bettet das Integral über eine 12-Punkt-Gauss-Legendre-
Quadratur direkt in Gurobi ein. Die Potenzen für \(\beta=2\) und \(\beta=3\)
werden direkt als allgemeine nichtlineare Gurobi-Constraints
(`GENCONSTR_NL`) modelliert. Jede Quadraturstützstelle besitzt eine kompakte
nichtlineare Integranden-Constraint; eine weitere nichtlineare Constraint
definiert daraus \(Pd_{ik}(t_i)\). Die Quadraturgrenzen hängen von der
Entscheidungsvariablen \(t_i\) ab. Zusammen mit Maschinenzuordnung und
Reihenfolge entsteht ein nichtkonvexes MINLP (`NonConvex=2`). Es gibt keinen
Renewal-Callback und keine Lazy Cuts.

Für Operation (i) wird die erwartete Ausfallverzögerung näherungsweise durch

\[
\Delta_i=\sum_k \frac{Pd_{ik}(t_i)Y_{ik}}{\lambda_k}
\]

bestimmt. Für jeden Job wird daraus der erwartete Reparaturpuffer

\[
B_u^{NL}=\sum_{i\in\mathcal O_u}\Delta_i
\]

gebildet. Die nicht fortgepflanzte Due-Date-Constraint lautet

\[
C_u+B_u^{NL}\le d_u\qquad\forall u\in\mathcal J.
\]

Der Alpha-Servicegrad ist keine
Optimierungsconstraint; er wird ausschließlich nach der Optimierung durch die
Monte-Carlo-Simulation bestimmt. Zusammen mit den
Weibull-Integralen bleibt das Referenzmodell ein MINLP. `service_scope` steht
für diese Pipeline auf `"job"`.

Die einzige aktive Simulation verwendet dieselbe operationsbezogene
Midpoint-Snapshot-Wahrscheinlichkeit wie das Referenzmodell. Bedingt auf einen
Ausfall wird eine verbleibende exponentielle Reparaturdauer gezogen. Die
resultierende Verzögerung wird über die festen Job- und Maschinenkanten nach
rechts fortgepflanzt. Damit gilt je Operation konsistent
\(E[\Delta_i]=Pd_i(t_i)/\lambda_i\); es gibt weder eine zusätzliche Skalierung
noch einen Renewal-Prozess oder Leerlaufausfälle.

## GNN-Ersatzmodell

`gurobi_gnn` schätzt für jeden Job den nichtlinearen Reparaturpuffer
\(\widehat B_u\). Die Knotenfeatures sind:

1. nominaler Start und nominale Fertigstellung,
2. Bearbeitungszeit relativ zum Weibull-Skalenparameter,
3. Reparaturrate und Weibull-Formparameter.

Der Due-Date-Slack wird nicht als separates GNN-Feature verwendet. Die Due
Date dient als Normalisierungshorizont für Start- und Fertigstellungszeiten.

Es werden drei bewusst getrennte Ersatzmodelle trainiert und verglichen:

1. `linear`: knotenseitige lineare Transformation ohne Message Passing,
2. `job`: Message Passing ausschließlich über feste Jobpräzedenzkanten,
3. `sage`: vollständiges GraphSAGE über feste Jobpräzedenzkanten und
   entscheidungsabhängige Maschinenkanten.

Nur in der vollständigen `sage`-Variante werden Maschinenkanten durch

\[
U_{ijk}=1
\iff i\text{ ist unmittelbarer Vorgänger von }j\text{ auf Maschine }k
\]

aktiviert. Damit tritt auch das Produkt einer Maschinenkante mit einem
Knotenzustand, also \(U_{ijk}h_j^{(\ell-1)}\), nur bei `sage` auf. Die
`job`-Variante verwendet stattdessen ausschließlich die bereits durch den
Auftrag vorgegebenen Kanten und benötigt deshalb weder \(U_{ijk}\) noch eine
Linearisierung von \(U_{ijk}h_j^{(\ell-1)}\). `linear` verwendet überhaupt
keine Kanten. Ein jobspezifisches Pooling fasst in allen drei Varianten die
Operationsknoten jedes Jobs zusammen. Für jeden Job gilt im eingebetteten
Modell dieselbe unskalierte Constraint wie im Referenzmodell:

\[
C_u+\widehat B_u\le d_u.
\]

Das GNN-Modell ist ein MILP, während das Referenzmodell das schwierige MINLP
bleibt.

## Trainingsdaten

Fix-and-Optimize erzeugt Maschinenzuordnungen und unmittelbare
Maschinenfolgen. Im Modus `nonlinear_evaluated` wird jede feste Struktur vor
der nichtlinearen Auswertung und Simulation kanonisch auf ihren frühestmöglichen
Schedule nach links geschoben. Jeder Schedule-Graph wird in genau einer
CSV-Zeile gespeichert.

Die Trainings-Due-Date wird zunächst durch einen nominalen Makespanlauf
kalibriert und anschließend einheitlich als

\[
d=\left\lceil(1+\delta)C_{\max}^*\right\rceil,
\qquad \delta\in\{0{,}02,0{,}05,0{,}10,0{,}20\},
\]

gesetzt. Dadurch besitzen Training und kontrollierte Modellvergleiche dieselbe
Definition der relativen Due-Date-Slack.

Ein fehlgeschlagener Instanzlauf beendet die Erzeugung nicht mehr. Bleiben für
eine Instanz nach dem normalen Kandidatenlauf zu wenige lösbare Kandidaten
übrig, wird sie sofort übersprungen. Der aktuelle Fortschritt wird nach jeder
Instanz atomar in
`02_data/gnn_dataset/generation_summary.json` gesichert. Die Datei enthält pro
Split alle erfolgreichen und übersprungenen Instanzen einschließlich
Fehlermeldung.
Diese Zeile enthält unter anderem

```text
job_ids=[1,2,3,...]
job_ontime_probabilities=[P_1^MC,P_2^MC,P_3^MC,...]
operation_job_indices=[...]
```

Damit kann ein Graph beliebig viele Jobs und ebenso viele unterschiedliche
Joblabels enthalten. `job_probability_label_method` dokumentiert die
Monte-Carlo-Pünktlichkeitsanteile; die nichtlinearen Markov-Näherungen werden
getrennt im technisch weiterhin so benannten Feld
`nonlinear_job_ontime_probability_lbs` gespeichert.

Die aktive Simulation zieht für jede Operation genau einen Ausfallzustand mit
der nichtlinearen Wahrscheinlichkeit am nominalen Operationsmittelpunkt. Bei
einem Ausfall wird eine verbleibende exponentielle Reparaturdauer addiert. Die
Verzögerung verschiebt alle Nachfolger entlang der festen Job- und
Maschinenkanten nach rechts. Leerlaufausfälle, kontinuierliches Maschinenalter
und wiederholte Ausfälle innerhalb einer Operation sind nicht Bestandteil des
gewählten stochastischen Modells.

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

### Produktionsplan und Graphen plotten

Die Visualisierungen werden im jeweiligen Solverblock aktiviert:

```json
"plot_solution_schedule": true,
"plot_solution_graph": true,
"plot_candidate_graph": true,
"plot_solution_graph_style": "disjunctive",
"plot_output_directory": "plots/fjsp_solution_plots"
```

Dabei entstehen das maschinenbasierte Gantt-Chart, der Lösungsgraph und der
Kandidatengraph. Als Graphstile werden `disjunctive` und `machine_operation`
unterstützt.

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
