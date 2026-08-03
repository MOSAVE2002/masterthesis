# FJSP-Simulation mit Weibull-Ausfallverzögerung und eingebettetem GNN

Das Projekt vergleicht drei Formulierungen des Flexible Job Shop Scheduling
Problems:

- `gurobi`: lineares Basismodell ohne Zuverlässigkeit,
- `gurobi_nonlinear`: exakte Weibull-Ausfallwahrscheinlichkeit mit erwarteter
  Reparaturdauer,
- `gurobi_gnn`: festes FJSP-Kandidatengraph-GNN als Surrogat derselben
  operationenweisen Ausfallverzögerung.

Alle Modelle minimieren den Makespan. Im nichtlinearen und im GNN-Modell wird
die erwartete Ausfallverzögerung direkt zur Bearbeitungsdauer addiert. Die
aktuellen Formulierungen sind budgetfrei: Es gibt keine zusätzliche
Obergrenze für die gesamte Ausfallverzögerung. Stattdessen wirkt jede
Ausfallverzögerung über die effektive Operationsdauer direkt auf den Makespan.
Das lineare Basismodell `gurobi` besitzt keine Zuverlässigkeitsvariablen.

## Ausführung

Der Workflow wird in `config.json` eingestellt und mit

```bash
python3 main.py
```

gestartet. Nach einer Schemaänderung müssen Trainingsdaten und GNN-Modelle neu
erzeugt werden. Alte Modelle mit dem Ziel `total_failure_cost` sind absichtlich
inkompatibel.

Die vier Schalter unter `workflow` steuern die Pipeline unabhängig:

- `create_instances`: erzeugt und speichert die konfigurierten FJSP-Instanzen,
- `generate_training_data`: erzeugt die ausgewählten CSV-Datensplits,
- `train_gnn`: trainiert die konfigurierten GNN-Architekturen,
- `solve`: wertet die unter `solve.solvers` gewählten Modelle aus.

Sind mehrere Schalter aktiv, ist die Reihenfolge Instanzerzeugung,
Datengenerierung, GNN-Training und abschließend Solverauswertung.

### Voraussetzungen

Benötigt werden Python, eine funktionsfähige Gurobi-Installation samt Lizenz
sowie die Python-Pakete `gurobipy`, `numpy`, `torch` und `torch-geometric`.
Für die optionalen Plot-Funktionen werden zusätzlich `matplotlib` und
`networkx` benötigt.

### Projektstruktur

```text
FJSP_Simulation/
├── main.py                         # zentraler Workflow
├── config.json                     # Experiment- und Solverkonfiguration
├── 01_generator/                   # Instanzerzeugung und Splitverwaltung
├── 02_data/                        # Instanzen, CSV-Daten und Ergebnisse
├── 03_Gurobi/                      # lineare, nichtlineare und GNN-Modelle
├── 04_GraphNeuralNetworks/models/  # GNN-Architektur, Training und Datengenerator
├── helper/                         # gemeinsam verwendete Solver-Hilfen
└── 00_Archiv/                      # nicht aktiver historischer Bestand
```

`00_Archiv` gehört nicht zum aktiven Importgraphen und wird vom Workflow nicht
verwendet.

## Exaktes Weibull-Modell mit Vorgängerübergängen

Sei \(j\) der direkte Vorgänger von Operation \(i\) auf Maschine \(k\).
Der normierte Bearbeitungszeitsprung ist
\(q_{j,i,k}=|p_{i,k}-p_{j,k}|/p_k^{\max}\); für die erste Operation einer
Maschine gilt \(q_i=0\). Die Ausfallwahrscheinlichkeit lautet:

\[
\pi_i=1-\exp\left(-\left[
\left(\frac{R_i+p_i}{\eta_i}\right)^\beta
-\left(\frac{R_i}{\eta_i}\right)^\beta
+\gamma q_i\right]\right).
\]

Mit erwarteter Reparaturdauer \(\tau_k\) gilt

\[
\Delta_i=\sum_{k\in\mathcal M_i}\tau_k\pi_{i,k},
\qquad
D_i=\sum_{k\in\mathcal M_i}Y_{i,k}p_{i,k}+\Delta_i.
\]

\(D_i\) wird in Jobpräzedenz, Maschinenpräzedenz, Start-/Endzeitkopplung und
Makespan eingesetzt.

## GNN-Trainingsziel

Das GNN sagt für jede Operation die Ausfallwahrscheinlichkeit

\[
\widehat\pi_i\in[0,1]
\]

vor. Die maschinenabhängige Reparaturdauer wird erst danach exakt ausgewählt:

\[
\widehat\Delta_i
=\sum_k\tau_{i,k}Y_{i,k}\widehat\pi_i,\qquad
\widehat\Delta_{\mathrm{total}}=\sum_i\widehat\Delta_i.
\]

CSV-Spalten:

- `total_failure_delay`,
- `operation_failure_delays`,
- `operation_failure_probabilities`,
- `gnn_node_features`,
- `gnn_active_edge_indices`,
- `gnn_active_edge_features`,
- `reliability_graph_parameters`.

Die für die Datengenerierung ausgewählten CSV-Splits werden vollständig neu
erstellt. Sie speichern nur aktive Kanten des festen Supergraphen; inaktive
Nullkanten werden nicht redundant in jede Zeile geschrieben.

## Datengenerierung auswählen

Die Art der GNN-Trainingsdaten wird in `config.json` über
`training.data_generation.method` ausgewählt.

Für zufällig erzeugte zulässige Ablaufpläne:

```json
{
  "training": {
    "data_generation": {
      "method": "random_feasible"
    }
  }
}
```

Dabei entstehen im konfigurierten `output_directory`:

- `training/graphs_training.csv`,
- `valid/graphs_valid.csv`,
- `test/graphs_test.csv`.

Für mit Gurobi und fixierten Maschinenzuordnungen erzeugte Lösungsdaten:

```json
{
  "training": {
    "data_generation": {
      "method": "gurobi_fixed",
      "fixed_y": {
        "time_limit_seconds": 5,
        "fix_ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "minimum_ratio_by_max_dimension": {
          "5": 0.3,
          "7": 0.5
        }
      }
    }
  }
}
```

Dabei entstehen getrennte Dateien:

- `training/graphs_fixed_training.csv`,
- `valid/graphs_fixed_valid.csv`,
- `test/graphs_fixed_test.csv`.

Die aktuell konfigurierte Variante `gurobi_linear_labeled` verwendet dieselben
Dateinamen, trennt Optimierung und Labelberechnung aber in zwei Stufen:

```json
{
  "training": {
    "data_generation": {
      "method": "gurobi_linear_labeled"
    }
  }
}
```

Zuerst erzeugt ein lineares, budgetfreies FJSP-Modell unterschiedliche
zulässige Ablaufpläne. Anschließend werden die exakten Weibull-Wahrscheinlich-
keiten und Ausfallverzögerungen für jeden festen Ablauf analytisch berechnet.
Dadurch beeinflusst die nichtlineare Reliability-Funktion nicht die Suche nach
den Trainingsplänen, liefert aber weiterhin die exakten GNN-Zielwerte.

`time_limit_seconds` ist das Gurobi-Zeitlimit pro Optimierungslauf der beiden
Gurobi-basierten Datengeneratoren und muss größer als null sein. Der
Standardwert ist 5 Sekunden. Dieses spezielle Limit überschreibt dabei das
allgemeine `solvers.gurobi.common.TimeLimit`, verändert aber keine normalen
Solverläufe.

`minimum_ratio_by_max_dimension` erhöht die Mindest-Fixed-Ratio abhängig von
der größeren Dimension aus Job- und Maschinenanzahl. Mit der obigen
Konfiguration gilt:

- Dimension 3: unveränderte `fix_ratios`,
- Dimension 5 oder 6: nur Ratios ab `0.3`,
- ab Dimension 7: nur Ratios ab `0.5`.

Dadurch werden für größere Instanzen die besonders langsamen Fixed-Läufe mit
wenigen fixierten Maschinenzuordnungen vermieden. Die Schwellen und
Mindestwerte können in `config.json` angepasst oder durch ein leeres Objekt
`{}` deaktiviert werden.

Die drei Varianten überschreiben sich nicht gegenseitig. Wenn
`workflow.train_gnn` aktiviert ist, wählt das Training anhand von `method`
automatisch die passenden drei CSV-Dateien aus.

Welche Splits neu erzeugt werden, lässt sich unabhängig einstellen:

```json
"generate_splits": {
  "training": true,
  "valid": true,
  "test": true
}
```

Nur mit `true` markierte Splits werden berechnet und neu geschrieben.
Nicht ausgewählte CSV-Dateien bleiben unverändert erhalten. Dadurch kann
beispielsweise ausschließlich der Testdatensatz neu erstellt werden. Wenn
danach direkt trainiert werden soll, müssen trotzdem vollständige
Training-, Validierungs- und Testdateien vorhanden sein.

Die Workflow-Schalter sind unabhängig:

- `generate_training_data: true` erzeugt die ausgewählten CSV-Daten neu,
- `train_gnn: true` trainiert mit den bereits vorhandenen CSV-Daten,
- stehen beide Werte auf `true`, werden zuerst die Daten erzeugt und danach
  wird das GNN trainiert.

Bei `generate_training_data: false` müssen alle drei zum ausgewählten `method`
gehörenden CSV-Dateien bereits vollständig vorhanden sein.

Nach jedem GNN-Training werden die Laufzeiten im Terminal ausgegeben und unter
`training_time` in der zugehörigen `*_meta.json` gespeichert. Enthalten sind
die Wall-Clock-Zeit, Zeiten je Epoche, die Zeit bis zur besten Epoche sowie
GPU-Stunden und GPU-Tage. Eine integrierte Apple-GPU wird als eine GPU gezählt;
ihre Kernanzahl ist kein Multiplikator. Die Messung synchronisiert das
MPS-Backend, bevor die Zeit abgelesen wird. Verfügbare Hardware- und
PyTorch-Informationen stehen zusätzlich unter `training_hardware`. Beim
reinen CPU-Training bleiben `gpu_hours` und `gpu_days` leer.

Die Datengenerierung verwendet ausschließlich die unter
`instances.generation` konfigurierten Größen und Anzahlen. Alte Pickle-Dateien
anderer Größen dürfen im Instanzordner verbleiben und werden ignoriert.
Berücksichtigt werden auch `operations_per_job`, `instances_per_size`,
`split_ratios` und `random_seed`. Fehlt eine konfigurierte Instanz oder passen
ihre gespeicherten Metadaten nicht, fordert das Programm dazu auf,
`workflow.create_instances` einmal zu aktivieren.

## Knoten- und Kantenfeatures

Jeder Knoten verwendet genau drei Features:

1. `processing_time_over_eta`,
2. `machine_age_over_eta`,
3. `weibull_beta_over_5`.

Die Reparaturdauer ist bewusst kein GNN-Feature mehr. Das GNN lernt die
Ausfallwahrscheinlichkeit einer Operation; die ausgewählte Reparaturdauer wird
erst danach im MILP zur erwarteten Verzögerung verrechnet.

Die direkte Vorgängerkante \((j,i,k)\) besitzt:

\[
e_{j,i,k}=
\left[
\frac{|p_{i,k}-p_{j,k}|}{p_k^{\max}},
\frac{p_{j,k}}{\eta_k}
\right].
\]

Die Edge-Features gehen ausschließlich in Message-Passing-Nachrichten ein.
`linear` ist deshalb eine echte graphfreie Node-NN-Baseline. Knoten-, Kanten- und
Reliability-Graph-Schema werden in den Metadaten gespeichert und beim
Einbetten strikt geprüft.

## Fester Graph und Architekturen

`fixed_candidate` ist der feste Supergraph aller möglichen gerichteten
Maschinenkanten \((j,i,k)\). Die binäre Variable \(U_{j,i,k}\) aktiviert eine
Kante genau dann, wenn \(j\) der direkte Maschinenvorgänger von \(i\) ist.
Jede ausgewählte Operation besitzt höchstens eine solche eingehende Kante.
Deshalb sind `mean` und `sum` bei den Vorgängern mathematisch identisch und
bleiben ohne variable Division MILP-einbettbar.

Alle Architekturen erhalten denselben Knotenvektor

\[
x_i=\left[
\frac{p_i}{\eta_{k(i)}},
\frac{R_i}{\eta_{k(i)}},
\frac{\beta}{5}
\right].
\]

Trainierbar und in Gurobi einbettbar sind alle aktuellen Kombinationen:

- `linear` mit `aggregation: none`,
- `gcn` mit `aggregation: mean` oder `sum`,
- `sage` mit `aggregation: mean` oder `sum`,
- `mpnn` mit `aggregation: sum`,
- `gine` mit `aggregation: sum`.

`linear` verarbeitet jeden Knoten lokal und dient als graphfreie
NN-Baseline. `gcn` transformiert die gemeinsame Root-/Nachbarschafts-
aggregation. `sage` besitzt getrennte Gewichte für Root und Nachbarn:

\[
h_i^{(1)}
=\operatorname{ReLU}\left(
W_{\mathrm{root}}x_i+
W_{\mathrm{neigh}}\operatorname{AGG}_{j\in\mathcal N(i)}x_j+b
\right).
\]

`mpnn` bildet eine kantenkonditionierte ReLU-Nachricht und addiert sie zur
separaten Root-Transformation. `gine` verwendet
\(\operatorname{ReLU}(h_j+W_e e_{ji})\), einen lernbaren Root-Faktor
\(1+\epsilon\) und ein zweistufiges ReLU-MLP. Da höchstens ein direkter
Vorgänger aktiv ist, bleiben beide Varianten mit reiner Sum-Aggregation exakt
linear einbettbar.

Bis zu drei Schichten verwenden wieder denselben Architekturtyp. Der
operationenweise Ausgabekopf besitzt zusätzlich einen Skip über \(x_i\) und
liefert eine auf \([0,1]\) begrenzte Ausfallwahrscheinlichkeit
\(\widehat\pi_i\). Trainiert wird direkt gegen die exakte
Operationswahrscheinlichkeit. Alte Age-Message-CSV-Dateien werden absichtlich
abgelehnt, weil sie eine andere Zuverlässigkeitsgleichung repräsentieren.

## GNN-Einbettung in Gurobi

Die trainierten Gewichte sind Konstanten. Das Produkt aus direktem
Vorgänger-Gate \(U_{j,i,k}\) und einem beschränkten Hidden-Feature wird mit
Big-M exakt linearisiert. Die beiden Kantenfeatures sind Konstanten und werden
direkt mit \(U_{j,i,k}\) gewichtet. Jede Transformation und jede ReLU wird in
Gurobi als Variable und Nebenbedingung aufgebaut. Weil die Reparaturdauer von
der Maschinenwahl abhängt, werden zusätzlich Hilfsvariablen

\[
Z_{i,k}=Y_{i,k}\widehat\pi_i
\]

mit den exakten Binär-Kontinuierlich-Bedingungen

\[
0\le Z_{i,k}\le \widehat\pi_i,\qquad
Z_{i,k}\le Y_{i,k},\qquad
Z_{i,k}\ge \widehat\pi_i-(1-Y_{i,k})
\]

linearisiert. Für jede Operation gilt anschließend:

\[
\widehat\Delta_i=\sum_k\tau_{i,k}Z_{i,k},
\qquad
D_i
=\sum_kY_{i,k}p_{i,k}
+\widehat\Delta_i.
\]

GNN-Ausgabe liegt damit direkt auf dem Zielfunktionspfad:

\[
(Y,X,U)\rightarrow\text{direkte Vorgänger und Übergangsfeatures}\rightarrow\mathrm{GNN}
\rightarrow\widehat\pi\rightarrow
Y\widehat\pi\rightarrow\widehat\Delta\rightarrow D\rightarrow C_{\max}.
\]

Nach dem Training werden alle Message-Passing-Modelle zusätzlich ohne Kanten
ausgewertet. Die Ablation wird in den Metadaten gespeichert.
Ob ein fehlender Grapheneinfluss das Speichern verhindern soll, wird über
`training.gnn.validation.enforce_graph_influence` konfiguriert.

## Konfiguration

Relevante Felder:

- `training.data_generation.reliability_ranges.repair_duration`,
- `constraint.weibull.reliability_graph.beta` für die Weibull-Form,
- `constraint.weibull.reliability_graph.transition_gamma` für die Stärke des
  Vorgängerübergangs,
- `training.gnn.combinations`,
- `architecture_ablation.layers` und
  `architecture_ablation.hidden_channels` als gemeinsames Trainingsraster
  für alle Einträge in `training.gnn.combinations`,
- `training.gnn.validation.interval_epochs` für den Abstand zwischen zwei
  Validierungsläufen; der Standardwert ist `10`,
- `training.gnn.validation.patience_epochs` für Early Stopping nach einer
  festgelegten Anzahl Epochen ohne Verbesserung; der Standardwert ist `10`.

`layers` und `hidden_channels` dürfen jeweils eine einzelne Ganzzahl oder eine
Liste sein. Werden beide als Listen angegeben, wird ihr kartesisches Produkt
trainiert:

```json
"architecture_ablation": {
  "layers": [1, 2, 3],
  "hidden_channels": [16, 64]
}
```

Dieses Raster erzeugt sechs Modelle pro Eintrag in
`training.gnn.combinations`. Layerzahlen sind derzeit auf `1`, `2` und `3`
begrenzt. Die konkreten Hyperparameter stehen in Verzeichnis und Dateiname,
beispielsweise:

```text
fixed_candidate/sage_sum_global_add_layers3_hidden64/
fjsp_gnn_fixed_candidate_sage_sum_global_add_layers3_hidden64_total_failure_delay_seed42.pt
```

Die Instanzen bleiben zwischen Training, Validierung und Test disjunkt. Bei
`gurobi_fixed` und `gurobi_linear_labeled` wird die Samplezahl je Instanz und
Split über `training.data_generation.samples_per_split` festgelegt. Für
`random_feasible` dient `samples_per_instance` als Rückfallwert.

`constraint.enforce`, `enforce_constraint` und
`weibull_budget_per_operation` werden aus Kompatibilitätsgründen teilweise
noch akzeptiert, erzeugen in den aktuellen nichtlinearen und GNN-Modellen aber
keine Budgetnebenbedingung. Die Formulierungen minimieren die
Ausfallverzögerung ausschließlich indirekt über die effektiven Dauern und
`C_max`.

## Prüfungen

Im Repository ist derzeit keine automatisierte Testsuite enthalten. Für eine
schnelle Syntaxprüfung aller aktiven Module kann ausgeführt werden:

```bash
python3 -m compileall -q main.py 01_generator 03_Gurobi \
  04_GraphNeuralNetworks helper
```

Ein vollständiger Lauf über `python3 main.py` kann je nach aktivierten
Workflow-Schaltern Instanzen oder CSV-Dateien überschreiben und lang laufende
Gurobi- beziehungsweise GNN-Jobs starten. Vorher deshalb immer die vier
Schalter unter `workflow` prüfen.
