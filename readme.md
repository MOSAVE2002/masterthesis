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

## Wirtschaftliche Zielfunktion und weiches Serviceziel

Die produktiven Modelle minimieren gemeinsam

\[
\sum_{i,k} c_k p_{ik}Y_{ik}
+c_{Halle}\,C_{\max}
+c_{SL}\sum_{u\in\mathcal J}L_u^{SL}.
\]

Der erste Term erfasst die Maschinenkosten während der Bearbeitung. Der zweite
Term bildet die Hallenbetriebskosten bis zum Makespan ab. Der dritte Term
bestraft die zeitliche Verletzung des weichen Service-Level-Ziels. Die
Kostenraten werden über `objective.facility_cost_per_time` und
`objective.service_violation_cost_per_time` konfiguriert. Beide stehen aktuell
auf eins. Dieselbe Kostenstruktur wird im Grundmodell, im nichtlinearen Modell
und im GNN-Modell verwendet.

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
Profilklassen ausgeführt werden. Ihre Bearbeitungszeit entsteht aus einer
ganzzahligen Basiszeit im konfigurierten Bereich `[10, 30]`, dem
Geschwindigkeitsfaktor des Profils und einem kleinen Operationsrauschen. Die
Profile werden unter `instances.generation.machine_profiles` konfiguriert und
bleiben bei der analytischen Erzeugung der Trainingslabels unverändert.
Eine Zeiteinheit entspricht zehn Minuten. Weil der Basiszeitbereich gegenüber
der früheren Konfiguration auf `[10, 30]` verdoppelt wurde, wurden auch die
Weibull-Skalen und mittleren Reparaturzeiten proportional verdoppelt. Damit
bleiben Bearbeitungszeit/Ausfallskala und Bearbeitungszeit/Reparaturdauer
vergleichbar. Die Weibull-Skalen betragen 120 ZE (`old`) und 200 ZE (`new`),
die mittleren exponentiellen Reparaturzeiten 60 ZE beziehungsweise zehn Stunden
für `old` und 30 ZE beziehungsweise fünf Stunden für `new`.

## Due Dates der generierten Instanzen

Die reguläre Instanz- und Benchmarkgenerierung verwendet jobspezifische Due
Dates nach der Total-Work-Content-Methode (TWK). Da die Maschinenzuordnung zum
Generierungszeitpunkt noch nicht feststeht, wird für jede Operation die mittlere
Bearbeitungszeit über ihre zulässigen Maschinen verwendet:

\[
\bar p_i=\frac{1}{|\mathcal M_i|}\sum_{k\in\mathcal M_i}p_{ik},
\qquad
d_u=\left\lceil h\sum_{i\in\mathcal O_u}\bar p_i\right\rceil.
\]

Der Faktor \(h\) wird über `instances.generation.due_dates.factors`
konfiguriert und zyklisch den Instanzen zugewiesen. Dadurch erhalten Jobs mit
unterschiedlichem Arbeitsinhalt im Allgemeinen unterschiedliche Due Dates. Die
unter `training.data_generation.adaptive_due_dates` konfigurierte spätere
Kalibrierung der Trainingskopien ist davon getrennt. Dazu wird einmal ein
nominaler Makespan \(C^*_{\max}\) bestimmt und der instanzspezifische
Ausgangsfaktor

\[
h_0=\frac{C^*_{\max}}{\frac{1}{|J|}\sum_{u\in J}\sum_{i\in\mathcal O_u}\bar p_i}
\]

berechnet. Für Offset \(\delta\) gilt dann
\(h=(1+\delta)h_0\) und jeder Job behält seine TWK-Due-Date
\(d_u=\lceil h\sum_i\bar p_i\rceil\). Die Standardoffsets
`[0.00, 0.15, 0.30, 0.45]` erzeugen enge, mittlere und lockere
Trainingsvarianten derselben physischen Instanz.

Der Solverbenchmark verwendet dieselbe Kalibrierung mit den Offsets
`[0.00, 0.15, 0.30]`. Die drei Varianten einer Instanz besitzen dieselbe
physische Struktur, Maschinenparameter und nominale Makespan-Kalibrierung; nur
der Faktor \(h=(1+\delta)h_0\) und damit die Due Dates ändern sich. Sie werden
getrennt unter `benchmark/twk_d0p00`, `benchmark/twk_d0p15` und
`benchmark/twk_d0p30` gespeichert und je Stufe von allen Solvern unverändert
wiederverwendet. So lassen sich enge bis lockere Termine als kontrollierter
Faktor testen, ohne die Instanzstruktur oder die Maschinenknappheit mitzuwandeln.
Für lokale und mittlere Läufe ist die nominale Kalibrierung auf zehn Sekunden
pro physischer Instanz eingestellt; auf dem Cluster kann dieses Budget bei
größeren Instanzen weiter erhöht werden.

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

gebildet. Mit dem Servicegrad \(\alpha\) und der nichtnegativen weichen
Verletzungsvariablen \(L_u^{SL}\) lautet die konservative Markov-Constraint

\[
C_u+\frac{B_u^{NL}}{1-\alpha}\le d_u+L_u^{SL},
\qquad L_u^{SL}\ge0,
\qquad\forall u\in\mathcal J.
\]

Aktuell gilt \(\alpha=0{,}9\), also der Skalierungsfaktor zehn. Bei
\(L_u^{SL}=0\) fordert die Constraint die konservative Markov-Untergrenze von
90 Prozent. Bei positiver Verletzung ist sie ausdrücklich ein weiches Ziel und
keine Wahrscheinlichkeitsgarantie. Im Grundmodell gilt \(B_u=0\); dort misst
dieselbe Kostenkomponente nur die nominale Due-Date-Verletzung. Zusammen mit
den Weibull-Integralen bleibt das Referenzmodell ein MINLP.

Die einzige aktive Simulation zieht je Replikation und eingesetzter Maschine
einen Weibull-verteilten ersten Ausfallzeitpunkt sowie eine exponentielle
Reparaturdauer. Daraus entsteht ein gemeinsames Ausfallintervall der Maschine.
Beginnt eine Operation innerhalb dieses Intervalls, wartet sie bis zum Ende der
Reparatur. Tritt der Ausfall während der Bearbeitung auf, gilt Preempt-Resume:
Die bereits geleistete Arbeit bleibt erhalten und die Bearbeitung wird nach der
Reparatur fortgesetzt. Die resultierende Verzögerung wird über die festen Job-
und Maschinenkanten nach rechts fortgepflanzt.

Damit bildet die Simulation im Gegensatz zur operationsbezogenen
Midpoint-Näherung auch Ausfälle in Leerlaufzeiten und eine konsistente
Maschinenhistorie ab. Nach der ersten Reparatur wird jedoch kein weiterer
Ausfall erzeugt; es handelt sich also nicht um einen Renewal-Prozess. Die
simulierten Verzögerungen sind bewusst ein eigenständiges Trainings- und
Evaluationsziel und müssen nicht operationsweise mit
\(Pd_i(t_i)/\lambda_i\) übereinstimmen.

Die Simulation verändert einen festen Schedule nicht und gibt keine
Nebenbedingung an den Solver zurück. Sie wird sowohl für simulationsbasierte
GNN-Trainingslabels als auch mit frischen Seeds für die unabhängige
Post-Solve-Evaluation verwendet. Ein Job gilt
in einer Replikation als pünktlich, wenn seine simulierte Fertigstellungszeit
seine Due Date nicht überschreitet. Über viele Replikationen entsteht daraus
für jeden Job eine empirische On-Time-Wahrscheinlichkeit. Die Evaluation
vergleicht damit, wie viele beziehungsweise welche Schedules unter
unterschiedlichen Due-Date-, Ausfall- und Reparaturparametern robust
funktionieren.

## GNN-Ersatzmodell

`gurobi_gnn` schätzt für jeden Job die simulierte erwartete
Fertigstellungsverzögerung \(\widehat B_u^{MC}\). Diese schließt die
Right-Shift-Weitergabe über Job- und Maschinenkanten ein. Die Knotenfeatures
sind:

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
Modell dieselbe weiche Service-Level-Struktur wie im Referenzmodell:

\[
C_u+\frac{\widehat B_u^{MC}}{1-\alpha}\le d_u+L_u^{SL},
\qquad L_u^{SL}\ge0.
\]

Das GNN-Modell ist ein MILP, während das Referenzmodell das schwierige MINLP
bleibt.

## Trainingsdaten

Fix-and-Optimize erzeugt Maschinenzuordnungen und unmittelbare
Maschinenfolgen. Im Modus `nonlinear_evaluated` werden die von Gurobi
optimierten Start- und Fertigstellungszeiten übernommen. Der feste Schedule
wird anschließend mit der maschinenbezogenen Single-Failure-Simulation
simuliert. Das Label ist
der Stichprobenmittelwert von \(C_u^{sim}-C_u^{nom}\), einschließlich der
Weitergabe vorgelagerter Verzögerungen. Jeder Schedule-Graph wird in genau
einer CSV-Zeile gespeichert.

Die Trainings-Due-Dates werden zunächst durch einen nominalen Makespanlauf
kalibriert. Mit dem mittleren Work Content \(\bar w\), dem jobspezifischen Work
Content \(w_j\) und \(f^*=C_{\max}^*/\bar w\) gilt

\[
d_j=\left\lceil(1+\delta)f^*w_j\right\rceil,
\qquad \delta\in\{0{,}00,0{,}15,0{,}30,0{,}45\},
\]

Dadurch behalten Jobs mit unterschiedlichem Work Content unterschiedliche Due
Dates, während die Stufen relativ zum nominalen Makespan kalibriert sind.

Ein fehlgeschlagener Instanzlauf beendet die Erzeugung nicht mehr. Bleiben für
eine Instanz nach dem normalen Kandidatenlauf zu wenige lösbare Kandidaten
übrig, wird sie sofort übersprungen. Der aktuelle Fortschritt wird nach jeder
Instanz atomar in
`02_data/gnn_dataset/generation_summary.json` gesichert. Die Datei enthält pro
Split alle erfolgreichen und übersprungenen Instanzen einschließlich
Fehlermeldung. Zusätzlich wird pro Split und insgesamt ein kompakter,
nicht blockierender Qualitätsbericht gespeichert und auf der Konsole
ausgegeben. Er enthält die Graphenzahl, die Verteilungen der Weibull-Faktoren
und Auswahlkategorien sowie Mittelwert und Standardabweichung der Joblabels
und Mittelwert und Median ihrer Monte-Carlo-Standardfehler.
Diese Zeile enthält unter anderem

```text
job_ids=[1,2,3,...]
simulated_expected_completion_delay=[B_1^MC,B_2^MC,B_3^MC,...]
operation_job_indices=[...]
```

Damit kann ein Graph beliebig viele Jobs und ebenso viele unterschiedliche
Joblabels enthalten. Zusätzlich speichert die CSV Standardfehler,
Replikationszahl, Seed und Monte-Carlo-Pünktlichkeitsanteile. In der aktuellen
Pilotkonfiguration werden 256 Replikationen pro Kandidat verwendet; für die
endgültigen Trainingsdaten ist dieser Wert zu erhöhen. Für die
Zuverlässigkeitsvariation werden die ursprünglichen Weibull-Skalen einer
Instanz mit 0,8, 1,0 und 1,2 multipliziert. Die Reihenfolge dieser Faktoren
wird instanzspezifisch und reproduzierbar rotiert, sodass bei vier Poolläufen
jede Instanz alle drei Stufen sieht und die zusätzliche Stufe nicht immer
dieselbe ist.

Die adaptive Auswahl hält weiterhin Kandidaten unmittelbar unter und über der
empirischen Servicegrenze von 0,9. Zusätzlich werden Kandidaten mit niedriger,
mittlerer und hoher simulierter erwarteter Fertigstellungsverzögerung direkt
ausgewählt. Wirtschaftlich gute, Pareto-günstige und strukturell verschiedene
Schedules bleiben als Anker erhalten. Die früheren Kategorien mit lediglich
minimaler beziehungsweise maximaler Pünktlichkeit entfallen, weil sie neben
den beiden Servicegrenzen wenig zusätzliche Information für das eigentliche
Verzögerungslabel lieferten.

Die Daten werden unter `02_data/gnn_dataset` geschrieben. Die aktiven Modelle
liegen in den architekturspezifischen Unterverzeichnissen von
`04_GraphNeuralNetworks/trained_gnn_models`.

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

Der zusätzliche Schalter `solve.create_instances` steuert die getrennten
Benchmark-, Extrapolations- und Stressinstanzen. Bei `true` wird für jeden unter
`solve.evaluation` aktivierten Tier der vollständig konfigurierte Instanzsatz im
zugehörigen Faktorordner neu erzeugt. Bei `false` werden vorhandene Instanzen
nur wiederverwendet, wenn ihr Dateisatz exakt zum konfigurierten Plan passt.

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
