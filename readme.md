# Szenariofreies stochastisches FJSP mit Weibull-MINLP und GNN

Die aktive Pipeline verwendet wieder die nichtlineare Ausfallwahrscheinlichkeit
am Mittelpunkt jeder Operation. Für Maschine (k) gelten eine
Weibull-Ausfallzeit und eine exponentielle Reparaturzeit. Der Codeparameter
`weibull_alpha` bezeichnet die Weibull-Skala (im Manuskript `eta`), keinen
Servicegrad:

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

## Wirtschaftliche Zielfunktion und gepufferte Terminverletzung

Die produktiven Modelle minimieren gemeinsam

\[
\sum_{i,k} c_k p_{ik}Y_{ik}
+c_{Halle}\,C_{\max}
+c_D\sum_{u\in\mathcal J}L_u.
\]

Der erste Term erfasst die Maschinenkosten während der Bearbeitung. Der zweite
Term bildet die Hallenbetriebskosten bis zum Makespan ab. Der dritte Term
bewertet die gepufferte Terminverletzung. Die
Kostenraten werden über `objective.facility_cost_per_time` und
`objective.tardiness_cost_per_time` konfiguriert. Beide stehen aktuell
auf eins. Dieselbe Kostenstruktur wird im Grundmodell, im nichtlinearen Modell
und im GNN-Modell verwendet.

## Maschinenprofile

Die Instanzgenerierung verwendet zwei bewusst unterschiedliche
Maschinenarten:

- `old`: niedrige Kosten, geringere Geschwindigkeit, Weibull-Formparameter
  \(\beta=3\) und schwächere Zuverlässigkeits- und Reparaturparameter,
- `new`: höhere Kosten, höhere Geschwindigkeit, größere Weibull-Skala,
  Weibull-Formparameter \(\beta=2\) und schnellere Reparatur.

Kosten, Geschwindigkeit, Weibull-Skala, Weibull-Form und Reparaturrate sind in
den Basisprofilen konstant. Für 20 % der Trainingsinstanzen je Größenklasse
wird reproduzierbar ein Jitter von ±5 % auf Weibull-Skala und Reparaturrate
angewendet; Kostenrate, Geschwindigkeit und Weibull-Form bleiben unverändert.
Validierungs- und Testinstanzen verwenden ausschließlich die festen
Basisprofile. Jede Operation kann auf Maschinen aus mindestens zwei
Profilklassen ausgeführt werden. Ihre Bearbeitungszeit entsteht aus einer
ganzzahligen Basiszeit im konfigurierten Bereich `[10, 30]`, dem
Geschwindigkeitsfaktor des Profils und einem kleinen Operationsrauschen. Die
Profile werden unter `instances.generation.machine_profiles` konfiguriert und
bleiben bei der analytischen Erzeugung der Trainingslabels unverändert.
Alle Modellzeiten werden ausschließlich in abstrakten Zeiteinheiten (ZE)
angegeben; `instances.generation.time_unit` ist `"ZE"`. Bearbeitungszeiten,
Start- und Fertigstellungszeiten, Due Dates, Puffer, Weibull-Skalen und
Reparaturdauern haben die Einheit ZE. Reparaturraten haben die Einheit 1/ZE,
Kostenraten GE/ZE (GE = Geldeinheiten); Weibull-Formparameter und
Geschwindigkeitsfaktoren sind dimensionslos. Der MAE der Verzögerungsprognose
wird in ZE gemessen, der MSE in ZE².

Die Weibull-Skalen betragen 120 ZE (`old`) und 200 ZE (`new`), die mittleren
exponentiellen Reparaturzeiten 60 ZE für `old` und 30 ZE für `new`. Veraltete Einheitenannotationen werden beim Laden als ZE übernommen.
Für das neue lokale Labelziel müssen Daten und Gewichte neu erzeugt werden.
Solver-Zeitlimits und gemessene Rechenzeiten bleiben reale Sekunden (`s`).

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

gebildet. Die gepufferte Terminbedingung lautet gemäß Manuskript

\[
C_u+B_u^{NL}\le d_u+L_u,\qquad L_u\ge0.
\]

Es gibt keinen Servicegrad und keinen Faktor `1/(1-alpha)` im Modell.
Die Zielfunktion bewertet die Terminverletzung mit `c_D`; im nominalen
Grundmodell gilt `B_u=0`. Eine Wahrscheinlichkeit für Termintreue wird hier
nicht garantiert. Das Referenzmodell bleibt wegen der Integrale ein MINLP.

Die aktive lokale Pufferdefinition verwendet weiterhin einen ersten Weibull-Ausfall
und eine exponentielle Reparatur pro Maschine. Für eine Operation mit nominalem
Mittelpunkt \(t_i\) zählt nur die dort verbleibende Reparaturzeit:

\[
Z_i=\mathbf 1(F_k\le t_i)\max(F_k+R_k-t_i,0),\qquad
B_u=\mathbb E\!\left[\sum_{i\in\mathcal O_u} Z_i\right]
=\sum_{i\in\mathcal O_u} Pd_{ik}(t_i)/\lambda_k.
\]

Es gibt dabei keine Störungsfortpflanzung und keine Verrechnung mit späteren
Leerlaufzeiten. Trainingslabels sind deterministische Erwartungswerte. Die
Post-Solve-Auswertung in `05_Simulation/preempt_resume.py` simuliert die
Ausführung: ein erster Ausfall und eine Reparatur je Maschine, Warten oder
Unterbrechen/Fortsetzen und Weitergabe der Verzögerung über feste Job- und
Maschinenreihenfolgen. `evaluation.simulation.model` ist `preempt_resume`.
`evaluation.service_level_threshold=0.9` dient ausschließlich zur Bewertung
empirischer Termintreue. Die lokale Simulation `local_midpoint.py` bleibt für
numerische Prüfungen der Trainingslabels verfügbar; sie bewertet keine
vollständige Ausführung. Die Trainingslabels bleiben unverändert lokale Puffer.

## GNN-Ersatzmodell

`gurobi_gnn` schätzt für jeden Job den lokalen erwarteten Reparaturpuffer
\(\widehat B_u\). Die vier Knotenfeatures sind:

1. nominaler Operationsmittelpunkt relativ zur Weibull-Skala \(t_i/\alpha_k\),
2. skalierte Reparaturrate \(\lambda_k\alpha_k/10\),
3. skalierter Weibull-Formparameter \(\beta_k/5\),
4. mittlere Reparaturdauer relativ zu 60 ZE: \(1/(60\lambda_k)\).

Hier ist \(k\) die zugewiesene Maschine und \(t_i=(S_i+C_i)/2\).
Alle vier Features sind dimensionslos; 60 ZE ist eine feste Zeitreferenz.
Liefertermine gehen nicht in die Features ein. Die bekannten Maschinenparameter
und der nominale Mittelpunkt reichen zur Beschreibung des lokalen Erwartungswerts.
Labels und Vorhersagen bleiben in ZE; der MAE wird nicht umskaliert.

Im MILP wird das erste Feature als \(\sum_k (T_iY_{ik})/\alpha_k\)
berechnet. Die Produkte \(T_iY_{ik}\) sind bereits exakt linearisiert. Die
übrigen Features sind affine Summen über \(Y_{ik}\). Die Featureausdrücke
benötigen keine zusätzlichen Variablen; JOB behält ausschließlich Jobkanten.
Die Eingangsgrenzen berücksichtigen die zulässigen Bearbeitungszeiten und den
seriellen Planungshorizont. Das neue Graphschema ist
`direct_machine_and_job_predecessor_physical_features_v10`.

Alte Datensätze und Gewichte mit fünf oder sieben Eingaben passen nicht zu
diesem Eingang. Das Skript `migrate_local_buffer_features.py` konvertiert die
bisherigen sieben Features in einen neuen Datensatz, prüft die rekonstruierte
Pufferformel und erhält Pläne, Kanten und Labels. Alte Dateien werden dabei
nicht überschrieben. Anschließend müssen Vier-Input-Netze trainiert werden.

Es werden drei bewusst getrennte Ersatzmodelle trainiert und verglichen:

1. `linear`: knotenseitiges ReLU-MLP ohne Message Passing,
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
Modell dieselbe unskalierte gepufferte Terminbedingung wie im Referenzmodell:

\[
C_u+\widehat B_u\le d_u+L_u,
\qquad L_u\ge0.
\]

Das GNN-Modell ist ein MILP, während das Referenzmodell das schwierige MINLP
bleibt.

## Trainingsdaten

Fix-and-Optimize erzeugt Maschinenzuordnungen, unmittelbare Maschinenfolgen und
nominale Startzeiten. Zulässige Lösungen vom Zeitlimit sind erlaubt. Vor dem
Labeln werden Jobreihenfolge und Maschinenüberschneidungen geprüft.

`helper/local_buffer.py` berechnet die Labels mit 128 Quadraturpunkten und prüft
sie unabhängig mit 256 Punkten. Überschreitet die Summe der absoluten Differenzen
pro Job 1e-6 ZE, wird der Kandidat nicht als gültiges Label übernommen. Das ist
ein numerischer Konvergenzcheck, keine mathematisch zertifizierte Fehlerschranke.
Identische lokale Zustände bekommen damit dieselben Erwartungswerte ohne
Monte-Carlo-Streuung. Die Zielspalte heißt
`expected_local_midpoint_repair_buffer`; die CSV enthält zusätzlich
`local_buffer_numerical_errors`. Markov-Untergrenzen werden nur als Diagnose
unter `local_buffer_service_lower_bounds` gespeichert.

Pro Instanz werden vier Poolläufe mit bis zu 20 Kandidaten und je einer Sekunde
Optimierungszeit ausgeführt. Die Due-Date-Kalibrierung behält die relativen
Offsets `[0.0, 0.15, 0.3, 0.45]`. Die Weibull-Faktoren `[0.8, 1.0, 1.2]` werden
reproduzierbar rotiert. Die Auswahl `buffer_structure_cost` erzeugt 12 Graphen:
vier Kosten/Puffer-Kombinationen, sechs zur Abdeckung niedriger, mittlerer und
hoher Jobpuffer sowie zwei für strukturelle Vielfalt. Fehlen Kosten/Puffer-
Kombinationen im Pool, werden die Plätze durch Pufferabdeckung aufgefüllt.
Die Auswahlkosten sind nominale Maschinenkosten plus Makespan plus nominale
Verspätung (Gewichte eins). Teure zulässige Pläne bleiben ausdrücklich zugelassen.
Servicegrad-Kategorien steuern diese Auswahl nicht mehr.

Die Duplikaterkennung berücksichtigt die vier Features (auf sieben
Nachkommastellen), Jobzugehörigkeiten und gerichtete Kanten einschließlich
Mehrfachkanten. Gleiche Maschinenfolgen mit anderen Startzeiten bleiben erhalten.
Trainings-, Validierungs- und Testinstanzen werden getrennt aufgeteilt.

Die Konfiguration erzeugt 579 Instanzen je Größe, für 3–5 Jobs und 3–5
Maschinen. Der 80/10/10-Split enthält je Größenklasse 463 Trainingsinstanzen
sowie 58 Validierungs- und 58 Testinstanzen. Bei 12 Graphen pro Instanz sind
das 50.004 Trainingsgraphen sowie jeweils 6.264 Validierungs- und Testgraphen,
sofern keine Instanz wegen zu weniger gültiger Kandidaten übersprungen wird.
93 der 463 Trainingsinstanzen je Größe, insgesamt 837 Instanzen beziehungsweise
10.044 Graphen, stammen aus dem Jitterarm. Fortschritt, ausgelassene Instanzen,
Jitterzuordnung, Labelverteilungen und Quadraturdifferenzen stehen in
`02_data/gnn_dataset/generation_summary.json`.

Neue Daten liegen unter `02_data/gnn_dataset`, neue Modelle
unter `04_GraphNeuralNetworks/trained_gnn_models`.
Das Training vergleicht LINEAR, JOB und GraphSAGE mit zwei Schichten und je
vier oder acht Hidden Nodes. Aktiv bleiben Adam, MSE und ReLU. Die Batchgröße
beträgt 256. Mit `cache_batches: true` werden einmal gemischte Graphbatches
wiederverwendet und pro Epoche in neuer Reihenfolge verarbeitet; das spart die
wiederholte Graphzusammenstellung. Diese Option unterscheidet sich von einer
neuen Mischung einzelner Graphen je Epoche. `false` verwendet weiterhin den
normalen PyG-DataLoader.

`train_from_file(..., test_csv_path=None)` erlaubt Modellwahl ausschließlich
auf der Validierung. Die Modellmetadaten enthalten dann `test_evaluated: false`.
Die finale numerische Analyse wird als Teil von `--workflow evaluate`
ausgeführt und schreibt ihre Tabellen, ECDF- und Performance-Profile sowie
Gap-over-Time- und Incumbent-over-Time-Vergleiche unter
`06_Evaluation/results/numerical_analysis`.

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
.venv/bin/python main.py
```

Die aktuelle Konfiguration erzeugt die Instanzen und Trainingsdaten neu,
trainiert die konfigurierten Vier-Feature-Modelle und startet anschließend die
Solver. Die abschließende Evaluation ist ausgeschaltet.
Einzelne Phasen lassen sich ausdrücklich starten:

```bash
.venv/bin/python main.py --workflow create-instances generate-training-data train-gnn
.venv/bin/python main.py --workflow train-gnn
.venv/bin/python main.py --workflow solve
.venv/bin/python main.py --workflow evaluate
```

`config.json` steuert die fünf Phasen `create_instances`,
`generate_training_data`, `train_gnn`, `solve` und `evaluate`. Die optionale
Auswertung simuliert die gespeicherten Solver-Schedules unabhängig nach und
schreibt die Vergleichstabellen nach `06_Evaluation/results`.
Die Lösungsdateien liegen unter `02_data/fjsp_solutions`;
derselbe konfigurierte Pfad wird beim Schreiben und Lesen verwendet. Bei
`solve evaluate` werden ausschließlich die Dateien des aktuellen Laufs verglichen.
Ein alleiniger `evaluate`-Aufruf liest den konfigurierten Lösungsordner.
Kommandozeilenphasen überschreiben die Schalter aus der Konfiguration auch für
die nachgelagerte Auswertung.

Jede neu erzeugte Lösungsdatei enthält neben Status, Incumbent-Anzahl,
Zielfunktionswert, finaler Schranke, Gap und Laufzeit auch die Zahl der
Branch-and-Bound-Knoten, die stärkste während der Bearbeitung des Wurzelknotens
beobachtete Schranke, Zeitpunkt und Zielfunktionswert des ersten sowie des
besten Incumbents. Zusätzlich entsteht neben jeder neuen Lösungsdatei eine
`*_solver_progress.csv` mit dem zeitlichen Verlauf von Incumbent, Best Bound,
relativem Gap, Knotenzahl und Lösungszahl. Änderungen dieser zentralen Größen,
der erste weitere MIP-/MIPNODE-Callback nach jeweils mindestens 0,25 Sekunden
ohne Änderung sowie der Endzustand werden gespeichert. Modellgrößen werden als
kontinuierliche, binäre und
ganzzahlige Variablen, lineare Matrix-Nonzeros sowie lineare, quadratische,
allgemeine und nichtlineare Constraints gespeichert. Die Wurzelknotenschranke
ist damit eine Callback-basierte Root-Bound einschließlich der am Wurzelknoten
wirksamen Schnitte, nicht der Wert einer separat gelösten, schnittfreien
LP-Relaxation. `evaluate_solutions.py` übernimmt diese Werte in die
Schedule-CSV und ergänzt die Indikatoren Incumbent vorhanden, optimal oder
Gap höchstens 1 %, Zeitlimit mit Incumbent und keine zulässige Lösung gefunden.

`solve.nominal_warm_start` steuert die gemeinsame nominale Startlösung und ist
standardmäßig `false`. Bei `false` erhalten die Modelle keine übergebene
Startlösung und laufen in der Reihenfolge von `solve.solvers`.
Bei `true` muss `gurobi` in `solve.solvers` stehen: Es wird je Instanz zuerst
gelöst, und alle GNNs sowie das nichtlineare Modell erhalten dessen gleiche
Startlösung als veränderbaren Gurobi-Startvorschlag. Liefert das nominale Modell
keinen Incumbent, laufen die anderen Modelle ohne diesen Vorschlag weiter.
Alle Solver laufen nacheinander im selben Python-Prozess wie `main.py`.
Es werden keine zusätzlichen Solverprozesse gestartet. `solve_manifest.json`
enthält die bereits abgeschlossenen Läufe. Ein nativer Solverabsturz kann den
gesamten Lauf beenden. Die normale Lösezeitgrenze bleibt 60 Sekunden.

Die Auswertung berechnet die lokale deterministische Pufferformel für alle
zulässigen Pläne erneut, einschließlich der nominalen Referenz. `result_table.csv`
enthält `reference_total_cost`, `reference_due_date_violation`,
`maximum_repair_buffer_underestimation` und `jobs_underestimated_by_more_than_one`.
Die Referenzkosten verwenden die gespeicherten Kostenkoeffizienten und
`max(0, C_job + B_reference - due_date)`, auch für die nominalen Pläne. Ein fehlender Incumbent erhält keine erfundenen
Kosten oder Pufferfehler. `job_comparison.csv` enthält die einzelnen Jobfehler.

Mit `--config PFAD` ist ein separater Versuch ohne Änderung der Hauptkonfiguration
möglich. Relative Datenpfade beziehen sich weiterhin auf das Repository. Der
Integrationstest und seine Konfigurationen liegen unter
`06_Evaluation/results/pipeline_integration_20260910`.

Die Instanzgrößen für
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
