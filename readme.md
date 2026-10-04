# FJSP Simulation and GNN Pipeline

Dieses Repository enthält eine vollständig über `config.json` gesteuerte
Python-Pipeline zur Erzeugung, Verarbeitung und Lösung von Flexible-Job-Shop-
Instanzen. Die README beschreibt ausschließlich den Aufbau und die Bedienung
des Codes. Die mathematische Herleitung der Modelle ist Bestandteil der
Masterarbeit und wird hier nicht wiederholt.

## Pipeline

`main.py` ist der zentrale Einstiegspunkt. Die aktivierten Phasen werden in
dieser Reihenfolge ausgeführt:

| Phase | Aufgabe | Zentrales Modul |
|---|---|---|
| `create_instances` | Instanzen erzeugen und in Datensplits ablegen | `01_generator/instance_generator.py` |
| `generate_training_data` | Optimierungslösungen in gelabelte Graphdaten umwandeln | `04_GraphNeuralNetworks/models/generate_fix_and_optimize_training_data.py` |
| `train_gnn` | Konfigurierte GNN-Architekturen trainieren | `04_GraphNeuralNetworks/models/model_training_FJSP_GNN.py` |
| `solve` | Benchmark- und Extrapolationsinstanzen lösen | `helper/experiment_plan.py`, `helper/solve_runner.py` |
| `simulate` | Gespeicherte Produktionspläne nachträglich simulieren | `05_Simulation/simulation.py` |

Die Phasen sind einzeln schaltbar. Dadurch können vorhandene Instanzen,
Datensätze und Modelle wiederverwendet werden.

## Projektstruktur

```text
FJSP_Simulation/
├── main.py
├── config.json
├── requirements.txt
├── pyproject.toml
├── 01_generator/
│   └── instance_generator.py
├── 02_data/
│   ├── fjsp_instances/
│   ├── gnn_dataset/
│   └── fjsp_solutions/
├── 03_Gurobi/
│   ├── build_fjsp.py
│   ├── build_fjsp_with_nonlinear.py
│   └── build_fjsp_with_gnn.py
├── 04_GraphNeuralNetworks/
│   ├── models/
│   │   ├── generate_fix_and_optimize_training_data.py
│   │   ├── gnn_architecture.py
│   │   └── model_training_FJSP_GNN.py
│   └── trained_gnn_models/
├── 05_Simulation/
│   └── simulation.py
├── helper/
└── cluster_files/
```

### Zentrale Dateien

- `main.py` liest die Konfiguration und koordiniert alle Pipelinephasen.
- `config.json` enthält sämtliche veränderbaren Laufparameter und Pfade.
- `01_generator/instance_generator.py` erzeugt, speichert und lädt
  FJSP-Instanzen.
- `03_Gurobi/build_fjsp.py` erstellt das gemeinsame FJSP-Grundmodell.
- `03_Gurobi/build_fjsp_with_nonlinear.py` erweitert das Grundmodell um die
  nichtlineare Zuverlässigkeitskomponente.
- `03_Gurobi/build_fjsp_with_gnn.py` lädt ein trainiertes GNN und bettet es in
  das Gurobi-Modell ein.
- `04_GraphNeuralNetworks/models/generate_fix_and_optimize_training_data.py`
  erzeugt Graph-CSV-Dateien aus Optimierungslösungen.
- `04_GraphNeuralNetworks/models/model_training_FJSP_GNN.py` lädt die
  Graphdaten und trainiert die konfigurierten Modelle.
- `05_Simulation/simulation.py` liest gespeicherte Lösungen und führt die
  Monte-Carlo-Nachsimulation durch.

### Helper-Module

- `buffer_candidate_selection.py`: entfernt doppelte Kandidaten und wählt die
  zu speichernden Graphen aus.
- `due_date_calibration.py`: erzeugt die für Trainings- und Solve-Instanzen
  benötigten kalibrierten Fälligkeitstermine.
- `economic_objective.py`: ergänzt die gemeinsame Zielfunktion und
  Verspätungsvariablen.
- `experiment_plan.py`: erzeugt oder validiert Benchmark- und
  Extrapolationspläne.
- `gnn_model_registry.py`: findet trainierte Modelle und prüft deren Metadaten.
- `gurobi_solution_writer.py`: schreibt Solverergebnisse in ein einheitliches
  Textformat.
- `local_buffer.py`: stellt den gemeinsamen Vertrag für GNN-Zielwerte bereit.
- `sequence_setup.py`: enthält die Graphstruktur und gemeinsam verwendete
  GNN-Features.
- `solution_plots.py`: erzeugt Gantt-, Lösungsgraph- und Kandidatengrafiken.
- `solve_runner.py`: führt alle geplanten Solver- und Modellkombinationen aus.
- `start_solve_ins.py`: baut und optimiert eine einzelne Solverinstanz.
- `stochastic_fjsp.py`: validiert Maschinenprofile und stellt stochastische
  Maschinenparameter bereit.
- `training_data_quality.py`: schreibt den Fortschritt der Datengenerierung in
  `generation_summary.json`.


## Konfiguration

Die Pipeline besitzt keine Kommandozeilenoptionen. Alle Einstellungen werden
aus `config.json` gelesen. Relative Pfade werden relativ zum Repository
aufgelöst.

### Workflow auswählen

```json
"workflow": {
  "create_instances": false,
  "generate_training_data": false,
  "train_gnn": false,
  "solve": false,
  "simulate": false
}
```

In der eingecheckten Konfiguration sind alle Phasen deaktiviert. Dadurch führt
ein unbeabsichtigter Start zu keinen Änderungen an Daten oder Modellen.

### Instanzerzeugung

Der Block `instances.generation` konfiguriert:

- Job- und Maschinenzahlen,
- die Anzahl der Operationen je Job,
- die Anzahl der Instanzen je Größenkombination,
- Zufallsseed und Bearbeitungszeitbereich,
- Maschinenprofile und Parameterstreuung,
- Due-Date-Faktoren,
- die Aufteilung in Training, Validierung und Test.

`workflow.create_instances: true` erzeugt die konfigurierten Pickle-Dateien
neu. Dabei werden bestehende generierte Dateien in den Split-Verzeichnissen
ersetzt.

### Trainingsdatengenerierung

Der Block `training.data_generation` bestimmt unter anderem:

- das Ausgabeverzeichnis,
- die Zahl der Graphen je Instanz,
- die verwendeten Zufallsseeds,
- die Variation der Zuverlässigkeitsparameter,
- die Einstellungen der temporären Optimierungsläufe.

Die Phase schreibt je Split eine CSV-Datei und zusätzlich eine
`generation_summary.json`. Die Summary enthält den Laufstatus, die verwendeten
Metadaten sowie die Anzahl erfolgreicher und übersprungener Instanzen.

### GNN-Training

Der Block `training.gnn` enthält:

- das Modellverzeichnis,
- die zu trainierenden Architekturkombinationen,
- Epochenzahl, Batchgröße und Lernrate,
- Validierungsintervall und Early-Stopping-Patience,
- den Zufallsseed.

Für jede konfigurierte Architektur entstehen eine PyTorch-Datei (`.pt`) und
eine zugehörige Metadatendatei (`_meta.json`). Der Code unterstützt die
Architekturtypen `linear`, `job` und `sage`.

### Solve-Plan

Der Block `solve` legt fest:

- welche Solver ausgeführt werden,
- wo Lösungsdateien und Manifest gespeichert werden,
- welche Benchmark- oder Extrapolationstiers aktiv sind,
- welche Job-/Maschinenkombinationen zu einem Tier gehören,
- wie viele Instanzen je Kombination verwendet werden.

Mit `solve.create_instances: true` werden die Solve-Instanzen der aktiven Tiers
vor dem Lösen neu erzeugt. Bei `false` validiert der Code den vorhandenen
Dateisatz, bevor er ihn wiederverwendet.

Unterstützte Solverbezeichnungen:

- `gurobi`: gemeinsames FJSP-Grundmodell,
- `gurobi_nonlinear`: nichtlineare Modellerweiterung,
- `gurobi_gnn`: GNN-gestützte Modellerweiterung.

Der Block `solvers.gurobi.common` enthält gemeinsame Gurobi-Parameter. Die
Unterblöcke `nonlinear` und `gnn` enthalten solverabhängige Einstellungen.
Der Block `objective` stellt die Kostenparameter bereit, die `main.py` an die
Solvermodelle weitergibt.

### Simulation

Der Block `simulation` definiert:

- das Verzeichnis der einzulesenden Lösungen,
- das Verzeichnis der zugehörigen Instanzen,
- die Ergebnis-CSV,
- Anzahl und Seed der Replikationen.

Die Simulation sucht rekursiv nach Lösungsdateien und schreibt alle Ergebnisse
in eine gemeinsame CSV-Datei.

## Ausführung

Nach Aktivierung der gewünschten Workflowphasen wird die Pipeline gestartet:

```bash
.venv/bin/python main.py
```

Die Phasen laufen nacheinander im selben Python-Prozess. `main.py` startet
keine separaten Solverprozesse.

## Erzeugte Dateien

| Verzeichnis | Inhalt |
|---|---|
| `02_data/fjsp_instances/training` | Trainingsinstanzen |
| `02_data/fjsp_instances/valid` | Validierungsinstanzen |
| `02_data/fjsp_instances/test` | Testinstanzen |
| `02_data/fjsp_instances/benchmark` | Benchmarkinstanzen nach Tierkonfiguration |
| `02_data/fjsp_instances/extrapolation` | Extrapolationsinstanzen nach Tierkonfiguration |
| `02_data/gnn_dataset` | Graph-CSV-Dateien und `generation_summary.json` |
| `04_GraphNeuralNetworks/trained_gnn_models` | Modellgewichte und Metadaten |
| `02_data/fjsp_solutions` | Lösungsdateien und Solve-Manifeste |
| `05_Simulation/results` | Ergebnisse der Nachsimulation |

Die Lösungsdateien enthalten Solverstatus, Laufzeit, Zielfunktionswert,
Optimalitätsschranke, Gap, Knotenzahl, Modellgröße und die gespeicherte
Produktionsplanung.

