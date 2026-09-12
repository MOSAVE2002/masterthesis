# Ausführungssimulation und Pufferprüfung

Die aktive Post-Solve-Auswertung verwendet `preempt_resume.py`.
`evaluation.simulation.model` muss `preempt_resume` sein.

Pro Wiederholung wird je Maschine ein erster Weibull-Ausfallzeitpunkt und eine
exponentielle Reparaturdauer gezogen. Alle Operationen derselben Maschine
teilen dieses Ausfallereignis. Eine Operation wartet, wenn sie während einer
Reparatur beginnen würde; bei einem Ausfall während der Bearbeitung wird sie
unterbrochen und nach Reparatur fortgesetzt. Verzögerungen werden über die
festgelegten Job- und direkten Maschinenvorgänger weitergegeben. Geplante
Startzeiten bleiben Untergrenzen, Maschinenzuordnungen und Reihenfolgen fest.
Weitere Ausfälle nach der ersten Reparatur werden nicht gezogen.

Termintreue ist der Anteil `simulierte Fertigstellung <= Liefertermin`.
`evaluation.service_level_threshold` (aktuell 0.9) wird ausschließlich zum
Bewerten dieser Ergebnisse verwendet. Er verändert weder Optimierung noch
Referenzkosten. Kandidaten derselben Instanz nutzen dieselben Zufallsströme.

Die GNN-Trainingslabels bleiben die deterministische Summe lokaler erwarteter
Restreparaturen am nominalen Operationsmittelpunkt aus `helper/local_buffer.py`.
Sie approximieren den Puffer des nichtlinearen Modells und sind nicht die
mittlere fortgepflanzte Fertigstellungsverzögerung aus dieser Simulation.
`local_midpoint.py` dient ausschließlich zur numerischen Prüfung dieser Labels.
