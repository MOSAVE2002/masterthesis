"""
Buffer-Allokations-Optimierung mit PySCIPOpt-ML.

Liest ein zuvor trainiertes Keras-MLP (siehe train_mlp.py) ein und bettet es
als MIP-Constraint in ein SCIP-Modell ein.

Modell:
  Variablen:
    C_i in Z>=0     fuer i = 1..n   (Buffer-Groessen, ganzzahlig, >= 0)
                    C_1 wird auf den Platzhalterwert (typ. 0) fixiert.
    throughput in R   (Output des MLP)

  Eingabevektor des MLP (gleiche Reihenfolge wie beim Training):
    [mu_1, ..., mu_n, C_1, ..., C_n]
    mu_i sind feste Parameter (vorgegeben), C_i sind Entscheidungsvariablen.

  Ziel:    min  sum_i C_i
  s.t.     throughput >= min_throughput
           throughput == MLP([mu, C])

Hinweis zu Bounds:
  Fuer eine numerisch stabile MIP-Formulierung von ReLU-Netzen sind enge
  Schranken auf den Eingabevariablen entscheidend. Da das Netz ausserhalb
  der Trainingsverteilung ohnehin nicht zuverlaessig ist, wird als
  praktische Obergrenze fuer C_i standardmaessig der maximale C-Wert aus
  den Trainingsdaten verwendet (siehe meta-File). Ueber --c_upper kann das
  ueberschrieben werden.
"""