"""
Trainiert ein einfaches Sequential-MLP (Dense + ReLU) zur Vorhersage des
Throughputs eines Makespan mit n Jobs m Maschinen

#TODO Welche Eingabe nehme ich? Maschinenbelegung und Auslastung Maschine oder was anderes?
Eingabe pro Sample (flach): ?
Ausgabe: Makespan


Das trainierte Modell wird als .keras-Datei gespeichert, sodass es vom
Optimierungs-Skript (optimize_buffers.py) eingelesen und via pyscipopt-ml
in ein MIP eingebettet werden kann.
"""
import os
import pandas as pd


def load_dataset():
    df = pd.read_csv(csv_path, sep=";")

def build_mlp():
    pass

def train_and_save():
    pass


if __name__ == '__main__':
    pass