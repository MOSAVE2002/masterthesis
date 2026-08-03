from pathlib import Path

# Das entfernen, nervt übel
ROOT_DIR = Path(__file__).resolve().parents[1]
NEURALNETWORK_DIR = ROOT_DIR / "00_Archiv"
MODEL_DIR = NEURALNETWORK_DIR / "legacy"

MODEL_FILENAME = "simple_makespan_model.pt"
METADATA_FILENAME = "simple_makespan_model_meta.json"

MODEL_PATH = MODEL_DIR / MODEL_FILENAME
METADATA_PATH = MODEL_DIR / METADATA_FILENAME
