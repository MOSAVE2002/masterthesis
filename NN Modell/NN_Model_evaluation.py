from pathlib import Path
import torch

model_path = Path("NN Modell/i5_k5_1.pt")
bundle = torch.load(model_path, map_location="cpu", weights_only=False)

print(bundle["instance_name"])
print("feature Columns")
print(bundle["feature_columns"])
print("mean")
print(bundle["mean"])
print("std")
print(bundle["std"])