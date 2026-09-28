import torch
from models.qCLSTabTransformer import CLSQuantumTabTransformerBinaryClassifier
from models.TabTransformer import TabTransformerBinaryClassifier
from utils.Preprocessor import Standardizer

def checkpoint_from_model(model: CLSQuantumTabTransformerBinaryClassifier) -> dict:
    checkpoint = {
        "model_config": model.config,
        "model_state_dict": model.state_dict(),
        "threshold": float(model.threshold),
        "feature_cols": list(model.feature_cols),
        "scaler_mean": torch.as_tensor(model.scaler.mean, dtype=torch.float32),
        "scaler_std": torch.as_tensor(model.scaler.std, dtype=torch.float32),
    }
    return checkpoint


def model_from_checkpoint(checkpoint: dict) -> CLSQuantumTabTransformerBinaryClassifier:
    is_quantum = "qufex_params" in checkpoint["model_config"]

    model = CLSQuantumTabTransformerBinaryClassifier(**checkpoint["model_config"]) if is_quantum else TabTransformerBinaryClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.threshold = checkpoint["threshold"]
    model.feature_cols = checkpoint["feature_cols"]
    model.scaler = Standardizer(mean=checkpoint["scaler_mean"].numpy(), std=checkpoint["scaler_std"].numpy())

    if not is_quantum:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model.eval()

    return model