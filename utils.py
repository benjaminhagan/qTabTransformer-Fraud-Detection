import numpy as np
import pandas as pd
import torch
from qCLSTabTransformer import CLSQuantumTabTransformerBinaryClassifier
from TabTransformer import TabTransformerBinaryClassifier
from dataclasses import dataclass

AMEX_CATEGORICAL_COLS = [
    "B_30", "B_38",
    "D_114", "D_116", "D_117", "D_120", "D_126",
    "D_63", "D_64", "D_66", "D_68"
]


def preprocess_amex(train_data: pd.DataFrame, train_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Convert AMEX's multi-row-per-customer data into a purely numerical
    one-row-per-customer DataFrame.

    Baseline preprocessing:
      - Keep the latest statement for each customer
      - Drop customer_ID and statement date
      - Zero-impute numerical features
      - One-hot encode categorical features
      - Add target as 'Class'
    """

    df = train_data.copy()
    labels = train_labels.copy()

    # ---------------------------------------------------------
    # 1. Select the latest statement for each customer
    # ---------------------------------------------------------
    df["S_2"] = pd.to_datetime(df["S_2"])

    df = (
        df.sort_values(["customer_ID", "S_2"])
          .groupby("customer_ID", as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 2. Attach customer-level target
    # ---------------------------------------------------------
    labels = labels.rename(columns={"target": "Class"})

    df = df.merge(
        labels[["customer_ID", "Class"]],
        on="customer_ID",
        how="inner"
    )

    # ---------------------------------------------------------
    # 3. Drop ID and date
    # ---------------------------------------------------------
    df = df.drop(columns=["customer_ID", "S_2"])

    # ---------------------------------------------------------
    # 4. Separate categorical and numerical columns
    # ---------------------------------------------------------
    categorical_cols = [
        col for col in AMEX_CATEGORICAL_COLS
        if col in df.columns
    ]

    numerical_cols = [
        col for col in df.columns
        if col not in categorical_cols + ["Class"]
    ]

    # ---------------------------------------------------------
    # 5. Zero-impute numerical features
    # ---------------------------------------------------------
    df[numerical_cols] = df[numerical_cols].fillna(0)

    # ---------------------------------------------------------
    # 6. Give missing categorical values their own category
    # ---------------------------------------------------------
    df[categorical_cols] = df[categorical_cols].fillna("__MISSING__")

    # ---------------------------------------------------------
    # 7. One-hot encode categorical features
    # ---------------------------------------------------------
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # 8. Make everything numerical
    # ---------------------------------------------------------
    feature_cols = [
        col for col in df.columns
        if col != "Class"
    ]

    df[feature_cols] = df[feature_cols].astype(np.float32)
    df["Class"] = df["Class"].astype(np.float32)

    return df

@dataclass(eq=False)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std
    


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