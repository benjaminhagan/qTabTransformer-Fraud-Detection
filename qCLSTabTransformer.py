import copy
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

import pennylane as qml

#dev_name = "cuda" if torch.cuda.is_available() else "cpu"
dev_name = "cpu"
PIN_MEMORY = dev_name == "cuda" #Pins memory only for GPU acceleration

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Data
# -------------------------
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X: np.ndarray):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean, std)

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std


def make_dataloaders(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 512,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    seed: int = 42,
):
    
    seed_everything(seed)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    indices = list(range(len(X)))

    # First split: train vs temp
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=test_size + val_size, random_state=seed, stratify=y
    )

    # Second split: val vs test
    val_fraction_of_temp = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=1.0 - val_fraction_of_temp, random_state=seed, stratify=y_temp
    )

    test_df = df.iloc[idx_test].copy()

    scaler = Standardizer.fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    if use_weighted_sampler:
        class_counts = np.bincount(y_train.astype(int))
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY
    )

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "pos_weight": pos_weight,
        "test_df": test_df,
    }


# -------------------------
# Tokenizer
# -------------------------
class NumericalTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# -------------------------
# Quantum circuit
# -------------------------
def build_pqc_vit(n_qubits: int):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weight_0, weight_1):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        indices = n_qubits // 2

        # U1
        for i in range(indices):
            qml.RX(weight_0[0], wires=2 * i)
            qml.RZ(weight_0[1], wires=2 * i + 1)
            qml.CNOT(wires=[2 * i, 2 * i + 1])

        for i in range(1, indices):
            qml.RX(weight_0[0], wires=2 * i - 1)
            qml.RZ(weight_0[1], wires=2 * i)
            qml.CNOT(wires=[2 * i - 1, 2 * i])

        # V1
        for i in range(indices):
            qml.CZ(wires=[2 * i, 2 * i + 1])

        indices = indices // 2

        # U2
        for i in range(indices):
            qml.RX(weight_1[0], wires=2 * (2 * i))
            qml.RY(weight_1[1], wires=2 * (2 * i + 1))
            qml.CNOT(wires=[2 * (2 * i), 2 * (2 * i + 1)])

        for i in range(1, indices):
            qml.RX(weight_1[0], wires=2 * (2 * i - 1))
            qml.RY(weight_1[1], wires=2 * (2 * i))
            qml.CNOT(wires=[2 * (2 * i - 1), 2 * (2 * i)])

        # V2
        for i in range(indices):
            qml.CZ(wires=[2 * (2 * i), 2 * (2 * i + 1)])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumCLSBlock(nn.Module):
    """
    Applies the quantum circuit only to the CLS token embedding.

    """
    def __init__(
        self,
        d_model: int,
        quantum_dim: int = 8,
        n_qubits: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert quantum_dim % n_qubits == 0, "quantum_dim must be divisible by n_qubits"

        self.d_model = d_model
        self.quantum_dim = quantum_dim
        self.n_qubits = n_qubits
        self.n_chunks = quantum_dim // n_qubits

        self.compress = nn.Linear(d_model, quantum_dim)
        self.expand = nn.Linear(quantum_dim, d_model)
        self.pre_act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.weight_0 = nn.Parameter(0.01 * torch.randn(2))
        self.weight_1 = nn.Parameter(0.01 * torch.randn(2))

        self.circuit = build_pqc_vit(n_qubits)

    def _run_quantum_chunk(self, chunk_2d: torch.Tensor) -> torch.Tensor:
        outs = []
        for sample in chunk_2d:
            out = self.circuit(sample, self.weight_0, self.weight_1)

            if isinstance(out, (list, tuple)):
                out = torch.stack(
                    [o if torch.is_tensor(o) else torch.tensor(o, device=sample.device, dtype=sample.dtype)
                     for o in out]
                )

            out = out.to(device=sample.device, dtype=sample.dtype)
            outs.append(out)

        return torch.stack(outs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)  
        x = self.pre_act(x)
        x = self.dropout(x)

        chunks = torch.chunk(x, chunks=self.n_chunks, dim=-1)

        quantum_outs = []
        for chunk in chunks:
            quantum_outs.append(self._run_quantum_chunk(chunk))

        x = torch.cat(quantum_outs, dim=-1) 
        x = self.dropout(x)
        x = self.expand(x)

        return x


# -------------------------
# Model
# -------------------------
class CLSQuantumTabTransformerBinaryClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        quantum_dim: int = 8,
        n_qubits: int = 4,
    ):
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features=n_features, d_token=d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.quantum_cls = QuantumCLSBlock(
            d_model=d_token,
            quantum_dim=quantum_dim,
            n_qubits=n_qubits,
            dropout_rate=dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)     
        cls = self.cls_token.expand(x.size(0), -1, -1)  
        z = torch.cat([cls, tokens], dim=1)    

        z = self.encoder(z)                                 # classical attention blocks
        cls_out = z[:, 0]                                   

        cls_out = self.quantum_cls(cls_out)                 # quantum only 
        logits = self.head(cls_out).squeeze(-1)            
        return logits


# -------------------------
# Metrics
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        loss = self.alpha * ((1 - pt) ** self.gamma) * bce
        return loss.mean()


# -------------------------
# Train/Eval
# -------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    pos_weight,
    device,
    lr: float = 3e-4,
    epochs: int = 20,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
):
    model.to(device)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    criterion = FocalLoss(alpha=1.0, gamma=2.0) #Replaced loss function with Focal Loss to handle the imbalanced data
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_state = None
    best_val_auprc = -1.0

    auprcs = []
    aurocs = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auprc"])

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state = copy.deepcopy(model.state_dict())

            # Choose classification threshold using validation set
            precision, recall, thresholds = precision_recall_curve(
                val_metrics["targets"],
                val_metrics["probs"]
            )

            precision = precision[:-1]
            recall = recall[:-1]

            #Use F1 score to find balanced threshold (good heuristic, but can be changed based on real-world prioritizations of precision/recall)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            best_threshold = thresholds[np.argmax(f1)]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_auprc={val_metrics['auprc']:.5f} | "
            f"val_auroc={val_metrics['auroc']:.5f}"
        )

        auprcs.append(val_metrics['auprc'])
        aurocs.append(val_metrics['auroc'])

    if best_state is not None:
        model.load_state_dict(best_state)
        model.threshold = best_threshold

    return model, auprcs, aurocs

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    auprc = average_precision_score(targets, probs)
    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = float("nan")

    return {"auprc": auprc, "auroc": auroc, "probs": probs, "targets": targets}

@torch.no_grad()
def infer_tabtransformer(model: torch.nn.Module, df: pd.DataFrame, target_col: str = "Class", seed: int = 42):

    model.eval()
    device = torch.device(dev_name)
    seed_everything(seed)

    X = df[model.feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)

    X = model.scaler.transform(X).astype(np.float32)
    X = torch.from_numpy(X)

    loader = DataLoader(
        X,
        batch_size=8192,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    probs_list = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu())

    probs = torch.cat(probs_list).numpy()
    auprc = average_precision_score(targets, probs)
    preds = (probs >= model.threshold).astype(int)

    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = float("nan")

    return  {"auprc": auprc, "auroc": auroc, "probs": probs, "preds": preds, "targets": targets}

def run_experiment(df: pd.DataFrame, lr=3e-4):
    device = torch.device(dev_name)

    data = make_dataloaders(
        df,
        target_col="Class",
        test_size=0.2,
        val_size=0.1,
        batch_size=512,
        use_weighted_sampler=True,
        seed=42,
    )

    # model = CLSQuantumTabTransformerBinaryClassifier(
    #     n_features=len(data["feature_cols"]),
    #     d_token=64,
    #     n_heads=8,
    #     n_layers=3,
    #     d_ff=256,
    #     dropout=0.1,
    #     quantum_dim=8,
    #     n_qubits=4,
    # )

    # model = CLSQuantumTabTransformerBinaryClassifier(
    #     n_features=len(data["feature_cols"]),
    #     d_token=32,
    #     n_heads=4,
    #     n_layers=2,
    #     d_ff=128,
    #     dropout=0.1,
    #     quantum_dim=4,
    #     n_qubits=4,
    # )

    model = CLSQuantumTabTransformerBinaryClassifier(
        n_features=len(data["feature_cols"]),
        d_token=32,
        n_heads=4,
        n_layers=2,
        d_ff=16,
        dropout=0.1,
        quantum_dim=4,
        n_qubits=4,
    )

    model.scaler = data["scaler"]
    model.feature_cols = data["feature_cols"]
    
    model, auprcs, aurocs = train_model(
        model,
        data["train_loader"],
        data["val_loader"],
        data["pos_weight"],
        device=device,
        lr=lr,
        epochs=20,
    )

    test_metrics = evaluate(model, data["test_loader"], device)
    # print(f"Test AUPRC: {test_metrics['auprc']}")
    # print(f"Test AUROC: {test_metrics['auroc']}")

    return model, test_metrics, data['test_df']
