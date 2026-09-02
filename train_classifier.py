"""
train_classifier.py
-----------------------
Trains the FormLSTM classifier on synthetic biomechanical training data
(see synthetic_data.py for why synthetic data is used) and saves a
checkpoint containing both the model weights and the fitted feature
normalizer.

Usage:
    python train_classifier.py --epochs 60 --n-per-class 300
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from synthetic_data import generate_dataset, CLASSES
from form_classifier import FormLSTM, FeatureNormalizer


def train(epochs: int, n_per_class: int, batch_size: int, lr: float, output_path: str, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = generate_dataset(n_per_class=n_per_class, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FormLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        if epoch % 10 == 0 or epoch == epochs:
            train_acc = correct / total
            print(f"Epoch {epoch:3d}/{epochs}  loss={total_loss/total:.4f}  train_acc={train_acc:.3f}")

    # Evaluation on held-out synthetic test set
    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
        logits = model(x_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    print("\n=== Held-out test set evaluation ===")
    print(classification_report(y_test, preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(CLASSES)
    print(confusion_matrix(y_test, preds))

    torch.save({
        "model_state_dict": model.state_dict(),
        "normalizer_state": normalizer.state_dict(),
        "classes": CLASSES,
    }, output_path)
    print(f"\nSaved checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train FormCoach's rep-quality LSTM classifier.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="../models/form_classifier.pt")
    args = parser.parse_args()

    train(args.epochs, args.n_per_class, args.batch_size, args.lr, args.output)


if __name__ == "__main__":
    main()
