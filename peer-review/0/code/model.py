import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


class CloudMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_lr():
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )


def make_xgb(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        random_state=42,
    )


def run_cv_and_test(clf_lr, clf_xgb, clf_mlp, x_tr_sc, x_te_sc, y_tr, ps):
    results = {}

    lr_s = cross_val_score(clf_lr, x_tr_sc, y_tr, cv=ps, scoring="roc_auc")
    xgb_s = cross_val_score(clf_xgb, x_tr_sc, y_tr, cv=ps, scoring="roc_auc")
    mlp_s = cross_val_score(clf_mlp, x_tr_sc, y_tr, cv=ps, scoring="roc_auc")

    for name, clf, scores in [
        ("Logistic Regression", clf_lr, lr_s),
        ("XGBoost", clf_xgb, xgb_s),
        ("MLP", clf_mlp, mlp_s),
    ]:
        clf.fit(x_tr_sc, y_tr)
        prob = clf.predict_proba(x_te_sc)[:, 1]
        pred = clf.predict(x_te_sc)
        results[name] = {
            "cv_scores": scores,
            "prob": prob,
            "pred": pred,
        }
    return results


def train_mlp_early_stopping(
    x_train_sc,
    y_train,
    val_mask,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=2048,
    n_epochs=80,
    patience=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr_t = torch.tensor(x_train_sc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    w_pos = neg_count / pos_count

    sample_w = torch.where(y_tr_t.squeeze() == 1, w_pos, 1.0)
    train_ds = TensorDataset(x_tr_t, y_tr_t, sample_w)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_val_t = torch.tensor(x_train_sc[val_mask], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_train[val_mask], dtype=torch.float32).unsqueeze(1).to(device)

    mlp_model = CloudMLP(x_train_sc.shape[1]).to(device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        mlp_model.train()
        batch_losses = []

        for xb, yb_batch, wb in train_loader:
            xb = xb.to(device)
            yb_batch = yb_batch.to(device)
            wb = wb.to(device)

            logits = mlp_model(xb)
            loss = (criterion(logits, yb_batch).squeeze() * wb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(float(np.mean(batch_losses)))

        mlp_model.eval()
        with torch.no_grad():
            val_logits = mlp_model(x_val_t)
            val_w = torch.where(y_val_t.squeeze() == 1, w_pos, 1.0)
            vl = (criterion(val_logits, y_val_t).squeeze() * val_w).mean().item()

        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in mlp_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        mlp_model.load_state_dict(best_state)

    return mlp_model, train_losses, val_losses, best_val_loss, device
