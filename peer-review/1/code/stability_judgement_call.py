import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve

import feature_engineering
import get_embedding
import lightgbModel


FEATURE_SLICE = slice(2, 10)  # NDAI..AN


def mode_settings():
    """Return the four preprocessing modes."""
    return {
        "raw": {"standardize": False, "sigma3": False},
        "standardize": {"standardize": True, "sigma3": False},
        "sigma3": {"standardize": False, "sigma3": True},
        "standardize_sigma3": {"standardize": True, "sigma3": True},
    }


def read_npz(path):
    """Load one npz file and return data + key."""
    npz = np.load(path)
    key = list(npz.files)[0]
    return npz[key], key


def write_npz(path, arr, key="arr_0"):
    """Save one array into npz."""
    np.savez(path, **{key: arr})


def safe_std(std):
    """Replace zero std with 1."""
    std = np.asarray(std, dtype=np.float64)
    std[std == 0] = 1.0
    return std


def fit_train_stats(train_npz_paths):
    """Fit mean/std on train images."""
    blocks = []
    for path in train_npz_paths:
        arr, _ = read_npz(path)
        blocks.append(arr[:, FEATURE_SLICE].astype(np.float64))
    x = np.vstack(blocks)
    mean = x.mean(axis=0)
    std = safe_std(x.std(axis=0))
    return mean, std


def apply_mode(arr, mean, std, mode_name, sigma_threshold=3.0):
    """Apply one preprocessing mode to one array."""
    out = arr.copy()
    before = out.shape[0]
    cfg = mode_settings()[mode_name]

    # Remove rows with extreme z-score across any feature.
    if cfg["sigma3"]:
        z = (out[:, FEATURE_SLICE] - mean) / std
        keep = (np.abs(z) <= sigma_threshold).all(axis=1)
        out = out[keep]

    # Standardize feature columns only.
    if cfg["standardize"]:
        out[:, FEATURE_SLICE] = (out[:, FEATURE_SLICE] - mean) / std

    after = out.shape[0]
    return out, before, after


def preprocess_npz_files(
    image_ids,
    train_image_ids,
    source_dir="../data",
    output_dir="../data/stability_judgement_call",
    sigma_threshold=3.0,
):
    """Create 4 preprocessed npz variants for each image."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [source_dir / f"{img}.npz" for img in train_image_ids]
    mean, std = fit_train_stats(train_paths)

    rows = []
    for mode_name in mode_settings():
        for image_id in image_ids:
            arr, key = read_npz(source_dir / f"{image_id}.npz")
            out, before, after = apply_mode(arr, mean, std, mode_name, sigma_threshold)

            out_path = output_dir / f"{image_id}__{mode_name}.npz"
            write_npz(out_path, out, key=key)

            rows.append(
                {
                    "mode": mode_name,
                    "image_id": image_id,
                    "n_before": int(before),
                    "n_after": int(after),
                    "n_dropped": int(before - after),
                    "drop_rate": float((before - after) / before),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["mode", "image_id"]).reset_index(drop=True)
    summary.to_csv(output_dir / "preprocess_summary.csv", index=False)

    return summary


def run_mode_feature_pipeline(
    mode_name,
    image_ids,
    work_dir="../data/stability_judgement_call",
    config_path="configs/exp_028_finetune.yaml",
    weights_path="../results/exp_028/exp_028_finetune.pt",
):
    """Run embedding + feature engineering for one mode."""
    work_dir = Path(work_dir)
    files = [f"{img}__{mode_name}.npz" for img in image_ids]

    # get_embedding writes <stem>_ae.csv into work_dir
    get_embedding.get_embedding(
        config_path=config_path,
        weights_path=weights_path,
        labeled_only=False,
        filepaths=files,
        data_dir=str(work_dir),
        output_dir=str(work_dir),
    )

    # feature_engineering writes <stem>.csv into work_dir
    feature_engineering.feature_engineering(
        data_dir=str(work_dir),
        path_output=str(work_dir),
        embedding_dir=str(work_dir),
        labeled_only=False,
        filepaths=files,
    )


def train_lightgbm_mode(
    mode_name,
    train_image_ids,
    test_image_id,
    work_dir="../data/stability_judgement_call",
    model_dir="../results/stability_judgement_call",
    top_n_features=105,
    n_estimators=1000,
    excluded_prefixes=None,
):
    """Train and evaluate LightGBM for one mode."""
    work_dir = Path(work_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_frames = [pd.read_csv(work_dir / f"{img}__{mode_name}.csv") for img in train_image_ids]
    test_df = pd.read_csv(work_dir / f"{test_image_id}__{mode_name}.csv")

    df_train = pd.concat(train_frames, ignore_index=True)

    output = lightgbModel.run_lightgbm_pipeline(
        df_train=df_train,
        df_test=test_df,
        top_n_features=top_n_features,
        excluded_prefixes=excluded_prefixes,
        n_estimators=n_estimators,
        plot_features=False,
        plot_cm=False,
    )

    y_true = output["y_test_lgb"].to_numpy()
    y_pred = output["results"]["y_pred"]
    y_prob = output["model_final"].predict_proba(output["X_test_top"])[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    metrics = {
        "mode": mode_name,
        "n_train": int(len(output["X_train_top"])),
        "n_test": int(len(output["X_test_top"])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "cv_best_score": float(output["grid_search"].best_score_),
    }

    # Save one model file per mode in results/.
    model_path = model_dir / f"lgb_model__{mode_name}.pkl"
    joblib.dump(output["model_final"], model_path)

    return {
        "mode": mode_name,
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr,
        "output": output,
    }


def run_full_experiment(
    image_ids,
    train_image_ids,
    test_image_id,
    source_dir="../data",
    work_dir="../data/stability_judgement_call",
    model_dir="../results/stability_judgement_call",
    config_path="configs/exp_028_finetune.yaml",
    weights_path="../results/exp_028/exp_028_finetune.pt",
    sigma_threshold=3.0,
    top_n_features=105,
    n_estimators=1000,
    excluded_prefixes=None,
):
    """Run the full 4-mode stability experiment."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary = preprocess_npz_files(
        image_ids=image_ids,
        train_image_ids=train_image_ids,
        source_dir=source_dir,
        output_dir=work_dir,
        sigma_threshold=sigma_threshold,
    )

    results = {}
    rows = []

    for mode_name in mode_settings():
        print(f"\nmode: {mode_name}")
        run_mode_feature_pipeline(
            mode_name=mode_name,
            image_ids=image_ids,
            work_dir=work_dir,
            config_path=config_path,
            weights_path=weights_path,
        )
        result = train_lightgbm_mode(
            mode_name=mode_name,
            train_image_ids=train_image_ids,
            test_image_id=test_image_id,
            work_dir=work_dir,
            model_dir=model_dir,
            top_n_features=top_n_features,
            n_estimators=n_estimators,
            excluded_prefixes=excluded_prefixes,
        )
        results[mode_name] = result
        rows.append(result["metrics"])

    metrics_df = pd.DataFrame(rows).sort_values("mode").reset_index(drop=True)
    metrics_df.to_csv(work_dir / "stability_metrics.csv", index=False)

    return {
        "preprocess_summary": summary,
        "results_by_mode": results,
        "metrics_df": metrics_df,
    }


def plot_roc_curves(
    results_by_mode,
    title="ROC Comparison",
    save_path="../figs/stability_judgement_call_roc.png",
):
    """Plot ROC curves and save the figure."""
    plt.figure(figsize=(8, 6))
    for mode_name, result in results_by_mode.items():
        auc = result["metrics"]["roc_auc"]
        plt.plot(result["fpr"], result["tpr"], label=f"{mode_name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"ROC figure saved: {save_path}")
    plt.show()
