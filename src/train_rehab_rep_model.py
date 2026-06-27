import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "codex" / "artifacts"
BASE_FEATURES = [
    "knee_flexion",
    "hip_flexion",
    "trunk_lean",
    "ankle_dorsiflexion",
    "femur_tibia_ratio",
    "tibia_torso_ratio",
    "min_visibility",
    "side_width_ratio",
]


def aggregate_reps(df, active_knee):
    if active_knee:
        df = df[df["knee_flexion"] <= active_knee].copy()

    rows = []
    keys = ["subject_id", "source_id", "rep_id", "label"]
    for key, group in df.groupby(keys, sort=False):
        row = {
            "subject_id": key[0],
            "source_id": key[1],
            "rep_id": int(key[2]),
            "label": int(key[3]),
            "n_frames": int(len(group)),
        }
        for name in BASE_FEATURES:
            values = group[name].astype(float)
            quantiles = values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=0))
            row[f"{name}_min"] = float(values.min())
            row[f"{name}_max"] = float(values.max())
            row[f"{name}_range"] = float(values.max() - values.min())
            row[f"{name}_q10"] = float(quantiles[0.1])
            row[f"{name}_q25"] = float(quantiles[0.25])
            row[f"{name}_q50"] = float(quantiles[0.5])
            row[f"{name}_q75"] = float(quantiles[0.75])
            row[f"{name}_q90"] = float(quantiles[0.9])
        rows.append(row)

    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def make_model(name, params, seed):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=params["C"],
                max_iter=3000,
                random_state=seed,
            ),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(name)


def candidate_grid():
    for active_knee in [170, 160, 150, 140]:
        for c in [0.1, 0.3, 1.0, 3.0]:
            yield {"model": "logreg", "params": {"C": c}, "active_knee": active_knee}
        for model in ["extra_trees", "random_forest"]:
            for max_depth in [3, 4, 5]:
                yield {
                    "model": model,
                    "params": {"n_estimators": 400, "max_depth": max_depth, "min_samples_leaf": 2},
                    "active_knee": active_knee,
                }


def loso_eval(rep_df, candidate, seed):
    feature_columns = [c for c in rep_df.columns if c not in {"subject_id", "source_id", "rep_id", "label"}]
    x = rep_df[feature_columns].to_numpy("float32")
    y = rep_df["label"].to_numpy("int64")
    groups = rep_df["subject_id"].to_numpy()
    preds = np.zeros_like(y)

    for train_idx, test_idx in LeaveOneGroupOut().split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            preds[test_idx] = int(np.bincount(y[train_idx]).argmax())
            continue
        model = make_model(candidate["model"], candidate["params"], seed)
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx] = model.predict(x[test_idx])

    report = metrics(y, preds)
    by_subject = {
        subject: metrics(y[groups == subject], preds[groups == subject])
        for subject in sorted(set(groups))
    }
    return report, by_subject, feature_columns


def split_eval(rep_df, candidate, feature_columns, seed):
    x = rep_df[feature_columns].to_numpy("float32")
    y = rep_df["label"].to_numpy("int64")
    groups = rep_df["subject_id"].to_numpy()
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, groups=groups))
    model = make_model(candidate["model"], candidate["params"], seed)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    return {
        "test_subjects": sorted(set(groups[test_idx])),
        "train_subjects": sorted(set(groups[train_idx])),
        "test": metrics(y[test_idx], pred),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    df = pd.read_csv(args.features)
    candidates = []

    for candidate in candidate_grid():
        rep_df = aggregate_reps(df, candidate["active_knee"])
        loso, by_subject, feature_columns = loso_eval(rep_df, candidate, args.seed)
        candidates.append(
            {
                **candidate,
                "rows": int(len(rep_df)),
                "subjects": int(rep_df["subject_id"].nunique()),
                "labels": {str(k): int(v) for k, v in rep_df["label"].value_counts().sort_index().items()},
                "loso": loso,
                "loso_by_subject": by_subject,
                "feature_columns": feature_columns,
            }
        )

    best = max(candidates, key=lambda c: (c["loso"]["balanced_accuracy"], c["loso"]["macro_f1"]))
    final_rep_df = aggregate_reps(df, best["active_knee"])
    final_features = best["feature_columns"]
    x = final_rep_df[final_features].to_numpy("float32")
    y = final_rep_df["label"].to_numpy("int64")
    final_model = make_model(best["model"], best["params"], args.seed)
    final_model.fit(x, y)

    out_model = Path(args.model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "feature_columns": final_features,
                "base_features": BASE_FEATURES,
                "active_knee": best["active_knee"],
                "label_map": {"correct": 1, "incorrect": 0},
            },
            f,
        )

    split = split_eval(final_rep_df, best, final_features, args.seed)
    report = {
        "features": args.features,
        "model": str(out_model),
        "best": {k: best[k] for k in ["model", "params", "active_knee", "rows", "subjects", "labels", "loso", "loso_by_subject"]},
        "split_eval": split,
        "candidates": sorted(candidates, key=lambda c: c["loso"]["balanced_accuracy"], reverse=True),
    }
    Path(args.metrics).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(out_model), "metrics": args.metrics, "best": report["best"], "split_eval": split}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(ARTIFACT_DIR / "rehab24_mediapipe_features.csv"))
    parser.add_argument("--model", default=str(ARTIFACT_DIR / "rehab24_rep_model.pkl"))
    parser.add_argument("--metrics", default=str(ARTIFACT_DIR / "rehab24_rep_model_metrics.json"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
