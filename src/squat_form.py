import argparse
import csv
import json
import math
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "codex" / "artifacts"
FEATURES = [
    "knee_flexion",
    "hip_flexion",
    "trunk_lean",
    "ankle_dorsiflexion",
    "femur_tibia_ratio",
    "tibia_torso_ratio",
]

EC3D_JOINTS = {
    "hip_l": 12,
    "knee_l": 13,
    "ankle_l": 14,
    "foot_l": 19,
    "hip_r": 9,
    "knee_r": 10,
    "ankle_r": 11,
    "foot_r": 22,
    "shoulder_l": 5,
    "shoulder_r": 2,
}
REHAB_JOINTS = {
    "hip_l": 16,
    "knee_l": 17,
    "ankle_l": 18,
    "foot_l": 19,
    "hip_r": 21,
    "knee_r": 22,
    "ankle_r": 23,
    "foot_r": 24,
    "shoulder_l": 6,
    "shoulder_r": 11,
}

MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
MP_L_HIP, MP_R_HIP = 23, 24
MP_L_KNEE, MP_R_KNEE = 25, 26
MP_L_ANKLE, MP_R_ANKLE = 27, 28
MP_L_HEEL, MP_R_HEEL = 29, 30
MP_REQUIRED = [MP_L_SHOULDER, MP_R_SHOULDER, MP_L_HIP, MP_R_HIP, MP_L_KNEE, MP_R_KNEE, MP_L_ANKLE, MP_R_ANKLE]


def angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return float("nan")
    cos = float(np.dot(ba, bc) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _points(poses, joints, name):
    return poses[:, joints[name], :]


def vertical_axis_and_sign(poses, joints):
    hip_mid = (_points(poses, joints, "hip_l") + _points(poses, joints, "hip_r")) / 2
    shoulder_mid = (_points(poses, joints, "shoulder_l") + _points(poses, joints, "shoulder_r")) / 2
    vec = shoulder_mid - hip_mid
    axis = int(np.argmax(np.nanmedian(np.abs(vec), axis=0)))
    sign = 1.0 if np.nanmedian(vec[:, axis]) >= 0 else -1.0
    return axis, sign


def segment_lengths(poses, joints):
    femur = (
        np.linalg.norm(_points(poses, joints, "hip_l") - _points(poses, joints, "knee_l"), axis=1)
        + np.linalg.norm(_points(poses, joints, "hip_r") - _points(poses, joints, "knee_r"), axis=1)
    ) / 2
    tibia = (
        np.linalg.norm(_points(poses, joints, "knee_l") - _points(poses, joints, "ankle_l"), axis=1)
        + np.linalg.norm(_points(poses, joints, "knee_r") - _points(poses, joints, "ankle_r"), axis=1)
    ) / 2
    hip_mid = (_points(poses, joints, "hip_l") + _points(poses, joints, "hip_r")) / 2
    shoulder_mid = (_points(poses, joints, "shoulder_l") + _points(poses, joints, "shoulder_r")) / 2
    torso = np.linalg.norm(hip_mid - shoulder_mid, axis=1)
    return {
        "femur": float(np.nanmedian(femur)),
        "tibia": float(np.nanmedian(tibia)),
        "torso": float(np.nanmedian(torso)),
    }


def features_from_3d_frame(frame, joints, vertical_axis, vertical_sign, lengths):
    def pt(name):
        return frame[joints[name]]

    hip_mid = (pt("hip_l") + pt("hip_r")) / 2
    shoulder_mid = (pt("shoulder_l") + pt("shoulder_r")) / 2
    up = np.zeros(3)
    up[vertical_axis] = vertical_sign
    femur_tibia = lengths["femur"] / (lengths["tibia"] + 1e-8)
    tibia_torso = lengths["tibia"] / (lengths["torso"] + 1e-8)
    return {
        "knee_flexion": (angle(pt("ankle_l"), pt("knee_l"), pt("hip_l")) + angle(pt("ankle_r"), pt("knee_r"), pt("hip_r"))) / 2,
        "hip_flexion": (angle(pt("knee_l"), pt("hip_l"), pt("shoulder_l")) + angle(pt("knee_r"), pt("hip_r"), pt("shoulder_r"))) / 2,
        "trunk_lean": angle(hip_mid + up, hip_mid, shoulder_mid),
        "ankle_dorsiflexion": (angle(pt("foot_l"), pt("ankle_l"), pt("knee_l")) + angle(pt("foot_r"), pt("ankle_r"), pt("knee_r"))) / 2,
        "femur_tibia_ratio": femur_tibia,
        "tibia_torso_ratio": tibia_torso,
    }


def _append_pose_rows(rows, poses, meta_rows, joints, dataset):
    axis, sign = vertical_axis_and_sign(poses, joints)
    for subject_id, group in pd.DataFrame(meta_rows).groupby("subject_id", sort=False):
        idx = group.index.to_numpy()
        lengths = segment_lengths(poses[idx], joints)
        for frame_idx, meta in zip(idx, group.to_dict("records")):
            feats = features_from_3d_frame(poses[frame_idx], joints, axis, sign, lengths)
            if all(np.isfinite(feats[name]) for name in FEATURES):
                rows.append({**meta, "dataset": dataset, "vertical_axis": axis, **feats})


def load_ec3d(ec3d_path):
    with Path(ec3d_path).open("rb") as f:
        raw = pickle.load(f)
    labels = pd.DataFrame(raw["labels"], columns=["act", "subject", "ec3d_label", "rep_id", "frame_id"])
    labels[["ec3d_label", "rep_id"]] = labels[["ec3d_label", "rep_id"]].astype(int)
    squat = labels["act"] == "SQUAT"
    labels = labels[squat].reset_index(drop=True)
    poses = raw["poses"][squat.to_numpy()].transpose(0, 2, 1).astype("float32")
    meta = []
    for _, row in labels.iterrows():
        meta.append(
            {
                "subject_id": f"ec3d_{row.subject}",
                "frame_id": int(row.frame_id),
                "rep_id": int(row.rep_id),
                "label": int(row.ec3d_label == 1),
                "ec3d_label": int(row.ec3d_label),
                "source_id": row.subject,
            }
        )
    rows = []
    _append_pose_rows(rows, poses, meta, EC3D_JOINTS, "ec3d")
    return pd.DataFrame(rows)


def _side_camera(cam17_orientation):
    if cam17_orientation == "profile":
        return "cam17"
    if cam17_orientation == "front":
        return "cam18"
    return "half-profile"


def load_rehab24(rehab_dir):
    root = Path(rehab_dir)
    seg = pd.read_csv(root / "Segmentation.csv", sep=";")
    seg = seg[(seg["exercise_id"] == 6) & (seg["mocap_erroneous"] == 0)]
    rows = []
    for path in sorted((root / "3d_joints" / "Ex6").glob("*.npy")):
        video_id = path.stem.replace("-30fps", "")
        video_seg = seg[seg["video_id"] == video_id]
        if video_seg.empty:
            continue
        poses = np.load(path)[:, :, :3].astype("float32")
        meta = []
        pose_indices = []
        for _, rep in video_seg.iterrows():
            start = max(0, int(rep.first_frame))
            end = min(len(poses) - 1, int(rep.last_frame))
            for frame_id in range(start, end + 1):
                pose_indices.append(frame_id)
                meta.append(
                    {
                        "subject_id": f"rehab_{int(rep.person_id):03d}",
                        "frame_id": frame_id,
                        "rep_id": int(rep.repetition_number),
                        "label": int(rep.correctness),
                        "ec3d_label": "",
                        "source_id": video_id,
                        "side_camera": _side_camera(rep.cam17_orientation),
                    }
                )
        if pose_indices:
            _append_pose_rows(rows, poses[np.array(pose_indices)], meta, REHAB_JOINTS, "rehab24")
    return pd.DataFrame(rows)


def build_dataset(args):
    frames = [
        load_ec3d(args.ec3d),
        load_rehab24(args.rehab24),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=FEATURES + ["label"])
    df["label"] = df["label"].astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    used_video_features = "dataset" in df.columns and "rehab24_mediapipe" in set(df["dataset"])
    report = {
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "by_dataset": df["dataset"].value_counts().to_dict(),
        "labels": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "note": "Training uses available 3D pose files. Dataset RGB videos were not present locally, so MediaPipe is used for live inference only.",
    }
    print(json.dumps(report, indent=2))
    return df


class SquatMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _split_groups(groups, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def _best_threshold(y_true, probs):
    best = (0.5, -1.0)
    for threshold in np.linspace(0.1, 0.9, 81):
        score = f1_score(y_true, probs >= threshold, average="macro", zero_division=0)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def _metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def _train_epochs(x, y, device, lr, weight_decay, noise_std, epochs, batch_size, val=None, patience=15):
    model = SquatMLP().to(device)
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    best_state, best_epoch, best_score, stale = None, 0, -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_t), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_t[idx]
            if noise_std:
                xb = xb + torch.randn_like(xb) * noise_std
            loss = loss_fn(model(xb), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val is None:
            continue
        vx, vy = val
        probs = predict_probs(model, vx, device)
        _, score = _best_threshold(vy, probs)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_epoch or epochs, best_score


def predict_probs(model, x, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)

    feature_path = Path(args.features)
    if feature_path.exists() and not args.rebuild:
        df = pd.read_csv(feature_path)
    else:
        build_args = argparse.Namespace(ec3d=args.ec3d, rehab24=args.rehab24, out=args.features)
        df = build_dataset(build_args)
    used_video_features = "dataset" in df.columns and "rehab24_mediapipe" in set(df["dataset"])
    raw_rows = int(len(df))
    if args.max_train_knee:
        df = df[df["knee_flexion"] <= args.max_train_knee].copy()

    trainval_idx, test_idx = _split_groups(df["subject_id"].to_numpy(), args.test_size, args.seed)
    train_idx_rel, val_idx_rel = _split_groups(df.iloc[trainval_idx]["subject_id"].to_numpy(), args.val_size, args.seed + 1)
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    x_all = df[FEATURES].to_numpy("float32")
    y_all = df["label"].to_numpy("int64")
    scaler = StandardScaler().fit(x_all[train_idx])
    x_scaled = scaler.transform(x_all).astype("float32")

    candidates = []
    for lr, weight_decay, noise_std in [(0.001, 1e-4, 0.02), (0.003, 1e-4, 0.02), (0.001, 1e-3, 0.01), (0.003, 1e-3, 0.01)]:
        model, best_epoch, val_f1 = _train_epochs(
            x_scaled[train_idx],
            y_all[train_idx],
            device,
            lr,
            weight_decay,
            noise_std,
            args.epochs,
            args.batch_size,
            val=(x_scaled[val_idx], y_all[val_idx]),
            patience=args.patience,
        )
        val_probs = predict_probs(model, x_scaled[val_idx], device)
        threshold, _ = _best_threshold(y_all[val_idx], val_probs)
        candidates.append(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "noise_std": noise_std,
                "best_epoch": int(best_epoch),
                "val_f1": float(val_f1),
                "threshold": float(threshold),
            }
        )

    best = max(candidates, key=lambda row: row["val_f1"])
    final_scaler = StandardScaler().fit(x_all[trainval_idx])
    final_x = final_scaler.transform(x_all).astype("float32")
    model, _, _ = _train_epochs(
        final_x[trainval_idx],
        y_all[trainval_idx],
        device,
        best["lr"],
        best["weight_decay"],
        best["noise_std"],
        max(1, best["best_epoch"]),
        args.batch_size,
        val=None,
    )
    trainval_probs = predict_probs(model, final_x[trainval_idx], device)
    threshold, _ = _best_threshold(y_all[trainval_idx], trainval_probs)
    test_probs = predict_probs(model, final_x[test_idx], device)

    out_model = Path(args.model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": FEATURES,
            "scaler_mean": final_scaler.mean_.astype("float32").tolist(),
            "scaler_scale": final_scaler.scale_.astype("float32").tolist(),
            "threshold": float(threshold),
            "classify_max_knee": float(args.max_train_knee) if args.max_train_knee else None,
        },
        out_model,
    )

    test_df = df.iloc[test_idx].copy()
    test_df["prob_correct"] = test_probs
    test_df["pred"] = (test_probs >= threshold).astype(int)
    by_dataset = {
        name: _metrics(part["label"].to_numpy(), part["prob_correct"].to_numpy(), threshold)
        for name, part in test_df.groupby("dataset")
    }
    by_subject = {
        name: float(accuracy_score(part["label"], part["pred"]))
        for name, part in test_df.groupby("subject_id")
    }
    ec3d_breakdown = {}
    ec3d = test_df[test_df["dataset"] == "ec3d"]
    if not ec3d.empty:
        ec3d_breakdown = {
            str(int(name)): float(accuracy_score(part["label"], part["pred"]))
            for name, part in ec3d.groupby("ec3d_label")
        }

    report = {
        "device": str(device),
        "features": FEATURES,
        "model": str(out_model),
        "feature_data": str(feature_path),
        "raw_rows": raw_rows,
        "rows": int(len(df)),
        "max_train_knee": float(args.max_train_knee) if args.max_train_knee else None,
        "subjects": int(df["subject_id"].nunique()),
        "trainval_subjects": sorted(df.iloc[trainval_idx]["subject_id"].unique().tolist()),
        "test_subjects": sorted(df.iloc[test_idx]["subject_id"].unique().tolist()),
        "best_candidate": best,
        "candidates": sorted(candidates, key=lambda row: row["val_f1"], reverse=True),
        "threshold": float(threshold),
        "trainval": _metrics(y_all[trainval_idx], trainval_probs, threshold),
        "test": _metrics(y_all[test_idx], test_probs, threshold),
        "test_by_dataset": by_dataset,
        "test_by_subject_accuracy": by_subject,
        "test_ec3d_label_accuracy": ec3d_breakdown,
        "skipped": (
            "EC3D RGB videos are still not present locally; this run trains on REHAB24 MediaPipe video features."
            if used_video_features
            else "Dataset RGB videos were not present locally, so training uses 3D pose angles; live inference uses MediaPipe."
        ),
    }
    metrics_path = Path(args.metrics)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(out_model), "metrics": str(metrics_path), "test": report["test"]}, indent=2))


@dataclass
class Calibration:
    femur: float
    tibia: float
    torso: float

    def ratios(self):
        return {
            "femur_tibia_ratio": self.femur / (self.tibia + 1e-8),
            "tibia_torso_ratio": self.tibia / (self.torso + 1e-8),
        }


def features_from_mediapipe(landmarks, calibration):
    lm = np.array([[p.x, p.y, getattr(p, "visibility", 1.0)] for p in landmarks], dtype="float32")
    xy = lm[:, :2]
    side = "left" if xy[MP_L_HIP, 0] > xy[MP_R_HIP, 0] else "right"
    hip, knee, ankle, heel, shoulder = (
        (MP_L_HIP, MP_L_KNEE, MP_L_ANKLE, MP_L_HEEL, MP_L_SHOULDER)
        if side == "left"
        else (MP_R_HIP, MP_R_KNEE, MP_R_ANKLE, MP_R_HEEL, MP_R_SHOULDER)
    )
    hip_mid = (xy[MP_L_HIP] + xy[MP_R_HIP]) / 2
    shoulder_mid = (xy[MP_L_SHOULDER] + xy[MP_R_SHOULDER]) / 2
    height = abs(float(((xy[MP_L_ANKLE, 1] + xy[MP_R_ANKLE, 1]) / 2) - shoulder_mid[1]))
    side_width = abs(float(xy[MP_L_HIP, 0] - xy[MP_R_HIP, 0])) + abs(float(xy[MP_L_SHOULDER, 0] - xy[MP_R_SHOULDER, 0]))
    feats = {
        "knee_flexion": angle(xy[ankle], xy[knee], xy[hip]),
        "hip_flexion": angle(xy[knee], xy[hip], xy[shoulder]),
        "trunk_lean": angle(hip_mid + np.array([0.0, -1.0]), hip_mid, shoulder_mid),
        "ankle_dorsiflexion": angle(xy[heel], xy[ankle], xy[knee]),
        "min_visibility": float(np.min(lm[MP_REQUIRED, 2])),
        "side_width_ratio": side_width / (height + 1e-8),
        "side": side,
    }
    feats.update(calibration.ratios())
    return feats


def calibrate(cap, pose, seconds, min_visibility):
    started = time.monotonic()
    frames = []
    while time.monotonic() - started < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            if min(p.visibility for i, p in enumerate(lm) if i in MP_REQUIRED) >= min_visibility:
                xy = np.array([[p.x, p.y] for p in lm], dtype="float32")
                frames.append(xy)
        cv2.putText(frame, "Stand tall side-on for calibration", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 220), 2)
        cv2.imshow("Codex Squat Model", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if not frames:
        raise RuntimeError("Calibration failed: no visible pose.")
    arr = np.stack(frames)
    femur = np.median(
        [
            np.linalg.norm(arr[:, MP_L_HIP] - arr[:, MP_L_KNEE], axis=1),
            np.linalg.norm(arr[:, MP_R_HIP] - arr[:, MP_R_KNEE], axis=1),
        ]
    )
    tibia = np.median(
        [
            np.linalg.norm(arr[:, MP_L_KNEE] - arr[:, MP_L_ANKLE], axis=1),
            np.linalg.norm(arr[:, MP_R_KNEE] - arr[:, MP_R_ANKLE], axis=1),
        ]
    )
    hip_mid = (arr[:, MP_L_HIP] + arr[:, MP_R_HIP]) / 2
    shoulder_mid = (arr[:, MP_L_SHOULDER] + arr[:, MP_R_SHOULDER]) / 2
    torso = np.median(np.linalg.norm(hip_mid - shoulder_mid, axis=1))
    return Calibration(float(femur), float(tibia), float(torso))


def load_artifact(path, device):
    artifact = torch.load(path, map_location=device, weights_only=False)
    model = SquatMLP().to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    mean = np.array(artifact["scaler_mean"], dtype="float32")
    scale = np.array(artifact["scaler_scale"], dtype="float32")
    threshold = float(artifact["threshold"])
    return model, mean, scale, threshold, artifact.get("classify_max_knee")


def live(args):
    device = _device(args.device)
    model, mean, scale, threshold, classify_max_knee = load_artifact(args.model, device)
    if args.classify_max_knee:
        classify_max_knee = args.classify_max_knee
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_model_frames.csv"
    fields = ["time_s", "prob_correct", "prediction", "quality", *FEATURES, "side", "min_visibility", "side_width_ratio"]

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        calibration = calibrate(cap, pose, args.calibration_seconds, args.min_visibility)
        window = deque(maxlen=args.window)
        started = time.monotonic()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic() - started
            if args.max_seconds and now >= args.max_seconds:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            quality = "No pose"
            prob = float("nan")
            pred = ""
            feats = {}
            if result.pose_landmarks:
                feats = features_from_mediapipe(result.pose_landmarks.landmark, calibration)
                if feats["min_visibility"] < args.min_visibility:
                    quality = "Move into frame"
                elif feats["side_width_ratio"] > args.max_side_width:
                    quality = "Turn more sideways"
                elif classify_max_knee and feats["knee_flexion"] > classify_max_knee:
                    quality = "Stand/neutral"
                else:
                    quality = "Pose OK"
                    x = np.array([[feats[name] for name in FEATURES]], dtype="float32")
                    x = (x - mean) / scale
                    prob = float(predict_probs(model, x, device)[0])
                    window.append(prob)
                    smooth = float(np.mean(window))
                    pred = "correct" if smooth >= threshold else "incorrect"
                mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            row = {field: "" for field in fields}
            row.update({"time_s": round(now, 3), "prob_correct": round(prob, 4) if np.isfinite(prob) else "", "prediction": pred, "quality": quality})
            row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items() if k in row})
            writer.writerow(row)

            color = (0, 180, 0) if pred == "correct" else (0, 80, 220)
            label = f"{quality}  {pred or ''}  p={prob:.2f}" if np.isfinite(prob) else quality
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Codex Squat Model", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved frames: {csv_path}")


def self_test():
    assert round(angle(np.array([0, 0]), np.array([1, 0]), np.array([1, 1]))) == 90
    poses = np.zeros((3, 3, 3), dtype="float32")
    poses[:, 0, 1] = 0.0
    poses[:, 1, 1] = 1.0
    joints = {"hip_l": 0, "hip_r": 0, "shoulder_l": 1, "shoulder_r": 1, "knee_l": 0, "knee_r": 0, "ankle_l": 0, "ankle_r": 0, "foot_l": 0, "foot_r": 0}
    assert vertical_axis_and_sign(poses, joints) == (1, 1.0)
    model = SquatMLP()
    assert model(torch.zeros(2, 6)).shape == (2,)
    print("self-test ok")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--ec3d", default=str(PROJECT_ROOT / "external" / "ec3d_repo" / "data" / "EC3D" / "data_3D.pickle"))
    build_p.add_argument("--rehab24", default=str(PROJECT_ROOT / "data" / "rehab24"))
    build_p.add_argument("--out", default=str(ARTIFACT_DIR / "squat_features.csv"))
    build_p.set_defaults(func=build_dataset)

    train_p = sub.add_parser("train")
    train_p.add_argument("--ec3d", default=str(PROJECT_ROOT / "external" / "ec3d_repo" / "data" / "EC3D" / "data_3D.pickle"))
    train_p.add_argument("--rehab24", default=str(PROJECT_ROOT / "data" / "rehab24"))
    train_p.add_argument("--features", default=str(ARTIFACT_DIR / "squat_features.csv"))
    train_p.add_argument("--model", default=str(ARTIFACT_DIR / "squat_mlp.pt"))
    train_p.add_argument("--metrics", default=str(ARTIFACT_DIR / "squat_mlp_metrics.json"))
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument("--epochs", type=int, default=80)
    train_p.add_argument("--patience", type=int, default=12)
    train_p.add_argument("--batch-size", type=int, default=1024)
    train_p.add_argument("--test-size", type=float, default=0.25)
    train_p.add_argument("--val-size", type=float, default=0.25)
    train_p.add_argument("--seed", type=int, default=7)
    train_p.add_argument("--max-train-knee", type=float, default=170)
    train_p.add_argument("--rebuild", action="store_true")
    train_p.set_defaults(func=train)

    live_p = sub.add_parser("live")
    live_p.add_argument("--model", default=str(ARTIFACT_DIR / "squat_mlp.pt"))
    live_p.add_argument("--device", default="auto")
    live_p.add_argument("--camera", type=int, default=0)
    live_p.add_argument("--out-dir", default=str(PROJECT_ROOT / "codex" / "live_captures"))
    live_p.add_argument("--calibration-seconds", type=float, default=3)
    live_p.add_argument("--min-visibility", type=float, default=0.55)
    live_p.add_argument("--max-side-width", type=float, default=0.55)
    live_p.add_argument("--classify-max-knee", type=float, default=0)
    live_p.add_argument("--window", type=int, default=5)
    live_p.add_argument("--max-seconds", type=float, default=0)
    live_p.set_defaults(func=live)

    test_p = sub.add_parser("self-test")
    test_p.set_defaults(func=lambda _args: self_test())
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
