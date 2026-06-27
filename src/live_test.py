import argparse
import csv
import json
import math
import pickle
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

try:
    from train_rehab_rep_model import aggregate_reps
except ModuleNotFoundError:
    from codex.train_rehab_rep_model import aggregate_reps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "codex/live_captures"
ARTIFACT_DIR = PROJECT_ROOT / "codex/artifacts"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
REQUIRED = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
REFERENCE_FEATURES = ARTIFACT_DIR / "rehab24_mediapipe_features.csv"
READY_SCREEN_SECONDS = 1.2
DRAW_EDGES = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (L_ANKLE, L_HEEL),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (R_ANKLE, R_HEEL),
]


def ensure_pose_model(path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(POSE_MODEL_URL, path)
    return path


def create_landmarker(model_path):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def landmark_visibility(landmark):
    return float(getattr(landmark, "visibility", 1.0))


def angle(a, b, c):
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return float("nan")
    cos = float(np.dot(ba, bc) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def clean_number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return round(value, 4) if np.isfinite(value) else None


def body_type(femur_tibia_ratio, tibia_torso_ratio):
    femur = "long femur" if femur_tibia_ratio >= 1.15 else "short femur" if femur_tibia_ratio <= 0.95 else "balanced femur"
    lower_leg = "long lower leg" if tibia_torso_ratio >= 0.85 else "short lower leg" if tibia_torso_ratio <= 0.55 else "balanced lower leg"
    return f"{femur}, {lower_leg}"


@dataclass
class BodyScan:
    source: str
    frames: int
    femur: float
    tibia: float
    torso: float
    upper_arm: float | None
    forearm: float | None
    arm: float | None
    leg: float
    femur_tibia_ratio: float
    tibia_torso_ratio: float
    arm_leg_ratio: float | None
    body_type: str

    def to_dict(self):
        return {
            "source": self.source,
            "frames": self.frames,
            "body_type": self.body_type,
            "relative_lengths": {
                "femur": clean_number(self.femur),
                "tibia": clean_number(self.tibia),
                "torso": clean_number(self.torso),
                "upper_arm": clean_number(self.upper_arm),
                "forearm": clean_number(self.forearm),
                "arm": clean_number(self.arm),
                "leg": clean_number(self.leg),
            },
            "ratios": {
                "femur_tibia": clean_number(self.femur_tibia_ratio),
                "tibia_torso": clean_number(self.tibia_torso_ratio),
                "arm_leg": clean_number(self.arm_leg_ratio),
            },
            "note": "Lengths are relative webcam units, not centimeters.",
        }


def _side_indices(side):
    if side == "left":
        return L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE, L_HEEL
    return R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE, R_HEEL


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def pose_features(landmarks):
    lm = np.array([[p.x, p.y, landmark_visibility(p)] for p in landmarks], dtype=float)
    side = "left" if lm[L_HIP, 0] > lm[R_HIP, 0] else "right"
    shoulder, elbow, wrist, hip, knee, ankle, heel = _side_indices(side)

    xy = lm[:, :2]
    hip_mid = (xy[L_HIP] + xy[R_HIP]) / 2
    shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
    trunk_lean = angle(hip_mid + np.array([0.0, -1.0]), hip_mid, shoulder_mid)

    femur = np.linalg.norm(xy[hip] - xy[knee])
    tibia = np.linalg.norm(xy[knee] - xy[ankle])
    torso = np.linalg.norm(xy[hip] - xy[shoulder])
    upper_arm = np.linalg.norm(xy[shoulder] - xy[elbow])
    forearm = np.linalg.norm(xy[elbow] - xy[wrist])
    arm = upper_arm + forearm
    leg = femur + tibia
    min_visibility = float(np.min(lm[REQUIRED, 2]))
    scan_visibility = float(np.min(lm[[shoulder, elbow, wrist, hip, knee, ankle], 2]))
    track_visibility = float(np.min(lm[[shoulder, hip, knee, ankle], 2]))
    side_width = float(abs(xy[L_HIP, 0] - xy[R_HIP, 0]) + abs(xy[L_SHOULDER, 0] - xy[R_SHOULDER, 0]))
    height = float(abs(np.mean([xy[L_ANKLE, 1], xy[R_ANKLE, 1]]) - shoulder_mid[1]))

    return {
        "side": side,
        "knee_flexion": angle(xy[ankle], xy[knee], xy[hip]),
        "hip_flexion": angle(xy[knee], xy[hip], xy[shoulder]),
        "trunk_lean": trunk_lean,
        "ankle_dorsiflexion": angle(xy[heel], xy[ankle], xy[knee]),
        "femur_tibia_ratio": femur / (tibia + 1e-8),
        "tibia_torso_ratio": tibia / (torso + 1e-8),
        "femur": femur,
        "tibia": tibia,
        "torso": torso,
        "upper_arm": upper_arm,
        "forearm": forearm,
        "arm": arm,
        "leg": leg,
        "arm_leg_ratio": arm / (leg + 1e-8),
        "min_visibility": min_visibility,
        "scan_visibility": scan_visibility,
        "track_visibility": track_visibility,
        "side_width_ratio": side_width / (height + 1e-8),
    }


def plausible_body_frame(feats, min_visibility=0.5):
    if feats is None:
        return False
    checks = ["femur_tibia_ratio", "tibia_torso_ratio", "femur", "tibia", "torso"]
    if not all(np.isfinite(feats.get(k, float("nan"))) for k in checks):
        return False
    if feats.get("scan_visibility", 0.0) < min_visibility:
        return False
    return 0.6 <= feats["femur_tibia_ratio"] <= 1.8 and 0.25 <= feats["tibia_torso_ratio"] <= 1.4


def body_scan_from_feature_rows(rows, source):
    df = pd.DataFrame(rows)
    df = df.replace("", np.nan)
    for col in ["femur_tibia_ratio", "tibia_torso_ratio", "femur", "tibia", "torso", "upper_arm", "forearm", "arm", "leg", "arm_leg_ratio"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "femur_tibia_ratio" not in df or "tibia_torso_ratio" not in df:
        raise RuntimeError("Body scan needs femur_tibia_ratio and tibia_torso_ratio columns.")

    df = df[
        df["femur_tibia_ratio"].between(0.6, 1.8)
        & df["tibia_torso_ratio"].between(0.25, 1.4)
    ]
    if df.empty:
        raise RuntimeError("Body scan failed: no plausible body-ratio frames.")

    ft = _median(df["femur_tibia_ratio"].to_numpy())
    tt = _median(df["tibia_torso_ratio"].to_numpy())
    tibia = _median(df["tibia"].to_numpy()) if "tibia" in df else 1.0
    femur = _median(df["femur"].to_numpy()) if "femur" in df else tibia * ft
    torso = _median(df["torso"].to_numpy()) if "torso" in df else tibia / (tt + 1e-8)
    upper_arm = _median(df["upper_arm"].to_numpy()) if "upper_arm" in df else None
    forearm = _median(df["forearm"].to_numpy()) if "forearm" in df else None
    arm = _median(df["arm"].to_numpy()) if "arm" in df else None
    leg = _median(df["leg"].to_numpy()) if "leg" in df else femur + tibia
    arm_leg = _median(df["arm_leg_ratio"].to_numpy()) if "arm_leg_ratio" in df else (arm / (leg + 1e-8) if arm else None)
    return BodyScan(
        source=source,
        frames=int(len(df)),
        femur=femur,
        tibia=tibia,
        torso=torso,
        upper_arm=upper_arm,
        forearm=forearm,
        arm=arm,
        leg=leg,
        femur_tibia_ratio=ft,
        tibia_torso_ratio=tt,
        arm_leg_ratio=arm_leg,
        body_type=body_type(ft, tt),
    )


def apply_body_scan(feats, body_scan):
    if feats is None or body_scan is None:
        return feats
    feats = dict(feats)
    feats.update(
        {
            "femur_tibia_ratio": body_scan.femur_tibia_ratio,
            "tibia_torso_ratio": body_scan.tibia_torso_ratio,
            "femur": body_scan.femur,
            "tibia": body_scan.tibia,
            "torso": body_scan.torso,
            "upper_arm": body_scan.upper_arm,
            "forearm": body_scan.forearm,
            "arm": body_scan.arm,
            "leg": body_scan.leg,
            "arm_leg_ratio": body_scan.arm_leg_ratio,
        }
    )
    return feats


def summarize_rep_frames(df):
    df = pd.DataFrame(df).copy()
    for col in ["time_s", "frame_id", "knee_flexion", "hip_flexion", "trunk_lean", "ankle_dorsiflexion", "femur_tibia_ratio", "tibia_torso_ratio"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    duration = None
    if "time_s" in df and df["time_s"].notna().any():
        duration = float(df["time_s"].max() - df["time_s"].min())
    elif "frame_id" in df and df["frame_id"].notna().any():
        duration = float((df["frame_id"].max() - df["frame_id"].min() + 1) / 30.0)
    return {
        "n_frames": int(len(df)),
        "duration_s": duration,
        "knee_flexion_min": float(df["knee_flexion"].min()),
        "hip_flexion_min": float(df["hip_flexion"].min()),
        "trunk_lean_max": float(df["trunk_lean"].max()),
        "ankle_dorsiflexion_min": float(df["ankle_dorsiflexion"].min()),
        "femur_tibia_ratio": float(df["femur_tibia_ratio"].median()),
        "tibia_torso_ratio": float(df["tibia_torso_ratio"].median()),
    }


def _range(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {"low": None, "high": None}
    return {"low": clean_number(series.quantile(0.10)), "high": clean_number(series.quantile(0.90))}


def build_reference_targets(reference_features, body_scan, nearest_reps=20):
    df = pd.read_csv(reference_features)
    correct = df[df["label"] == 1].copy()
    rows = []
    for _, group in correct.groupby(["subject_id", "source_id", "rep_id"], sort=False):
        if len(group) >= 6:
            rows.append(summarize_rep_frames(group))
    reps = pd.DataFrame(rows).dropna()
    if reps.empty:
        raise RuntimeError(f"No usable correct reps found in {reference_features}")

    reps["distance"] = (
        (reps["femur_tibia_ratio"] - body_scan.femur_tibia_ratio).abs()
        + (reps["tibia_torso_ratio"] - body_scan.tibia_torso_ratio).abs()
    )
    keep = min(len(reps), max(8, nearest_reps))
    similar = reps.nsmallest(keep, "distance")
    ranges = {
        "knee_flexion_min": _range(similar["knee_flexion_min"]),
        "trunk_lean_max": _range(similar["trunk_lean_max"]),
        "hip_flexion_min": _range(similar["hip_flexion_min"]),
        "ankle_dorsiflexion_min": _range(similar["ankle_dorsiflexion_min"]),
        "duration_s": _range(similar["duration_s"]),
    }
    return {
        "source": str(reference_features),
        "body_match": body_scan.body_type,
        "total_correct_reps": int(len(reps)),
        "matched_correct_reps": int(len(similar)),
        "ranges": ranges,
        "plain_language": {
            "depth": "knee_flexion_min: lower angle means deeper squat",
            "torso": "trunk_lean_max: larger angle means more forward lean",
            "tempo": "duration_s: estimated rep duration from dataset frames",
        },
    }


def _range_score(value, target, slack=0.0):
    lo, hi = target["low"], target["high"]
    if value is None or lo is None or hi is None or not np.isfinite(value):
        return 0.0, "missing"
    if lo - slack <= value <= hi + slack:
        return 100.0, "in range"
    width = max(hi - lo, 1.0)
    miss = lo - value if value < lo else value - hi
    return max(0.0, 100.0 - 60.0 * miss / width), "low" if value < lo else "high"


def score_against_targets(metrics, targets, model_prob=None):
    ranges = targets["ranges"]
    checks = [
        ("depth", "knee_flexion_min", 0.40, 3.0, True),
        ("torso lean", "trunk_lean_max", 0.25, 3.0, True),
        ("hip angle", "hip_flexion_min", 0.15, 3.0, True),
        ("ankle angle", "ankle_dorsiflexion_min", 0.05, 6.0, False),
        ("tempo", "duration_s", 0.15, 0.35, True),
    ]
    parts, cues = [], []
    for label, key, weight, slack, cueable in checks:
        value = metrics.get(key)
        score, status = _range_score(value, ranges[key], slack)
        parts.append(score * weight)
        if not cueable:
            continue
        if status == "high":
            if key == "knee_flexion_min":
                cues.append("go lower")
            elif key == "trunk_lean_max":
                cues.append("torso lean is above your target range")
            elif key == "duration_s":
                cues.append("rep is slower than the reference range")
            else:
                cues.append(f"{label} is above target range")
        elif status == "low":
            if key == "knee_flexion_min":
                cues.append("depth is deeper than the reference range; keep control")
            elif key == "duration_s":
                cues.append("slow down; tempo is faster than the reference range")
            else:
                cues.append(f"{label} is below target range")
    target_score = sum(parts)
    if model_prob is not None and np.isfinite(model_prob):
        final_score = 0.8 * target_score + 0.2 * model_prob * 100.0
    else:
        final_score = target_score
    return {
        "score": int(round(max(0.0, min(100.0, final_score)))),
        "target_score": clean_number(target_score),
        "model_confidence": clean_number(model_prob),
        "main_cue": cues[0] if cues else "matches the personalized reference range",
        "all_cues": cues[:3],
    }


def overall_label(score, model_prob):
    if score is None:
        return "UNKNOWN"
    if score >= 75 and (model_prob is None or model_prob >= 0.5):
        return "LIKELY CORRECT"
    if model_prob is not None and model_prob <= 0.35 and score < 70:
        return "LIKELY OFF"
    if score >= 60:
        return "NEEDS WORK"
    return "LIKELY OFF"


@dataclass
class RepDetector:
    squat_angle: float
    stand_angle: float
    count_depth_angle: float
    min_duration: float
    max_duration: float
    state: str = "standing"
    rep_id: int = 0
    start_time: float | None = None
    min_knee: float = 999.0
    frames: int = 0
    smooth_knee: float | None = None

    def update(self, knee_angle, now):
        event = ""
        if not np.isfinite(knee_angle):
            return event
        if self.smooth_knee is None:
            self.smooth_knee = knee_angle
        else:
            self.smooth_knee = 0.55 * self.smooth_knee + 0.45 * knee_angle
        knee_angle = self.smooth_knee

        if self.state == "standing" and knee_angle < self.squat_angle:
            self.state = "down"
            self.start_time = now
            self.min_knee = knee_angle
            self.frames = 1
            event = "rep_start"
        elif self.state == "down":
            self.frames += 1
            self.min_knee = min(self.min_knee, knee_angle)
            if knee_angle >= self.stand_angle and self.start_time is not None:
                duration = now - self.start_time
                if self.min_duration <= duration <= self.max_duration and self.frames >= 4 and self.min_knee <= self.count_depth_angle:
                    self.rep_id += 1
                    event = "rep_end"
                else:
                    event = "rep_cancel"
                self.state = "standing"
                self.start_time = None
                self.min_knee = 999.0
                self.frames = 0
        return event


def quality_message(feats, track_min_visibility, max_side_width):
    if feats is None:
        return "No pose", False
    if not all(np.isfinite(feats[k]) for k in ["knee_flexion", "hip_flexion", "trunk_lean", "ankle_dorsiflexion"]):
        return "Pose unstable", False
    if track_min_visibility and feats.get("track_visibility", 0.0) < track_min_visibility:
        return "Move into frame", False
    if feats["knee_flexion"] < 30 or feats["hip_flexion"] < 25 or feats["trunk_lean"] > 95:
        return "Pose unstable; step back side-on", False
    if feats["side_width_ratio"] > max_side_width:
        return "Turn more sideways", False
    return "Pose OK", True


def draw_text(frame, lines, color):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_pose(frame, landmarks, min_visibility):
    height, width = frame.shape[:2]
    pts = [(int(p.x * width), int(p.y * height), landmark_visibility(p)) for p in landmarks]
    for a, b in DRAW_EDGES:
        if pts[a][2] >= min_visibility and pts[b][2] >= min_visibility:
            cv2.line(frame, pts[a][:2], pts[b][:2], (70, 220, 255), 2)
    for i in REQUIRED:
        if pts[i][2] >= min_visibility:
            cv2.circle(frame, pts[i][:2], 4, (0, 180, 0), -1)


def draw_scan_progress(frame, progress, rows_count, min_rows):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    radius = min(width, height) // 8
    cv2.circle(frame, center, radius, (60, 60, 60), 10)
    cv2.ellipse(frame, center, (radius, radius), -90, 0, int(360 * progress), (70, 220, 255), 10)
    cv2.putText(frame, f"{int(progress * 100)}%", (center[0] - 45, center[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 220, 255), 3)
    cv2.putText(frame, f"scan frames: {rows_count}/{min_rows}", (center[0] - 105, center[1] + radius + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


def show_ready_screen(cap, pose, args, started):
    deadline = time.monotonic() + READY_SCREEN_SECONDS
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic() - started
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose.detect_for_video(image, int(now * 1000))
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        if landmarks:
            draw_pose(frame, landmarks, args.min_visibility)

        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 8
        cv2.circle(frame, center, radius, (0, 180, 0), -1)
        cv2.line(frame, (center[0] - radius // 2, center[1]), (center[0] - radius // 8, center[1] + radius // 3), (255, 255, 255), 12)
        cv2.line(frame, (center[0] - radius // 8, center[1] + radius // 3), (center[0] + radius // 2, center[1] - radius // 3), (255, 255, 255), 12)
        cv2.putText(frame, "Ready to start", (center[0] - 150, center[1] + radius + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 3)
        cv2.imshow("Codex Live Squat Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def calibrate_body(cap, pose, args, started):
    rows = []
    deadline = time.monotonic() + args.calibration_seconds if args.calibration_seconds else None
    progress = 0.0
    while (deadline is None or time.monotonic() < deadline) and len(rows) < args.calibration_min_frames:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic() - started
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose.detect_for_video(image, int(now * 1000))
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        feats = pose_features(landmarks) if landmarks else None
        if plausible_body_frame(feats, args.min_visibility) and feats["side_width_ratio"] <= args.max_side_width:
            rows.append(feats)
        if landmarks:
            draw_pose(frame, landmarks, args.min_visibility)
        progress = min(1.0, len(rows) / max(args.calibration_min_frames, 1))
        draw_scan_progress(frame, progress, len(rows), args.calibration_min_frames)
        draw_text(
            frame,
            [
                "Body scan: stand tall, side-on, arms visible",
                "Press q to cancel scan",
            ],
            (70, 220, 255),
        )
        cv2.imshow("Codex Live Squat Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if len(rows) < args.calibration_min_frames:
        raise RuntimeError("Body scan canceled: stand fully in frame, with wrist/ankle visible, and turn side-on.")
    if progress >= 1.0:
        show_ready_screen(cap, pose, args, started)
    return body_scan_from_feature_rows(rows, "webcam calibration")


def load_rep_model(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def rep_cue(frames, targets):
    if not frames:
        return "No tracked rep frames"
    metrics = summarize_rep_frames(frames)
    scored = score_against_targets(metrics, targets)
    cues = list(scored["all_cues"])
    if metrics["n_frames"] < 6:
        cues.append("too few tracked frames")
    return "; ".join(cues) if cues else "basic angles OK"


def live_cue(feats, detector, targets):
    if feats is None:
        return "Step fully into frame"
    if detector.state != "down":
        return "Start squat; score appears after you stand"
    ranges = targets["ranges"]
    depth_high = ranges["knee_flexion_min"]["high"]
    trunk_high = ranges["trunk_lean_max"]["high"]
    if depth_high is not None and feats["knee_flexion"] > depth_high:
        return "Go lower"
    if trunk_high is not None and feats["trunk_lean"] > trunk_high:
        return "Torso lean above target"
    return "Depth OK"


def score_rep(artifact, frames, rep_id, targets):
    cue = rep_cue(frames, targets)
    if not frames:
        return {"rep_id": rep_id, "prediction": "unknown", "prob_correct": None, "offness": None, "score": None, "message": "Last rep: unknown", "cue": cue}

    df = pd.DataFrame(frames)
    rep_df = aggregate_reps(df, artifact["active_knee"])
    if rep_df.empty:
        return {"rep_id": rep_id, "prediction": "unknown", "prob_correct": None, "offness": None, "score": None, "message": f"Last rep {rep_id}: unknown", "cue": cue}

    x = rep_df[artifact["feature_columns"]].to_numpy("float32")
    model = artifact["model"]
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        prob = float(model.predict_proba(x)[0][classes.index(1)])
    else:
        prob = float(model.predict(x)[0])
    pred = "correct" if prob >= 0.65 else "off" if prob <= 0.35 else "uncertain"
    model_label = "LIKELY CORRECT" if pred == "correct" else "LIKELY OFF" if pred == "off" else "UNCERTAIN"
    metrics = summarize_rep_frames(df)
    target_score = score_against_targets(metrics, targets, prob)
    label = overall_label(target_score["score"], prob)
    return {
        "rep_id": rep_id,
        "prediction": pred,
        "overall": label,
        "prob_correct": prob,
        "offness": 1.0 - prob,
        "score": target_score["score"],
        "metrics": {k: clean_number(v) for k, v in metrics.items()},
        "message": f"Last rep {rep_id}: score {target_score['score']}/100, {label} ({prob * 100:.0f}% model: {model_label})",
        "cue": target_score["main_cue"] if target_score["main_cue"] else cue,
        "target_feedback": target_score,
    }


def analyze_csv(args):
    df = pd.read_csv(args.analyze_csv)
    artifact = load_rep_model(args.model)
    body_scan = body_scan_from_feature_rows(df, f"csv:{args.analyze_csv}")
    targets = build_reference_targets(args.reference_features, body_scan)
    if "quality_ok" in df:
        df = df[pd.to_numeric(df["quality_ok"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "side_width_ratio" in df:
        side_width = pd.to_numeric(df["side_width_ratio"], errors="coerce")
        df = df[side_width <= args.max_side_width].copy()

    reps = []
    current_rep = []
    detector = RepDetector(args.squat_angle, args.stand_angle, args.count_depth_angle, args.min_rep_seconds, args.max_rep_seconds)
    source_id = Path(args.analyze_csv).stem
    for _, row in df.iterrows():
        knee = pd.to_numeric(row.get("knee_flexion"), errors="coerce")
        now = pd.to_numeric(row.get("time_s"), errors="coerce")
        if not np.isfinite(knee) or not np.isfinite(now):
            continue
        event = detector.update(float(knee), float(now))
        if event == "rep_start":
            current_rep = []
        if detector.state == "down":
            item = row.to_dict()
            item.update({"subject_id": "csv", "source_id": source_id, "rep_id": detector.rep_id + 1, "label": 0})
            item["femur_tibia_ratio"] = body_scan.femur_tibia_ratio
            item["tibia_torso_ratio"] = body_scan.tibia_torso_ratio
            current_rep.append(item)
        if event == "rep_end":
            reps.append(score_rep(artifact, current_rep, detector.rep_id, targets))
            current_rep = []
        elif event == "rep_cancel":
            current_rep = []

    report = {
        "body_scan": body_scan.to_dict(),
        "target_ranges": targets,
        "reps": reps,
    }
    out = Path(args.analysis_out) if args.analysis_out else Path(args.analyze_csv).with_suffix(".analysis.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved analysis: {out}")


def live_report(label, body_scan, targets, reps):
    return {
        "label": label,
        "body_scan": body_scan.to_dict(),
        "target_ranges": targets,
        "reps": reps,
    }


def write_live_report(path, label, body_scan, targets, reps):
    path.write_text(json.dumps(live_report(label, body_scan, targets, reps), indent=2), encoding="utf-8")


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    frame_path = out_dir / f"{stamp}_frames.csv"
    reps_path = out_dir / f"{stamp}_reps.json"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    artifact = load_rep_model(args.model)
    pose_model = ensure_pose_model(args.pose_model)
    detector = RepDetector(args.squat_angle, args.stand_angle, args.count_depth_angle, args.min_rep_seconds, args.max_rep_seconds)
    reps = []
    current_rep = []
    last_score = {"message": "No scored rep yet", "cue": "", "prob_correct": None, "score": None}
    fields = [
        "time_s",
        "label",
        "quality_ok",
        "quality",
        "rep_id",
        "event",
        "side",
        "knee_flexion",
        "hip_flexion",
        "trunk_lean",
        "ankle_dorsiflexion",
        "femur_tibia_ratio",
        "tibia_torso_ratio",
        "min_visibility",
        "scan_visibility",
        "track_visibility",
        "side_width_ratio",
        "prediction",
        "prob_correct",
        "offness",
        "score",
        "cue",
    ]

    with frame_path.open("w", newline="", encoding="utf-8") as f, create_landmarker(pose_model) as pose:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        started = time.monotonic()
        body_scan = calibrate_body(cap, pose, args, started)
        targets = build_reference_targets(args.reference_features, body_scan)
        print(json.dumps({"body_scan": body_scan.to_dict(), "target_ranges": targets}, indent=2), flush=True)
        write_live_report(reps_path, args.label, body_scan, targets, reps)
        print(f"Live report: {reps_path}", flush=True)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic() - started
                if args.max_seconds and now >= args.max_seconds:
                    break

                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = pose.detect_for_video(image, int(now * 1000))
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                feats = pose_features(landmarks) if landmarks else None
                feats = apply_body_scan(feats, body_scan)
                quality, quality_ok = quality_message(feats, args.track_min_visibility, args.max_side_width)
                event = detector.update(feats["knee_flexion"], now) if quality_ok else ""
                if event == "rep_start":
                    current_rep = []
                if quality_ok and detector.state == "down":
                    current_rep.append(
                        {
                            "subject_id": "live",
                            "source_id": stamp,
                            "rep_id": detector.rep_id + 1,
                            "label": 0,
                            "time_s": now,
                            **{k: feats[k] for k in artifact["base_features"]},
                        }
                    )
                if event == "rep_end":
                    last_score = score_rep(artifact, current_rep, detector.rep_id, targets)
                    reps.append({"time_s": now, "label": args.label, **last_score})
                    write_live_report(reps_path, args.label, body_scan, targets, reps)
                    print(json.dumps({"latest_rep": reps[-1]}, indent=2), flush=True)
                    current_rep = []
                elif event == "rep_cancel":
                    current_rep = []

                row = {k: "" for k in fields}
                row.update({"time_s": round(now, 3), "label": args.label, "quality_ok": int(quality_ok), "quality": quality})
                row["rep_id"] = detector.rep_id
                row["event"] = event
                row["prediction"] = last_score.get("prediction", "")
                row["cue"] = live_cue(feats, detector, targets) if quality_ok else quality
                if last_score.get("prob_correct") is not None:
                    row["prob_correct"] = round(last_score["prob_correct"], 4)
                    row["offness"] = round(last_score["offness"], 4)
                if last_score.get("score") is not None:
                    row["score"] = last_score["score"]
                if feats:
                    row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items() if k in row})
                writer.writerow(row)

                if landmarks:
                    draw_pose(frame, landmarks, args.min_visibility)
                color = (0, 180, 0) if quality_ok else (0, 80, 220)
                knee = feats["knee_flexion"] if feats else float("nan")
                draw_text(
                    frame,
                    [
                        f"{quality} | reps: {detector.rep_id} | knee: {knee:.0f} | {body_scan.body_type}",
                        live_cue(feats, detector, targets) if quality_ok else row["cue"],
                        last_score["message"],
                        last_score.get("cue", ""),
                        "Press q to quit, r to reset reps. No video is saved.",
                    ],
                    color,
                )
                cv2.imshow("Codex Live Squat Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    detector = RepDetector(args.squat_angle, args.stand_angle, args.count_depth_angle, args.min_rep_seconds, args.max_rep_seconds)
                    reps.clear()
                    current_rep.clear()
                    last_score = {"message": "No scored rep yet", "cue": "", "prob_correct": None, "score": None}
                    write_live_report(reps_path, args.label, body_scan, targets, reps)
        except KeyboardInterrupt:
            print("Stopped by keyboard interrupt; saving current report.", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    write_live_report(reps_path, args.label, body_scan, targets, reps)
    print(f"Saved frames: {frame_path}")
    print(f"Saved reps:   {reps_path}")


def self_test():
    det = RepDetector(squat_angle=165, stand_angle=170, count_depth_angle=155, min_duration=1.0, max_duration=8.0)
    events = [det.update(a, i * 0.2) for i, a in enumerate([180, 168, 150, 130, 150, 155, 160, 176, 180, 180])]
    assert events == ["", "", "rep_start", "", "", "", "", "", "rep_end", ""]
    assert det.rep_id == 1
    det = RepDetector(squat_angle=165, stand_angle=170, count_depth_angle=155, min_duration=1.0, max_duration=8.0)
    events = [det.update(a, i * 0.2) for i, a in enumerate([180, 140, 170, 180, 180])]
    assert "rep_cancel" in events
    assert det.rep_id == 0
    assert round(angle(np.array([0, 0]), np.array([1, 0]), np.array([1, 1]))) == 90
    body = BodyScan(
        source="self-test",
        frames=10,
        femur=1.1,
        tibia=1.0,
        torso=1.5,
        upper_arm=0.4,
        forearm=0.4,
        arm=0.8,
        leg=2.1,
        femur_tibia_ratio=1.1,
        tibia_torso_ratio=0.67,
        arm_leg_ratio=0.38,
        body_type=body_type(1.1, 0.67),
    )
    targets = build_reference_targets(REFERENCE_FEATURES, body)
    assert targets["matched_correct_reps"] > 0
    metrics = {
        "n_frames": 20,
        "duration_s": np.mean(list(targets["ranges"]["duration_s"].values())),
        "knee_flexion_min": np.mean(list(targets["ranges"]["knee_flexion_min"].values())),
        "hip_flexion_min": np.mean(list(targets["ranges"]["hip_flexion_min"].values())),
        "trunk_lean_max": np.mean(list(targets["ranges"]["trunk_lean_max"].values())),
        "ankle_dorsiflexion_min": np.mean(list(targets["ranges"]["ankle_dorsiflexion_min"].values())),
    }
    scored = score_against_targets(metrics, targets, 0.8)
    assert 0 <= scored["score"] <= 100
    assert quality_message({"knee_flexion": 150, "hip_flexion": 100, "trunk_lean": 40, "ankle_dorsiflexion": 150, "track_visibility": 0.2, "side_width_ratio": 0.2}, 0.15, 0.4) == ("Pose OK", True)
    assert quality_message({"knee_flexion": 150, "hip_flexion": 100, "trunk_lean": 40, "ankle_dorsiflexion": 150, "track_visibility": 0.1, "side_width_ratio": 0.2}, 0.15, 0.4)[1] is False
    print("self-test ok")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--model", default=str(ARTIFACT_DIR / "rehab24_rep_model.pkl"))
    parser.add_argument("--pose-model", default=str(ARTIFACT_DIR / "pose_landmarker_full.task"))
    parser.add_argument("--reference-features", default=str(REFERENCE_FEATURES))
    parser.add_argument("--analyze-csv", default="")
    parser.add_argument("--analysis-out", default="")
    parser.add_argument("--label", choices=["unknown", "correct", "incorrect"], default="unknown")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--track-min-visibility", type=float, default=0.15)
    parser.add_argument("--squat-angle", type=float, default=165)
    parser.add_argument("--stand-angle", type=float, default=170)
    parser.add_argument("--count-depth-angle", type=float, default=155)
    parser.add_argument("--calibration-seconds", type=float, default=0)
    parser.add_argument("--calibration-min-frames", type=int, default=20)
    parser.add_argument("--min-rep-seconds", type=float, default=1.0)
    parser.add_argument("--max-rep-seconds", type=float, default=8.0)
    parser.add_argument("--max-side-width", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    elif parsed.analyze_csv:
        analyze_csv(parsed)
    else:
        run(parsed)
