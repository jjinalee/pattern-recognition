import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from squat_form import (
    FEATURES,
    MP_REQUIRED,
    REHAB_JOINTS,
    Calibration,
    features_from_mediapipe,
    segment_lengths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "codex" / "artifacts"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


def side_camera(orientation, include_half_profile=False):
    if orientation == "front":
        return "Camera18-30fps-transposed"
    if orientation == "profile":
        return "Camera17-30fps"
    if include_half_profile:
        return "Camera18-30fps-transposed"
    return None


def build_jobs(rehab_dir, include_half_profile=False):
    root = Path(rehab_dir)
    seg = pd.read_csv(root / "Segmentation.csv", sep=";")
    seg = seg[(seg["exercise_id"] == 6) & (seg["mocap_erroneous"] == 0)].copy()
    jobs = {}
    for _, rep in seg.iterrows():
        camera = side_camera(rep.cam17_orientation, include_half_profile)
        if camera is None:
            continue
        key = (rep.video_id, camera)
        jobs.setdefault(key, []).append(rep)
    return jobs


def resize_for_pose(frame, max_width):
    if not max_width or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    return cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))


def ensure_pose_model(path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe pose model to {path}")
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


def extract(args):
    rehab_dir = Path(args.rehab24)
    pose_model = ensure_pose_model(args.pose_model)
    jobs = build_jobs(rehab_dir, args.include_half_profile)
    rows, landmark_rows, landmarks = [], [], []
    report = {
        "jobs": len(jobs),
        "include_half_profile": bool(args.include_half_profile),
        "min_visibility": args.min_visibility,
        "pose_model": str(pose_model),
        "max_width": args.max_width,
        "videos": {},
    }

    for (video_id, camera), reps in sorted(jobs.items()):
        video_path = rehab_dir / "videos" / "Ex6" / f"{video_id}-{camera}.mp4"
        joints_path = rehab_dir / "3d_joints" / "Ex6" / f"{video_id}-30fps.npy"
        if not video_path.exists() or not joints_path.exists():
            continue

        joints = np.load(joints_path)[:, :, :3].astype("float32")
        lengths = segment_lengths(joints, REHAB_JOINTS)
        calibration = Calibration(lengths["femur"], lengths["tibia"], lengths["torso"])
        frame_meta = {}
        for rep in reps:
            for frame_id in range(int(rep.first_frame), int(rep.last_frame) + 1):
                frame_meta[frame_id] = rep

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        seen = kept = no_pose = low_visibility = 0
        start = time.time()
        max_frame = max(frame_meta)
        frame_id = -1
        with create_landmarker(pose_model) as landmarker:
            while frame_id < max_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                rep = frame_meta.get(frame_id)
                if rep is None:
                    continue
                seen += 1
                frame = resize_for_pose(frame, args.max_width)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_id * 1000 / fps)
                result = landmarker.detect_for_video(image, timestamp_ms)
                if not result.pose_landmarks:
                    no_pose += 1
                    continue
                lm = result.pose_landmarks[0]
                if min(landmark_visibility(lm[i]) for i in MP_REQUIRED) < args.min_visibility:
                    low_visibility += 1
                    continue
                feats = features_from_mediapipe(lm, calibration)
                if not all(np.isfinite(feats[name]) for name in FEATURES):
                    continue

                mp_arr = np.array([[p.x, p.y, p.z, landmark_visibility(p)] for p in lm], dtype="float32")
                landmarks.append(mp_arr)
                landmark_rows.append((video_id, camera, frame_id))
                rows.append(
                    {
                        "subject_id": f"rehab_{int(rep.person_id):03d}",
                        "frame_id": frame_id,
                        "rep_id": int(rep.repetition_number),
                        "label": int(rep.correctness),
                        "source_id": video_id,
                        "dataset": "rehab24_mediapipe",
                        "side_camera": camera,
                        "video_path": str(video_path),
                        **{name: feats[name] for name in FEATURES},
                        "min_visibility": feats["min_visibility"],
                        "side_width_ratio": feats["side_width_ratio"],
                        "side": feats["side"],
                    }
                )
                kept += 1
        cap.release()
        report["videos"][f"{video_id}-{camera}"] = {
            "candidate_frames": seen,
            "kept_frames": kept,
            "no_pose": no_pose,
            "low_visibility": low_visibility,
            "seconds": round(time.time() - start, 2),
        }
        print(f"{video_id}-{camera}: kept {kept}/{seen}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if landmarks:
        video_ids, cameras, frame_ids = zip(*landmark_rows)
        np.savez_compressed(
            args.landmarks,
            landmarks=np.stack(landmarks),
            video_id=np.array(video_ids),
            camera=np.array(cameras),
            frame_id=np.array(frame_ids, dtype="int32"),
        )

    report.update(
        {
            "rows": int(len(df)),
            "subjects": int(df["subject_id"].nunique()) if not df.empty else 0,
            "labels": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()} if not df.empty else {},
            "out": args.out,
            "landmarks": args.landmarks,
        }
    )
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["rows", "subjects", "labels", "out", "landmarks"]}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehab24", default=str(PROJECT_ROOT / "data" / "rehab24"))
    parser.add_argument("--out", default=str(ARTIFACT_DIR / "rehab24_mediapipe_features.csv"))
    parser.add_argument("--landmarks", default=str(ARTIFACT_DIR / "rehab24_mediapipe_landmarks.npz"))
    parser.add_argument("--report", default=str(ARTIFACT_DIR / "rehab24_mediapipe_build.json"))
    parser.add_argument("--pose-model", default=str(ARTIFACT_DIR / "pose_landmarker_full.task"))
    parser.add_argument("--include-half-profile", action="store_true")
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--max-width", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
