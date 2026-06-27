# Technical Implementation Spec

## Datasets

### EC3D (primary)
- **Source**: https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises
- **3D data**: `data_3D.pickle` — shape `(29789, 3, 25)` → frames × (x,y,z) × joints
- **Skeleton**: 25-joint format (verify exact joint indices from repo README before mapping)
- **Labels**: 11 instruction labels per exercise (1 correct + 10 incorrect variations for squat)
- **Video**: 4 GoPro cameras in a ring around subject — use the camera closest to 90° side view
- **Subjects**: 4 per exercise type
- **Exercises used**: squat only (first pass)

### REHAB24-6 (secondary)
- **Source**: https://zenodo.org/records/13305826
- **3D data**: CSV files — 26 skeleton joints in 3D and 2D, 41 MoCap markers
- **Labels**: binary (correct / incorrect)
- **Video**: 2 synchronized RGB cameras (horizontal + vertical) — use the horizontal camera if it is the side view; verify against dataset documentation
- **Subjects**: 10
- **Exercises used**: squat only (first pass)

---

## Data Pipeline

### Step 1 — Camera selection
- EC3D: load all 4 camera angles, compute mean x-displacement of hip joints across frames. The camera with lowest lateral spread is closest to side view. Use that camera's video.
- REHAB24-6: check dataset documentation for camera orientation labels. Use the camera whose horizontal axis aligns with the sagittal plane.

### Step 2 — Run MediaPipe on side-view video
```
mediapipe.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```
- Extract `pose_landmarks` → 33 landmarks, each with (x, y, z, visibility)
- x, y are normalized to [0,1] by frame dimensions
- z is relative depth (unreliable — do not use z for angle computation)
- Store as `(N_frames, 33, 4)` array per video

### Step 3 — Temporal alignment
- EC3D `data_3D.pickle` is pre-extracted frame-by-frame. Align with video frames by index.
- REHAB24-6 CSV has frame-level annotations. Join on frame index.
- Drop frames where MediaPipe visibility < 0.5 on any of: hips (23,24), knees (25,26), ankles (27,28), shoulders (11,12)

### Step 4 — Derive segment lengths from 3D ground truth
For each subject, compute across all frames and take the median (robust to outliers):

```
femur_length    = median(||hip_joint - knee_joint||)       # per side, average left+right
tibia_length    = median(||knee_joint - ankle_joint||)     # per side, average left+right
torso_length    = median(||hip_midpoint - shoulder_midpoint||)
```

Joint indices to use:
- EC3D: verify from dataset — expected Kinect-style skeleton (SpineBase=0, HipLeft=12, KneeLeft=13, AnkleLeft=14, HipRight=16, KneeRight=17, AnkleRight=18, ShoulderLeft=4, ShoulderRight=8)
- REHAB24-6: verify from dataset CSV column headers

Store per-subject as `{subject_id: {femur, tibia, torso}}`.

### Step 5 — Normalize MediaPipe landmarks
Scale pixel/normalized coordinates by segment lengths so the model is body-size invariant.

```
scale = femur_length + tibia_length + torso_length  # total height proxy
landmark_normalized = (landmark_px - hip_midpoint_px) / scale
```

Hip midpoint = mean of landmarks 23 and 24. Centering on hip anchors the pose.

### Step 6 — Compute joint angles (sagittal plane only)
Use 2D (x, y) from MediaPipe — do not use z. For a sideways-facing user, the visible side joints carry the sagittal signal.

Detect which side faces camera: if left hip (23) x-coordinate > right hip (24) x-coordinate, the user faces right → left side joints are visible. Otherwise use right side joints.

Compute angles using the law of cosines on three joint positions:

```
angle(A, B, C) = arccos( dot(BA, BC) / (||BA|| * ||BC||) )
```

**Angles to compute per frame:**

| Feature | Joints (MediaPipe indices) |
|---|---|
| knee_flexion | ankle(27/28) → knee(25/26) → hip(23/24) |
| hip_flexion | knee(25/26) → hip(23/24) → shoulder(11/12) |
| trunk_lean | hip_midpoint → shoulder_midpoint, angle from vertical |
| ankle_dorsiflexion | heel(29/30) → ankle(27/28) → knee(25/26) |

Use left or right indices depending on visible side detected above.

### Step 7 — Build feature vector per frame

```
features = [
    knee_flexion,           # degrees
    hip_flexion,            # degrees
    trunk_lean,             # degrees from vertical
    ankle_dorsiflexion,     # degrees
    femur_tibia_ratio,      # femur / tibia (personalization context)
    tibia_torso_ratio,      # tibia / torso (personalization context)
]
```

Shape: `(N_frames, 6)`

### Step 8 — Label mapping

EC3D labels → binary:
- Label index 0 (correct squat) → `1`
- All other label indices → `0`
- Verify label index meanings from EC3D repo before applying this mapping

REHAB24-6 labels: already binary, use directly.

### Step 9 — Combine datasets

```python
df = pd.concat([ec3d_df, rehab_df], ignore_index=True)
# columns: subject_id, dataset, frame_id, knee_flexion, hip_flexion,
#          trunk_lean, ankle_dorsiflexion, femur_tibia_ratio, tibia_torso_ratio, label
```

Train/test split **by subject**, not by frame. Subjects in test set must not appear in train set. This prevents data leakage from the same person's body proportions.

---

## Model

### Input
`(6,)` vector per frame — the feature vector from Step 7.

### Output
Binary: `1` (correct) / `0` (incorrect)

### Architecture — MLP baseline
```
Input(6) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid
```

### Loss
`BCELoss`. Class weights to handle imbalance (EC3D has 1 correct label vs 10 incorrect).

### Augmentation
- Mirror: negate x-coordinates of all landmarks, swap left/right indices, recompute angles
- Noise: add Gaussian noise σ=0.01 to normalized landmark positions before angle computation (simulates MediaPipe jitter)

### Evaluation metrics
- F1 score (primary — dataset is imbalanced)
- Per-exercise accuracy broken down by correction type (EC3D's 11 labels)

---

## Inference (real-time)

### Calibration phase (one-time per user)
1. Ask user to stand in T-pose for 3 seconds
2. Run MediaPipe, derive segment lengths from visible landmarks:
   - Femur proxy: hip(23/24) → knee(25/26) distance in normalized coords
   - Tibia proxy: knee(25/26) → ankle(27/28)
   - Torso proxy: hip_midpoint → shoulder_midpoint
3. Store `{femur, tibia, torso}` for this session

### Exercise evaluation phase
1. Run MediaPipe at 30fps on webcam feed
2. Apply same normalization + angle computation as training pipeline
3. Feed feature vector to model → get prediction per frame
4. Smooth predictions over a rolling 5-frame window to reduce flicker

---

## Known Gaps
- **14 total subjects** across both datasets — sufficient for prototype, not for generalization claims
- EC3D joint index format needs verification from repo before writing the mapping code
- REHAB24-6 camera orientation needs verification from dataset documentation
- MediaPipe z-coordinate unused — trunk lean computed from 2D only, which is reliable from true side view but degrades if user is not fully sideways
