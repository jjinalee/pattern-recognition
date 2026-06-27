# Working Prototype Plan


1. Scan the body from the webcam.
   - Estimate relative femur, tibia, torso, and arm lengths from MediaPipe landmarks.
   - Report ratios like femur/tibia and tibia/torso instead of centimeters, because a normal webcam has no reliable real-world scale.

2. Pick a reference target from the dataset.
   - Use the existing REHAB24 MediaPipe feature file.
   - Look only at reps labeled correct.
   - Find reps with body ratios closest to the scanned user.
   - Turn those reps into target ranges for depth, torso lean, hip angle, ankle angle, and tempo.

3. Watch the user's squat.
   - Detect a rep from knee angle over time.
   - Start tracking when the knee begins bending, then count the rep when the user returns upright.
   - Count shallow reps too, then score them with a "go lower" cue instead of ignoring them.
   - Summarize the actual rep: depth, lean, tempo, and model confidence.

4. Output a clear report.
   - Body scan.
   - Dataset-derived target ranges.
   - Per-rep score.
   - One main cue, such as "go lower", "slow down", or "torso lean is above your target range".

What I am not doing yet:

- No centimeter measurements.
- No claim that the target is universally optimal.
- No full stance-width or toe-angle prescription, because the current side-view setup cannot measure those well.
- No new model architecture. The current rep-level model is enough for a prototype.

Main file for the demo:

```bash
.venv/bin/python codex/live_test.py
```

How to stand for the demo:

- Stand fully in frame.
- Turn side-on before the squat portion starts.
- Do slow reps, about 1-4 seconds each.
- Return fully upright between reps.

Current rep detector defaults:

- Starts watching when knee angle drops below `165`.
- Counts a rep if it reaches at least `155`.
- Finishes the rep when the user stands back up above `170`.
- Ignores very tiny/fast dips shorter than `1.0` second.

Not using for the demo:

- `codex/squat_form.py` is the older frame-by-frame model path. I am leaving it for now because `codex/build_rehab_video_features.py` still imports helper code from it.

Non-camera test/report mode:

```bash
.venv/bin/python codex/live_test.py --analyze-csv codex/live_captures/20260627-165942_frames.csv
```
