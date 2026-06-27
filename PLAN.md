# Personalized Exercise Form Correction — Project Plan

## Goal
Train a model that evaluates exercise form relative to the individual user's body, not a fixed universal template.

## Stack
- **Training data**: EC3D (primary) + REHAB24-6 (secondary)
- **Inference backbone**: MediaPipe Pose (33 landmarks)
- **Camera**: Mac built-in, user oriented sideways (sagittal view)

## Target Exercises
1. Squat (first)
2. Deadlift (second — shares hip hinge, features transfer)

## Pipeline

### Training
1. Identify side-view camera video in each dataset
2. Run MediaPipe on side-view RGB video → 33 landmarks per frame
3. Derive per-subject segment lengths (femur, tibia, torso) from 3D ground truth joints
4. Normalize MediaPipe landmarks by segment lengths
5. Compute joint angles from normalized landmarks
6. Train model on normalized angles + segment ratios → form label

### Personalization
Normalize joint angles by per-subject segment length ratios before the model sees them.
Handles body-type variation (e.g. long femurs change what a correct squat looks like) without per-user retraining.

### Model
- **Input**: normalized joint angles + subject segment ratios
- **Output**: correct / incorrect form (binary first, multi-class corrections later)
- **Architecture**: MLP baseline → GCN if MLP falls short

## Starting Point
Data pipeline — loading both datasets, identifying side-view camera, running MediaPipe, building normalized training features.
