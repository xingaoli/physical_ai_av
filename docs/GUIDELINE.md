# PhysicalAI AV Dataset - Extended Tools Guide

This document describes the custom tools added to the PhysicalAI Autonomous Vehicles dataset repository for data extraction, meta-action annotation, and visualization.

---

## Overview

The tools in this repository extend the original PhysicalAI AV dataset functionality with:

1. **Global timestamp alignment** - Ensures all cameras use the same 10Hz timeline
2. **Meta-action annotation** - Automatic labeling of longitudinal and lateral driving actions
3. **Keyframe selection** - Intelligent frame selection at action transition points
4. **Visualization** - Tools for visualizing annotations and analyzing statistics

---

## Tool List

### 1. Extract Egomotion (`tools/1_extract_egomotion.py`)

Extracts egomotion data aligned with a global 10Hz timeline.

**Features:**
- Global timestamp alignment (0-20s at 10Hz = 200 frames per video)
- All 7 cameras use exactly the same timestamps
- Output format matches original egomotion structure

**Usage:**
```bash
# Process a single chunk
python3 tools/1_extract_egomotion.py --chunks chunk_0000

# Process multiple chunks
python3 tools/1_extract_egomotion.py --chunks chunk_0000 chunk_0004 chunk_0008
```

**Output:**
- `labels/egomotion_corrected/egomotion.chunk_*.zip`
- Each video: `{uuid}.egomotion.parquet`

---

### 2. Extract Camera Frames (`tools/2_extract_camera_frames.py`)

Extracts video frames from all cameras using the global timeline.

**Features:**
- Extracts frames at global 10Hz timestamps (200 frames per video)
- Supports selective camera extraction
- All cameras perfectly aligned in time

**Usage:**
```bash
# Extract all cameras for a chunk
python3 tools/2_extract_camera_frames.py --chunks chunk_0000

# Extract specific cameras
python3 tools/2_extract_camera_frames.py --chunks chunk_0000 --cameras camera_front_wide_120fov camera_front_tele_30fov

# Extract from all chunks
python3 tools/2_extract_camera_frames.py
```

**Output:**
- `camera_frames/{camera_name}/{chunk_id}/{video_uuid}/`
- Frames named as `frame_{timestamp_ms}.jpg`

---

### 3. Meta-Action Annotation (`tools/3_meta_action_annotation.py`)

Automatically labels each frame with meta-actions and selects keyframes at action transitions.

**Meta-Action Categories:**

| Longitudinal | Lateral |
|--------------|---------|
| Gentle accelerate | Steer left |
| Gentle decelerate | Steer right |
| Maintain speed | Sharp steer left |
| Strong accelerate | Sharp steer right |
| Strong decelerate | Reverse left |
| Stop | Reverse right |
| Reverse | Go straight |

**Configuration:**
Edit `tools/3_meta_action_config.json` (copy from `3_meta_action_config.example.json`):
```json
{
  "strong_accel_threshold": 2.0,
  "gentle_accel_threshold": 0.5,
  "gentle_decel_threshold": -0.5,
  "strong_decel_threshold": -2.0,
  "stop_speed_threshold": 0.5,
  "sharp_steer_threshold": 0.08,
  "gentle_steer_threshold": 0.02,
  "acceleration_window": 5,
  "curvature_window": 5
}
```

**Usage:**
```bash
# Annotate a chunk
python3 tools/3_meta_action_annotation.py --chunks chunk_0000

# Annotate with visualization of first video
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --viz
```

**Output:**
- `labels/meta_actions/meta_actions.{chunk_id}/`
- Each video: `{uuid}.meta_actions.json`

**JSON Structure:**
```json
{
  "video_uuid": "uuid",
  "num_frames": 200,
  "num_keyframes": 50,
  "keyframes": [
    {
      "frame_idx": 0,
      "timestamp_sec": 0.0,
      "speed": 5.2,
      "acceleration": 0.3,
      "yaw_rate": 0.01,
      "long_action": "Gentle accelerate",
      "lat_action": "Go straight"
    }
  ],
  "action_statistics": {
    "longitudinal": {"Gentle accelerate": 44, ...},
    "lateral": {"Go straight": 101, ...}
  },
  "keyframe_indices": {
    "longitudinal": [0, 15, 30, ...],
    "lateral": [0, 10, 25, ...],
    "combined": [0, 10, 15, ...]
  },
  "smooth_data": {
    "timestamp_sec": [0.0, 0.1, 0.2, ...],
    "speed": [5.2, 5.3, 5.1, ...],
    "acceleration": [0.3, 0.2, 0.1, ...],
    "yaw_rate": [0.01, 0.005, 0.002, ...],
    "long_action": ["Gentle accelerate", ...],
    "lat_action": ["Go straight", ...]
  }
}
```

---

### 4. Visualize Meta-Actions (`tools/a4_visualize_meta_actions.py`)

Visualizes meta-action timeline for a single video using the generated JSON data.

**Features:**
- Speed profile plot
- Acceleration with longitudinal action background
- Yaw rate with lateral action background
- Keyframe markers
- Uses `smooth_data` from JSON for high-resolution curves

**Usage:**
```bash
# Visualize a specific video
python3 tools/a4_visualize_meta_actions.py chunk_0000 86de1c0c-e9cd-44ef-aad2-211c6b8a00da

# Save to specific path
python3 tools/a4_visualize_meta_actions.py chunk_0000 <uuid> /path/to/output.png
```

**Output:**
- PNG visualization with 4 subplots showing speed, acceleration, yaw rate, and keyframes

---

### 5. Analyze Meta-Actions (`tools/5_analyze_meta_actions.py`)

Computes statistics across processed chunks.

**Features:**
- Action distribution statistics
- Compression ratio analysis
- Visualization of action frequencies
- Keyframe density metrics

**Usage:**
```bash
# Analyze specific chunks
python3 tools/5_analyze_meta_actions.py --chunks chunk_0000 chunk_0001

# Analyze all processed chunks
python3 tools/5_analyze_meta_actions.py
```

**Output:**
- Terminal statistics
- Optional visualization plots

---

## Workflow Example

Complete pipeline for processing a new chunk:

```bash
# Step 1: Extract egomotion with global timestamps
python3 tools/1_extract_egomotion.py --chunks chunk_0000

# Step 2: Extract camera frames (optional, for visualization)
python3 tools/2_extract_camera_frames.py --chunks chunk_0000

# Step 3: Generate meta-action annotations
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --viz

# Step 4: Visualize a specific video
python3 tools/a4_visualize_meta_actions.py chunk_0000 <video_uuid>

# Step 5: Analyze statistics across chunks
python3 tools/5_analyze_meta_actions.py --chunks chunk_0000
```

---

## Environment Setup

1. Create `.env` file in the project root:
```bash
PHYSICAL_AI_AV_DATA_DIR=/path/to/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar
```

2. Install dependencies:
```bash
pip install physical-ai-av pandas numpy scipy matplotlib tqdm opencv-python
```

---

## Notes

- All tools use a global 10Hz timeline (200 frames per 20-second video)
- Meta-action detection uses smoothed signals with configurable thresholds
- Keyframes are selected at moments when either longitudinal or lateral actions change
- The `smooth_data` field in JSON contains per-frame action labels for all frames
