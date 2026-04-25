# Hybrid Fish Detection System v2.0

**Advanced motion-aware fish detection with adaptive environment tuning**

## ✨ Features

### Core Capabilities
- **Chrome Window Capture** - Auto-detects Chrome window, captures right-half only
- **Adaptive to Fullscreen** - Dynamically adjusts when window is resized or fullscreened
- **Motion-Only Filtering** - Ignores static background and oscillating overlays
- **Single Dot Per Fish** - Intelligently merges fragments with same trajectory
- **Trajectory Clustering** - Recognizes fish by velocity, heading, and class
- **Class Recognition** - Remembers fish by size, speed, and shape (5 classes)

### Adaptive Tuning
- **Real-time Brightness Adjustment** - Threshold adapts to lighting changes
- **Kernel Size Auto-tuning** - Morphological kernel adjusts based on motion
- **Environment Learning** - Calibrates to any game/application on first run
- **No Manual Calibration Needed** - Works on any content with motion

### Motion Analysis
- **Background Rejection** - Only tracks pixels that move
- **Oscillating Pattern Filter** - Removes non-moving UI overlays
- **Motion Threshold** - Configurable minimum motion intensity
- **Frame Differencing** - Detects motion between consecutive frames

## 🚀 Quick Start

```bash
cd zero_hybrid
pip install -r requirements.txt
python main.py