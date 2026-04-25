# 🎯 ZERO HYBRID v2.0 - Motion-Based Adaptive Fish Detection

**Complete adaptive detection system with trajectory clustering, automatic threshold tuning, and oscillation filtering.**

## ✨ Key Features

### 🎬 **Motion-Only Detection**
- **Pure frame differencing** - Only moving pixels trigger detection
- **Static background completely ignored** - No false positives from environment
- **Oscillation filtering** - Removes UI overlays with 0.5-2 sec frequency
- **Zero stationary targets** - Only motion = valid detection

### 🧠 **Adaptive Auto-Tuning**
- **Brightness target** - Threshold auto-adjusts to find middle gray (127) in motion regions
- **Real-time learning** - Adapts frame-by-frame as environment changes
- **Works on any game** - No calibration needed
- **Kernel auto-sizing** - Consolidation adapts to motion quantity

### 🎯 **Single Fixed Dot Per Fish**
- **Trajectory clustering** - Merges fragments with same heading & speed
- **Distance check** - ≤30px apart = same fish
- **Heading check** - ≤15° angle difference = same fish
- **Speed check** - ≤50% velocity difference = same fish
- **Keeps largest** - Absorbs smaller fragments into main target

### 🐠 **5-Class Fish Recognition**