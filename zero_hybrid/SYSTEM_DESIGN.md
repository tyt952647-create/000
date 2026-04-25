# ZERO HYBRID v5.0 - COMPLETE SYSTEM DESIGN

## Architecture Overview

### 🎯 Core Components

#### 1. **Motion Detection Engine**
- **Pure Frame Differencing**: Only moving pixels trigger detection
- **Adaptive Thresholding**: Brightness target 127 (middle gray)
- **Oscillation Filtering**: Removes 0.5-2 sec UI overlay patterns
- **Auto Kernel Sizing**: Adapts morphological kernel based on motion density
- **Status**: Completely independent of static background

#### 2. **Fish Classification (10+ Classes)**
- **Class 0**: Tiny (< 50 px²) - 1-shot kill
- **Class 1**: Very Small (50-150 px²) - 1 shot
- **Class 2**: Small (150-300 px²) - 1-2 shots
- **Class 3**: Medium-Small (300-600 px²) - 2 shots
- **Class 4**: Medium (600-1000 px²) - 3 shots
- **Class 5**: Medium-Large (1000-1800 px²) - 3-4 shots
- **Class 6**: Large (1800-3000 px²) - 4+ shots
- **Class 7**: Very Large (3000-5000 px²) - Heavy
- **Class 8**: Huge (5000-8000 px²) - Very Heavy
- **Class 9**: Massive (>8000 px²) - Boss

#### 3. **Trajectory Clustering**
- **Fragment Merging**: Same heading, speed, distance = same fish
- **Single Dot Per Fish**: One center point per target
- **Velocity Estimation**: Smooth motion vector from history
- **Lead Calculation**: Predicts impact point 3-4 frames ahead

#### 4. **OCR Multiplier Learning**
- **Score Change Detection**: Reads before/after scores
- **Multiplier Calculation**: (Score Delta) / (Shot Cost)
- **Per-Class Learning**: Stores average multiplier for each class
- **Confidence Metric**: 0 at 1 kill, increases to 1.0 at 10+ kills
- **Persistent Storage**: fish_multipliers.json

#### 5. **Bullet Conservation Manager**
- **Shot Tracking**: Records every shot and cost
- **Kill Tracking**: Confirms fish eliminations
- **Efficiency Ratio**: Points Per Shot metric
- **Cost Management**: 1.0 cost per shot (configurable)
- **Per-Class Stats**: Tracks shots/kills by fish class

#### 6. **Priority Selector**
- **Small Fish First**: Classes 0-2 prioritized
- **Expected Value = Base × Multiplier × Confidence**
- **Score Threshold**: Must exceed min_efficiency × shot_cost
- **Mandatory Fire Window**: If 30 seconds pass, fire at highest confidence
- **Aggressiveness Scaling**: Adjusts threshold based on setting

#### 7. **Aggressiveness Control**
- **+ Key**: Increase aggressiveness (lower threshold, more shots)
- **- Key**: Decrease aggressiveness (higher threshold, fewer shots)
- **Range**: 0.0 (conservative) to 1.0 (aggressive)
- **Effect on Min Efficiency**:
  - Conservative: 5.0x minimum (only obvious kills)
  - Balanced: 2.0x minimum
  - Aggressive: 0.5x minimum (opportunistic)

#### 8. **Adaptive Radar Display**
- **Standard Mode**: Color by fish class
- **Heatmap Mode**: Color by speed (red=fast, blue=slow)
- **Class Only**: Size proportional to class
- **Value Only**: Brightness indicates expected value
- **Mode Cycling**: Press 'R' to switch modes

### 📊 Data Flow
