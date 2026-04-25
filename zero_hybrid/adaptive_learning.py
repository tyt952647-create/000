"""
ADAPTIVE LEARNING SYSTEM - 6 Evolutionary Modules
Each module independently learns and adapts game-specific parameters
"""

import json
import numpy as np
import time
from collections import defaultdict, deque
from pathlib import Path

# ===========================
# 1. ADAPTIVE MASK LEARNER
# ===========================
class AdaptiveMaskLearner:
    """Learn motion detection parameters per game"""
    
    def __init__(self, storage_path='mask_profiles.json'):
        self.storage_path = storage_path
        self.game_profiles = defaultdict(lambda: {
            'threshold': 15,
            'adapt_rate': 5,
            'kernel_size': 5,
            'oscillation_period': 30,
            'last_updated': time.time()
        })
        self.load()
        
    def identify_game_by_motion_profile(self, motion_pixels: int, fps: float) -> str:
        """Fingerprint current game by motion characteristics"""
        return f"game_{int(motion_pixels)}_{int(fps*10)}"
    
    def learn_threshold(self, game_id: str, success_rate: float):
        """Adjust threshold based on detection success"""
        profile = self.game_profiles[game_id]
        if success_rate < 0.6:
            profile['threshold'] = max(5, profile['threshold'] - 1)
        elif success_rate > 0.9:
            profile['threshold'] = min(50, profile['threshold'] + 1)
        profile['last_updated'] = time.time()
    
    def learn_adapt_rate(self, game_id: str, brightness_variance: float):
        """Adjust adaptation speed based on lighting variance"""
        profile = self.game_profiles[game_id]
        if brightness_variance > 50:
            profile['adapt_rate'] = min(20, profile['adapt_rate'] + 0.5)
        else:
            profile['adapt_rate'] = max(1, profile['adapt_rate'] - 0.1)
    
    def learn_kernel_size(self, game_id: str, motion_density: float):
        """Adjust morphological kernel based on motion quantity"""
        profile = self.game_profiles[game_id]
        if motion_density > 0.3:
            profile['kernel_size'] = min(11, profile['kernel_size'] + 2)
        else:
            profile['kernel_size'] = max(3, profile['kernel_size'] - 1)
    
    def get_profile(self, game_id: str) -> dict:
        """Get learned parameters for game"""
        return self.game_profiles[game_id]
    
    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump({k: dict(v) for k, v in self.game_profiles.items()}, f)
    
    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.game_profiles = defaultdict(lambda: {
                    'threshold': 15, 'adapt_rate': 5, 'kernel_size': 5,
                    'oscillation_period': 30, 'last_updated': time.time()
                })
                for k, v in data.items():
                    self.game_profiles[k] = v
        except FileNotFoundError:
            pass


# ===========================
# 2. RADAR EVOLUTION SYSTEM
# ===========================
class RadarEvolutionSystem:
    """Adapt radar display modes based on target density"""
    
    def __init__(self):
        self.modes = {
            'dense': {'max_targets': 20, 'radius_scale': 2},
            'sparse': {'max_targets': 8, 'radius_scale': 5},
            'heatmap': {'max_targets': 1000, 'radius_scale': 1},
            'minimal': {'max_targets': 3, 'radius_scale': 10}
        }
        self.current_mode = 'dense'
        self.target_count_history = deque(maxlen=60)
        self.mode_effectiveness = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
    def update_mode_by_density(self, target_count: int):
        """Auto-select radar mode based on current target count"""
        self.target_count_history.append(target_count)
        
        avg_targets = np.mean(self.target_count_history) if self.target_count_history else 0
        
        if avg_targets > 15:
            self.current_mode = 'dense'
        elif avg_targets > 8:
            self.current_mode = 'sparse'
        elif avg_targets > 3:
            self.current_mode = 'heatmap'
        else:
            self.current_mode = 'minimal'
    
    def record_shot_effectiveness(self, hit: bool):
        """Track which mode was most effective for hits"""
        if hit:
            self.mode_effectiveness[self.current_mode]['hits'] += 1
        else:
            self.mode_effectiveness[self.current_mode]['misses'] += 1
    
    def get_best_mode(self) -> str:
        """Get radar mode with best hit ratio"""
        if not self.mode_effectiveness:
            return self.current_mode
        
        best_mode = None
        best_ratio = -1
        
        for mode, stats in self.mode_effectiveness.items():
            total = stats['hits'] + stats['misses']
            if total > 0:
                ratio = stats['hits'] / total
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_mode = mode
        
        return best_mode or self.current_mode
    
    def get_current_config(self) -> dict:
        return self.modes[self.current_mode]


# ===========================
# 3. FISH HEALTH TRACKER
# ===========================
class FishHealthTracker:
    """Learn fish HP patterns and invulnerability cycles"""
    
    def __init__(self):
        self.hp_cycles = defaultdict(lambda: {
            'normal': deque(maxlen=50),
            'invulnerable': deque(maxlen=50),
            'regenerating': deque(maxlen=50)
        })
        self.state_transitions = defaultdict(lambda: defaultdict(int))
        self.damage_events = deque(maxlen=1000)
        
    def record_damage_event(self, fish_class: int, before_area: float, after_area: float, shots: int):
        """Record damage taken"""
        delta = before_area - after_area
        
        event = {
            'class': fish_class,
            'delta': delta,
            'shots': shots,
            'dps': delta / max(shots, 1),
            'timestamp': time.time()
        }
        self.damage_events.append(event)
        
        # Classify state
        if delta > 20:
            self.hp_cycles[fish_class]['normal'].append(delta)
            self.state_transitions[fish_class]['normal'] += 1
        elif delta < 5:
            self.hp_cycles[fish_class]['invulnerable'].append(0)
            self.state_transitions[fish_class]['invulnerable'] += 1
        else:
            self.hp_cycles[fish_class]['regenerating'].append(delta)
            self.state_transitions[fish_class]['regenerating'] += 1
    
    def get_current_state(self, fish_class: int) -> str:
        """Predict current HP state"""
        transitions = self.state_transitions[fish_class]
        if not transitions:
            return 'normal'
        
        state = max(transitions, key=transitions.get)
        return state
    
    def predict_invulnerability_window(self, fish_class: int) -> float:
        """Estimate how long fish is invulnerable after hit"""
        cycles = self.hp_cycles[fish_class]['invulnerable']
        if not cycles:
            return 0.5
        
        return len(cycles) / max(len(self.damage_events), 1)
    
    def get_avg_dps_for_class(self, fish_class: int) -> float:
        """Get average damage per shot"""
        class_events = [e for e in self.damage_events if e['class'] == fish_class]
        if not class_events:
            return 1.0
        
        return np.mean([e['dps'] for e in class_events])


# ===========================
# 4. PLAYER POSITION PREDICTOR
# ===========================
class PlayerPositionPredictor:
    """Learn and predict cannon/player movement"""
    
    def __init__(self, prediction_frames=5):
        self.prediction_frames = prediction_frames
        self.position_history = deque(maxlen=60)
        self.velocity_smooth = deque(maxlen=10)
        self.last_shot_position = None
        
    def update_player_position(self, x: float, y: float):
        """Track player position over time"""
        self.position_history.append((x, y, time.time()))
        
        if len(self.position_history) > 1:
            prev_x, prev_y, prev_t = self.position_history[-2]
            dt = (time.time() - prev_t) + 0.001
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            self.velocity_smooth.append((vx, vy))
    
    def predict_future_position(self, frames_ahead: int = None) -> tuple:
        """Predict player position N frames in future"""
        if frames_ahead is None:
            frames_ahead = self.prediction_frames
        
        if not self.position_history or not self.velocity_smooth:
            return None
        
        curr_x, curr_y, _ = self.position_history[-1]
        avg_vx = np.mean([v[0] for v in self.velocity_smooth])
        avg_vy = np.mean([v[1] for v in self.velocity_smooth])
        
        future_x = curr_x + avg_vx * frames_ahead
        future_y = curr_y + avg_vy * frames_ahead
        
        return (future_x, future_y)
    
    def get_player_velocity(self) -> tuple:
        """Get current player velocity estimate"""
        if not self.velocity_smooth:
            return (0, 0)
        
        return (np.mean([v[0] for v in self.velocity_smooth]),
                np.mean([v[1] for v in self.velocity_smooth]))


# ===========================
# 5. BULLET TRAJECTORY LEARNER
# ===========================
class BulletTrajectoryLearner:
    """Learn bullet physics and accuracy patterns"""
    
    def __init__(self):
        self.trajectory_samples = deque(maxlen=500)
        self.accuracy_by_distance = defaultdict(lambda: {'hits': 0, 'misses': 0})
        self.accuracy_by_angle = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
    def record_shot(self, start_x: float, start_y: float, target_x: float, target_y: float, hit: bool):
        """Record shot trajectory"""
        dist = np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        angle = np.arctan2(target_y - start_y, target_x - start_x) * 180 / np.pi
        
        self.trajectory_samples.append({
            'distance': dist,
            'angle': angle,
            'hit': hit,
            'timestamp': time.time()
        })
        
        # Bin distance
        dist_bin = int(dist / 50) * 50
        # Bin angle
        angle_bin = int(angle / 30) * 30
        
        if hit:
            self.accuracy_by_distance[dist_bin]['hits'] += 1
            self.accuracy_by_angle[angle_bin]['hits'] += 1
        else:
            self.accuracy_by_distance[dist_bin]['misses'] += 1
            self.accuracy_by_angle[angle_bin]['misses'] += 1
    
    def get_accuracy_at_distance(self, distance: float) -> float:
        """Get estimated accuracy at given distance"""
        dist_bin = int(distance / 50) * 50
        stats = self.accuracy_by_distance[dist_bin]
        
        if stats['hits'] + stats['misses'] == 0:
            return 0.5  # Unknown
        
        return stats['hits'] / (stats['hits'] + stats['misses'])
    
    def get_accuracy_at_angle(self, angle: float) -> float:
        """Get estimated accuracy at given angle"""
        angle_bin = int(angle / 30) * 30
        stats = self.accuracy_by_angle[angle_bin]
        
        if stats['hits'] + stats['misses'] == 0:
            return 0.5
        
        return stats['hits'] / (stats['hits'] + stats['misses'])
    
    def predict_bullet_point_of_impact(self, start_x: float, start_y: float, 
                                      target_x: float, target_y: float) -> tuple:
        """Predict where bullet will actually go based on learned ballistics"""
        dist = np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        angle = np.arctan2(target_y - start_y, target_x - start_x)
        
        # Get accuracy multiplier
        accuracy = 0.5 * (self.get_accuracy_at_distance(dist) + self.get_accuracy_at_angle(angle * 180 / np.pi))
        
        # Add slight randomness based on accuracy
        deviation = (1 - accuracy) * 10
        
        impact_x = target_x + np.random.normal(0, deviation)
        impact_y = target_y + np.random.normal(0, deviation)
        
        return (impact_x, impact_y)


# ===========================
# 6. SCORE PATTERN ANALYZER
# ===========================
class ScorePatternAnalyzer:
    """Learn score patterns and predict volatility"""
    
    def __init__(self):
        self.score_history = deque(maxlen=300)
        self.wave_patterns = deque(maxlen=50)
        self.safe_reserve_estimates = deque(maxlen=100)
        
    def update_score(self, score: float):
        """Track score changes"""
        self.score_history.append((score, time.time()))
    
    def detect_wave_pattern(self) -> dict:
        """Detect if score follows wave/cycle pattern"""
        if len(self.score_history) < 20:
            return {'detected': False}
        
        scores = [s[0] for s in self.score_history]
        diffs = np.diff(scores)
        
        # FFT to find periodic patterns
        if len(diffs) > 10:
            fft = np.fft.fft(diffs)
            freqs = np.abs(fft)
            
            # Find dominant frequency
            dominant_freq = np.argmax(freqs[1:]) + 1
            
            if dominant_freq < len(freqs) / 10:  # Not just noise
                return {
                    'detected': True,
                    'period': dominant_freq,
                    'amplitude': np.std(diffs)
                }
        
        return {'detected': False}
    
    def calculate_safe_reserve(self, current_score: float) -> float:
        """Calculate safe reserve based on volatility"""
        if len(self.score_history) < 10:
            return 5
        
        recent_scores = [s[0] for s in list(self.score_history)[-50:]]
        volatility = np.std(recent_scores)
        
        # Safe reserve = 3x standard deviation
        safe = max(3, volatility * 3)
        self.safe_reserve_estimates.append(safe)
        
        return safe
    
    def predict_score_trend(self) -> str:
        """Predict if score is trending up, down, or stable"""
        if len(self.score_history) < 30:
            return 'unknown'
        
        old_avg = np.mean([s[0] for s in list(self.score_history)[:15]])
        new_avg = np.mean([s[0] for s in list(self.score_history)[-15:]])
        
        if new_avg > old_avg * 1.1:
            return 'increasing'
        elif new_avg < old_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'


# ===========================
# INTEGRATED ADAPTIVE SYSTEM
# ===========================
class AdaptiveLearningSystem:
    """Master system controlling all 6 learning modules"""
    
    def __init__(self):
        self.mask_learner = AdaptiveMaskLearner()
        self.radar_evolution = RadarEvolutionSystem()
        self.health_tracker = FishHealthTracker()
        self.player_predictor = PlayerPositionPredictor()
        self.bullet_learner = BulletTrajectoryLearner()
        self.score_analyzer = ScorePatternAnalyzer()
        
    def update_all_systems(self, frame_data: dict):
        """Update all learning modules each frame"""
        
        # Update mask learner
        if 'fish_classes' in frame_data:
            game_id = self.mask_learner.identify_game_by_motion_profile(
                frame_data.get('motion_pixels', 0),
                frame_data.get('fps', 30)
            )
            
            # Learn from success
            if 'detection_success' in frame_data:
                self.mask_learner.learn_threshold(game_id, frame_data['detection_success'])
        
        # Update radar
        if 'fish_count' in frame_data:
            self.radar_evolution.update_mode_by_density(frame_data['fish_count'])
        
        # Update score analyzer
        if 'score' in frame_data:
            self.score_analyzer.update_score(frame_data['score'])
    
    def get_all_metrics(self) -> dict:
        """Get summary of all learned systems"""
        return {
            'mask_profiles': len(self.mask_learner.game_profiles),
            'radar_mode': self.radar_evolution.current_mode,
            'health_states': len(self.health_tracker.hp_cycles),
            'bullet_samples': len(self.bullet_learner.trajectory_samples),
            'score_trend': self.score_analyzer.predict_score_trend(),
            'safe_reserve': self.score_analyzer.calculate_safe_reserve(0)
        }
