"""
Advanced Predictive AI System
- Multi-shot convergence prediction
- Preemptive firing with lead calculation
- Adaptive learning on fish HP cycles
- Corrupted data handling
"""

import numpy as np
import time
from collections import defaultdict, deque
from typing import List, Tuple, Optional, Dict
import json

# ===========================
# CONVERGENCE DETECTION
# ===========================
class ConvergenceDetector:
    """Detect when 2+ smallest class fish are on collision course"""
    
    def __init__(self, lookback_frames=30):
        self.lookback_frames = lookback_frames
        self.fish_trajectories = defaultdict(lambda: deque(maxlen=lookback_frames))
        self.convergence_events = deque(maxlen=100)
        self.collision_threshold = 50  # pixels
        
    def predict_collision(self, fish_list: List, time_horizon_frames: int = 15) -> List[Tuple]:
        """Predict which fish pairs will collide in next N frames"""
        collisions = []
        
        # Only check smallest class fish (class 0)
        small_fish = [f for f in fish_list if f.class_id == 0]
        
        if len(small_fish) < 2:
            return collisions
        
        for i, f1 in enumerate(small_fish):
            for f2 in small_fish[i+1:]:
                # Predict future positions
                for t in range(1, time_horizon_frames + 1):
                    f1_future_x = f1.x + f1.vx * t
                    f1_future_y = f1.y + f1.vy * t
                    
                    f2_future_x = f2.x + f2.vx * t
                    f2_future_y = f2.y + f2.vy * t
                    
                    dist = np.sqrt((f1_future_x - f2_future_x)**2 + (f1_future_y - f2_future_y)**2)
                    
                    if dist < self.collision_threshold:
                        convergence_point = (
                            int((f1_future_x + f2_future_x) / 2),
                            int((f1_future_y + f2_future_y) / 2)
                        )
                        collisions.append({
                            'fish_ids': (f1.id, f2.id),
                            'collision_frame': t,
                            'collision_point': convergence_point,
                            'confidence': max(0, 1.0 - (dist / self.collision_threshold))
                        })
                        break
        
        return collisions
    
    def track_trajectory(self, fish_id: int, x: float, y: float, vx: float, vy: float):
        """Track fish trajectory for pattern analysis"""
        self.fish_trajectories[fish_id].append({
            'x': x, 'y': y, 'vx': vx, 'vy': vy,
            'timestamp': time.time()
        })


# ===========================
# SPLASH DAMAGE CALCULATOR
# ===========================
class SplashDamageCalculator:
    """Calculate splash damage radius and optimize multi-target firing"""
    
    def __init__(self, base_splash_radius=40):
        self.base_splash_radius = base_splash_radius
        self.splash_history = deque(maxlen=200)
        self.radius_learning = 0.0  # Will adapt based on hits
        
    def calculate_splash_radius(self) -> float:
        """Adaptive splash radius based on game feedback"""
        if not self.splash_history:
            return self.base_splash_radius
        
        successful_splashes = [s for s in self.splash_history if s['hit']]
        if not successful_splashes:
            return self.base_splash_radius
        
        avg_distance_affected = np.mean([s['distance'] for s in successful_splashes])
        
        # Weighted average: 70% base, 30% learned
        learned_radius = (0.7 * self.base_splash_radius) + (0.3 * avg_distance_affected)
        return learned_radius
    
    def find_splash_optimal_point(self, fish_list: List, num_targets: int = 2) -> Tuple:
        """Find optimal shoot point to hit multiple fish with splash"""
        if not fish_list or len(fish_list) < num_targets:
            return None, 0
        
        # Get smallest N fish
        sorted_fish = sorted(fish_list, key=lambda f: f.area)[:num_targets]
        
        # Centroid of all targets
        cx = np.mean([f.x + f.w // 2 for f in sorted_fish])
        cy = np.mean([f.y + f.h // 2 for f in sorted_fish])
        
        splash_radius = self.calculate_splash_radius()
        
        # Check how many fish are within splash at centroid
        covered_fish = []
        for fish in fish_list:
            fx = fish.x + fish.w // 2
            fy = fish.y + fish.h // 2
            dist = np.sqrt((fx - cx)**2 + (fy - cy)**2)
            if dist <= splash_radius:
                covered_fish.append(fish.id)
        
        return (int(cx), int(cy)), len(covered_fish)
    
    def record_splash(self, x: float, y: float, fish_hit: List[int], success: bool):
        """Record splash event for learning"""
        self.splash_history.append({
            'x': x, 'y': y,
            'fish_hit': fish_hit,
            'distance': np.mean([len(f) for f in fish_hit]) if fish_hit else 0,
            'hit': success,
            'timestamp': time.time()
        })


# ===========================
# MULTI-SHOT STRATEGY
# ===========================
class MultiShotStrategy:
    """
    Manage escalating shot costs for preemptive convergence firing:
    - Shot 1: Base cost (1x)
    - Shot 2: Slight increase (1.1x)
    - Shot 3: Double cost (2x)
    Then reset to base
    """
    
    def __init__(self, base_cost: float = 1.0):
        self.base_cost = base_cost
        self.burst_state = None
        self.current_burst_index = 0
        self.burst_targets = []
        
    def get_shot_cost(self, shot_in_burst: int) -> float:
        """Get cost multiplier for Nth shot in convergence burst"""
        if shot_in_burst == 0:
            return self.base_cost * 1.0
        elif shot_in_burst == 1:
            return self.base_cost * 1.1
        elif shot_in_burst >= 2:
            return self.base_cost * 2.0
        return self.base_cost
    
    def start_convergence_burst(self, collision_event: Dict):
        """Initialize 3-shot burst for convergence"""
        self.burst_state = collision_event
        self.current_burst_index = 0
        self.burst_targets = collision_event['fish_ids']
        
    def get_next_shot_targets(self) -> Tuple[List[int], float]:
        """Get targets and cost for next shot in burst"""
        if not self.burst_state:
            return [], self.base_cost
        
        if self.current_burst_index >= 3:
            self.reset_burst()
            return [], self.base_cost
        
        cost = self.get_shot_cost(self.current_burst_index)
        targets = list(self.burst_targets)
        self.current_burst_index += 1
        
        return targets, cost
    
    def reset_burst(self):
        """Reset burst state after completion"""
        self.burst_state = None
        self.current_burst_index = 0
        self.burst_targets = []


# ===========================
# HP CYCLE LEARNING
# ===========================
class HPCycleLearner:
    """Learn patterns in fish health from kill data"""
    
    def __init__(self):
        self.class_hp_patterns = defaultdict(list)
        self.cycle_detection = defaultdict(lambda: {'up': 0, 'down': 0, 'stable': 0})
        self.confidence_thresholds = defaultdict(lambda: 0.5)
        
    def record_hp_event(self, fish_class: int, shots_fired: int, area_before: float, area_after: float):
        """Record HP change from shot impact"""
        area_delta = area_after - area_before
        
        if area_delta > 0:
            cycle_type = 'up'
        elif area_delta < -5:
            cycle_type = 'down'
        else:
            cycle_type = 'stable'
        
        self.class_hp_patterns[fish_class].append({
            'shots': shots_fired,
            'delta': area_delta,
            'type': cycle_type,
            'timestamp': time.time()
        })
        
        self.cycle_detection[fish_class][cycle_type] += 1
    
    def predict_shots_to_kill(self, fish_class: int, historical_data: List = None) -> Tuple[int, float]:
        """Predict shots needed to kill fish of class"""
        if not self.class_hp_patterns[fish_class]:
            return 3, 0.3  # Conservative default
        
        recent_kills = self.class_hp_patterns[fish_class][-20:]
        shots_list = [k['shots'] for k in recent_kills]
        
        avg_shots = np.mean(shots_list)
        std_shots = np.std(shots_list) if len(shots_list) > 1 else 0
        
        # Confidence based on consistency
        confidence = 1.0 - (std_shots / (avg_shots + 0.1))
        confidence = max(0.1, min(1.0, confidence))
        
        return int(np.ceil(avg_shots)), confidence
    
    def get_class_cycle_pattern(self, fish_class: int) -> str:
        """Get most common HP cycle pattern"""
        cycles = self.cycle_detection[fish_class]
        if not cycles:
            return 'unknown'
        
        return max(cycles, key=cycles.get)


# ===========================
# CORRUPTED DATA HANDLER
# ===========================
class CorruptedDataHandler:
    """Handle and recover from corrupted tracking or detection data"""
    
    def __init__(self, threshold_std_dev: float = 3.0):
        self.threshold_std_dev = threshold_std_dev
        self.anomaly_log = deque(maxlen=500)
        self.data_quality_score = 1.0
        
    def validate_fish_data(self, fish) -> bool:
        """Check if fish data appears corrupted"""
        issues = []
        
        # Check velocity sanity
        speed = np.sqrt(fish.vx**2 + fish.vy**2)
        if speed > 100:  # Too fast for real game (pixels per frame)
            issues.append('impossible_speed')
        
        # Check size sanity
        if fish.area <= 0 or fish.area > 50000:
            issues.append('impossible_area')
        
        # Check position sanity
        if fish.x < -1000 or fish.y < -1000:
            issues.append('off_screen_far')
        
        if issues:
            self.anomaly_log.append({
                'fish_id': fish.id,
                'issues': issues,
                'timestamp': time.time()
            })
            return False
        
        return True
    
    def correct_fish_state(self, fish, previous_state):
        """Attempt to correct corrupted fish state using previous known good state"""
        if not previous_state:
            return fish
        
        # If current position too far from last known, revert
        dist = np.sqrt((fish.x - previous_state['x'])**2 + (fish.y - previous_state['y'])**2)
        if dist > 200:
            fish.x = previous_state['x']
            fish.y = previous_state['y']
            fish.vx = previous_state['vx'] * 0.9  # Dampen old velocity
            fish.vy = previous_state['vy'] * 0.9
        
        return fish
    
    def update_data_quality(self, validation_results: List[bool]):
        """Update overall data quality score"""
        if not validation_results:
            return
        
        valid_count = sum(validation_results)
        new_score = valid_count / len(validation_results)
        
        # Weighted average: 80% new, 20% old
        self.data_quality_score = (0.8 * new_score) + (0.2 * self.data_quality_score)
    
    def get_data_quality(self) -> float:
        """Get current data quality score (0-1)"""
        return self.data_quality_score


# ===========================
# SHIFT-KEY AUTO-FIRE TOGGLE
# ===========================
class AutoFireManager:
    """Manage auto-fire mode with Shift key toggle"""
    
    def __init__(self):
        self.auto_fire_enabled = True
        self.last_toggle_time = time.time()
        self.toggle_cooldown = 0.2  # Debounce
        
    def toggle_auto_fire(self):
        """Toggle auto-fire on/off with debounce"""
        now = time.time()
        if now - self.last_toggle_time < self.toggle_cooldown:
            return
        
        self.auto_fire_enabled = not self.auto_fire_enabled
        self.last_toggle_time = now
        
        state_str = "🔴 ON" if self.auto_fire_enabled else "⚫ OFF"
        print(f"⚙️ Auto-Fire {state_str}")
        
    def should_fire(self, confidence: float = 0.0) -> bool:
        """Check if auto-fire should be active"""
        if not self.auto_fire_enabled:
            return False
        
        # Can still manually fire even if auto disabled
        return True


# ===========================
# INTEGRATED ADVANCED SYSTEM
# ===========================
class AdvancedPredictorAI:
    """Complete advanced prediction system combining all components"""
    
    def __init__(self):
        self.convergence_detector = ConvergenceDetector()
        self.splash_calculator = SplashDamageCalculator()
        self.multi_shot = MultiShotStrategy()
        self.hp_learner = HPCycleLearner()
        self.corruption_handler = CorruptedDataHandler()
        self.autofire_manager = AutoFireManager()
        
    def process_frame(self, fish_list: List, current_score: float) -> Dict:
        """
        Main frame processing: return firing decision with all data
        Returns: {
            'should_fire': bool,
            'targets': list,
            'convergence_points': list,
            'shot_count': int,
            'burst_costs': list,
            'splash_point': tuple,
            'data_quality': float,
            'confidence': float
        }
        """
        
        # Validate all fish data
        validation = [self.corruption_handler.validate_fish_data(f) for f in fish_list]
        self.corruption_handler.update_data_quality(validation)
        
        # Detect convergence events
        collisions = self.convergence_detector.predict_collision(fish_list)
        
        # Find splash optimal point
        splash_point, splash_coverage = self.splash_calculator.find_splash_optimal_point(fish_list, num_targets=2)
        
        # Get auto-fire state
        autofire_ready = self.autofire_manager.should_fire()
        
        result = {
            'should_fire': len(collisions) > 0 and autofire_ready,
            'collisions': collisions,
            'splash_point': splash_point,
            'splash_coverage': splash_coverage,
            'data_quality': self.corruption_handler.get_data_quality(),
            'autofire_enabled': self.autofire_manager.auto_fire_enabled,
            'burst_plan': None
        }
        
        # If convergence detected, plan 3-shot burst
        if collisions and current_score > 10:
            best_collision = max(collisions, key=lambda c: c['confidence'])
            self.multi_shot.start_convergence_burst(best_collision)
            
            burst_costs = [
                self.multi_shot.get_shot_cost(0),
                self.multi_shot.get_shot_cost(1),
                self.multi_shot.get_shot_cost(2)
            ]
            
            result['burst_plan'] = {
                'collision': best_collision,
                'shot_count': 3,
                'costs': burst_costs,
                'total_cost': sum(burst_costs)
            }
        
        return result
