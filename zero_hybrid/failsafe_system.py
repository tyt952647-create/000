"""
FAILSAFE & EMERGENCY SYSTEM
- Shift key toggle for auto-fire
- 6 automatic failsafes
- 6 escalating response protocols
- Emergency recovery
"""

import threading
import time
from collections import deque
from pynput import keyboard
import numpy as np

# ===========================
# SHIFT KEY LISTENER
# ===========================
class ShiftKeyListener:
    """Global keyboard listener for Shift key"""
    
    def __init__(self, callback=None):
        self.shift_pressed = False
        self.last_toggle_time = 0
        self.toggle_cooldown = 0.2
        self.callback = callback
        self.listener = None
        self.is_running = False
    
    def on_press(self, key):
        try:
            if key == keyboard.Key.shift:
                now = time.time()
                if now - self.last_toggle_time > self.toggle_cooldown:
                    self.shift_pressed = True
                    self.last_toggle_time = now
                    if self.callback:
                        self.callback()
        except AttributeError:
            pass
    
    def on_release(self, key):
        try:
            if key == keyboard.Key.shift:
                self.shift_pressed = False
        except AttributeError:
            pass
    
    def start(self):
        """Start listening for Shift key"""
        self.is_running = True
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
    
    def stop(self):
        """Stop listening"""
        self.is_running = False
        if self.listener:
            self.listener.stop()


# ===========================
# AUTO-FIRE TOGGLE
# ===========================
class AutoFireToggle:
    """Manage auto-fire on/off state"""
    
    def __init__(self, initial_state=True):
        self.enabled = initial_state
        self.state_history = deque(maxlen=100)
        self.toggle_count = 0
        
    def toggle(self):
        """Toggle auto-fire state"""
        self.enabled = not self.enabled
        self.toggle_count += 1
        self.state_history.append((self.enabled, time.time()))
        
        state_str = "🟢 ON" if self.enabled else "🔴 OFF"
        print(f"⚙️ Auto-Fire {state_str}")
        
        return self.enabled
    
    def force_enable(self):
        """Force enable (for emergency)"""
        self.enabled = True
    
    def force_disable(self):
        """Force disable (for safety)"""
        self.enabled = False
    
    def get_state_string(self) -> str:
        """Get state display string"""
        return "🟢 ON" if self.enabled else "🔴 OFF"


# ===========================
# FAILSAFE MANAGER
# ===========================
class FailsafeManager:
    """Monitor and manage 6 automatic failsafes"""
    
    def __init__(self):
        self.failsafes = {
            'fps_drop': {'threshold': 15, 'triggered': False},
            'data_corruption': {'threshold': 0.7, 'triggered': False},
            'catastrophic_loss': {'threshold': -50, 'triggered': False},
            'pattern_oscillation': {'threshold': 0.9, 'triggered': False},
            'memory_leak': {'threshold': 500, 'triggered': False},
            'score_zero': {'threshold': 0, 'triggered': False}
        }
        self.metric_history = deque(maxlen=300)
        
    def check_fps_drop(self, current_fps: float) -> bool:
        """Check if FPS dropped below threshold"""
        threshold = self.failsafes['fps_drop']['threshold']
        
        if current_fps < threshold:
            self.failsafes['fps_drop']['triggered'] = True
            return True
        else:
            self.failsafes['fps_drop']['triggered'] = False
            return False
    
    def check_data_corruption(self, data_quality: float) -> bool:
        """Check if tracking data is corrupted"""
        threshold = self.failsafes['data_corruption']['threshold']
        
        if data_quality < threshold:
            self.failsafes['data_corruption']['triggered'] = True
            return True
        else:
            self.failsafes['data_corruption']['triggered'] = False
            return False
    
    def check_catastrophic_loss(self, score: float, score_history: deque) -> bool:
        """Check for catastrophic score loss"""
        if not score_history or len(score_history) < 50:
            return False
        
        baseline = np.mean(list(score_history)[:25])
        threshold = self.failsafes['catastrophic_loss']['threshold']
        
        if score < baseline + threshold:
            self.failsafes['catastrophic_loss']['triggered'] = True
            return True
        else:
            self.failsafes['catastrophic_loss']['triggered'] = False
            return False
    
    def check_pattern_oscillation(self, score_history: deque) -> bool:
        """Detect if score is oscillating (stuck pattern)"""
        if len(score_history) < 30:
            return False
        
        recent = list(score_history)[-30:]
        
        # Check if score repeats same values
        unique_values = len(set(recent))
        threshold = 0.2  # If <20% unique, likely oscillating
        
        if unique_values / len(recent) < threshold:
            self.failsafes['pattern_oscillation']['triggered'] = True
            return True
        else:
            self.failsafes['pattern_oscillation']['triggered'] = False
            return False
    
    def check_memory_leak(self, object_counts: dict) -> bool:
        """Check if object tracking is growing unbounded"""
        self.metric_history.append(object_counts)
        
        if len(self.metric_history) < 100:
            return False
        
        old_avg = np.mean([m.get('fishes', 0) for m in list(self.metric_history)[:50]])
        new_avg = np.mean([m.get('fishes', 0) for m in list(self.metric_history)[-50:]])
        
        if new_avg > old_avg * 2:
            self.failsafes['memory_leak']['triggered'] = True
            return True
        else:
            self.failsafes['memory_leak']['triggered'] = False
            return False
    
    def check_score_zero(self, score: float) -> bool:
        """Check if score hit zero (game over)"""
        if score <= 0:
            self.failsafes['score_zero']['triggered'] = True
            return True
        else:
            self.failsafes['score_zero']['triggered'] = False
            return False
    
    def get_active_failsafes(self) -> list:
        """Get list of currently triggered failsafes"""
        return [name for name, state in self.failsafes.items() if state['triggered']]


# ===========================
# CONTINGENCY PROTOCOL
# ===========================
class ContingencyProtocol:
    """6-level escalating response to failsafes"""
    
    def __init__(self):
        self.level = 0  # 0 = normal, 1-6 = escalating
        self.protocols = {
            0: self._protocol_normal,
            1: self._protocol_gradual_reduction,
            2: self._protocol_high_confidence_only,
            3: self._protocol_elite_only,
            4: self._protocol_observe_only,
            5: self._protocol_emergency_mode,
            6: self._protocol_shutdown
        }
    
    def determine_level(self, active_failsafes: list) -> int:
        """Determine contingency level from failsafes"""
        failsafe_count = len(active_failsafes)
        
        if failsafe_count == 0:
            return 0
        elif failsafe_count == 1:
            return 1
        elif failsafe_count == 2:
            return 2
        elif failsafe_count == 3:
            return 3
        elif failsafe_count == 4:
            return 4
        elif failsafe_count >= 5:
            return 6  # Full shutdown
        else:
            return min(failsafe_count, 5)
    
    def _protocol_normal(self) -> dict:
        return {'fire_rate': 1.0, 'confidence_min': 0.0, 'target_classes': [0, 1, 2, 3, 4]}
    
    def _protocol_gradual_reduction(self) -> dict:
        return {'fire_rate': 0.5, 'confidence_min': 0.0, 'target_classes': [0, 1, 2, 3, 4]}
    
    def _protocol_high_confidence_only(self) -> dict:
        return {'fire_rate': 0.3, 'confidence_min': 0.8, 'target_classes': [0, 1, 2]}
    
    def _protocol_elite_only(self) -> dict:
        return {'fire_rate': 0.1, 'confidence_min': 0.95, 'target_classes': [0]}
    
    def _protocol_observe_only(self) -> dict:
        return {'fire_rate': 0.0, 'confidence_min': 1.0, 'target_classes': []}
    
    def _protocol_emergency_mode(self) -> dict:
        return {'fire_rate': 0.5, 'confidence_min': 0.9, 'target_classes': [0], 'cost_limit': 1}
    
    def _protocol_shutdown(self) -> dict:
        return {'fire_rate': 0.0, 'confidence_min': 2.0, 'target_classes': []}
    
    def get_rules(self) -> dict:
        """Get current protocol rules"""
        protocol_func = self.protocols.get(self.level, self._protocol_normal)
        return protocol_func()
    
    def set_level(self, level: int):
        """Manually set contingency level"""
        self.level = max(0, min(6, level))


# ===========================
# PERFORMANCE OPTIMIZER
# ===========================
class PerformanceOptimizer:
    """Auto-adjust system parameters based on FPS"""
    
    def __init__(self):
        self.optimization_level = 0  # 0-5
        self.fps_history = deque(maxlen=60)
        
    def update_fps(self, fps: float):
        """Track FPS over time"""
        self.fps_history.append(fps)
    
    def auto_optimize(self) -> int:
        """Auto-adjust optimization level based on FPS"""
        if len(self.fps_history) < 10:
            return 0
        
        avg_fps = np.mean(self.fps_history)
        
        if avg_fps >= 30:
            level = 0
        elif avg_fps >= 20:
            level = 1
        elif avg_fps >= 15:
            level = 2
        elif avg_fps >= 10:
            level = 3
        elif avg_fps >= 5:
            level = 4
        else:
            level = 5
        
        self.optimization_level = level
        return level
    
    def get_optimization_settings(self) -> dict:
        """Get optimization parameters for current level"""
        settings = {
            0: {'frame_skip': 0, 'resolution_scale': 1.0, 'detector_quality': 'high'},
            1: {'frame_skip': 0, 'resolution_scale': 0.8, 'detector_quality': 'high'},
            2: {'frame_skip': 1, 'resolution_scale': 0.6, 'detector_quality': 'medium'},
            3: {'frame_skip': 2, 'resolution_scale': 0.5, 'detector_quality': 'low'},
            4: {'frame_skip': 3, 'resolution_scale': 0.4, 'detector_quality': 'low'},
            5: {'frame_skip': 5, 'resolution_scale': 0.3, 'detector_quality': 'minimal'}
        }
        
        return settings[self.optimization_level]


# ===========================
# EMERGENCY RECOVERY
# ===========================
class EmergencyRecovery:
    """Handle crashes and attempt recovery"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retry_count = 0
        self.error_log = deque(maxlen=100)
        
    def log_error(self, source: str, error_msg: str):
        """Log error occurrence"""
        self.error_log.append({
            'source': source,
            'error': error_msg,
            'timestamp': time.time()
        })
        print(f"🚨 Error in {source}: {error_msg}")
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from error"""
        self.retry_count += 1
        
        if self.retry_count > self.max_retries:
            print(f"❌ Recovery failed after {self.max_retries} attempts")
            return False
        
        print(f"🔄 Recovery attempt {self.retry_count}/{self.max_retries}")
        time.sleep(1)
        
        return True
    
    def reset_retry_counter(self):
        """Reset retry counter on successful recovery"""
        self.retry_count = 0
    
    def get_error_summary(self) -> dict:
        """Get summary of recent errors"""
        if not self.error_log:
            return {}
        
        sources = {}
        for error in self.error_log:
            source = error['source']
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return sources


# ===========================
# INTEGRATED FAILSAFE SYSTEM
# ===========================
class IntegratedFailsafeSystem:
    """Master failsafe system controlling everything"""
    
    def __init__(self):
        self.shift_listener = ShiftKeyListener(callback=self.on_shift_pressed)
        self.autofire_toggle = AutoFireToggle(initial_state=True)
        self.failsafe_manager = FailsafeManager()
        self.contingency = ContingencyProtocol()
        self.performance_optimizer = PerformanceOptimizer()
        self.emergency_recovery = EmergencyRecovery()
        
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start failsafe system"""
        self.is_running = True
        self.shift_listener.start()
        print("✅ Failsafe system started")
    
    def stop(self):
        """Stop failsafe system"""
        self.is_running = False
        self.shift_listener.stop()
        print("⏹️ Failsafe system stopped")
    
    def on_shift_pressed(self):
        """Callback when Shift is pressed"""
        self.autofire_toggle.toggle()
    
    def update_all_checks(self, data: dict) -> dict:
        """Run all failsafe checks each frame"""
        
        fps = data.get('fps', 30)
        data_quality = data.get('data_quality', 1.0)
        score = data.get('score', 0)
        score_history = data.get('score_history', deque())
        object_counts = data.get('object_counts', {})
        
        # Update performance optimizer
        self.performance_optimizer.update_fps(fps)
        self.performance_optimizer.auto_optimize()
        
        # Run all failsafe checks
        self.failsafe_manager.check_fps_drop(fps)
        self.failsafe_manager.check_data_corruption(data_quality)
        self.failsafe_manager.check_catastrophic_loss(score, score_history)
        self.failsafe_manager.check_pattern_oscillation(score_history)
        self.failsafe_manager.check_memory_leak(object_counts)
        self.failsafe_manager.check_score_zero(score)
        
        # Get active failsafes
        active_failsafes = self.failsafe_manager.get_active_failsafes()
        
        # Determine contingency level
        contingency_level = self.contingency.determine_level(active_failsafes)
        self.contingency.set_level(contingency_level)
        
        # Get contingency rules
        rules = self.contingency.get_rules()
        
        return {
            'safe_to_fire': rules['fire_rate'] > 0,
            'autofire_enabled': self.autofire_toggle.enabled,
            'autofire_state': self.autofire_toggle.get_state_string(),
            'contingency_level': contingency_level,
            'active_failsafes': active_failsafes,
            'firing_rules': rules,
            'optimization_level': self.performance_optimizer.optimization_level,
            'optimization_settings': self.performance_optimizer.get_optimization_settings()
        }
