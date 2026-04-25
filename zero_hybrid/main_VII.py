import cv2
import numpy as np
import mss
import time
import pyautogui
import pytesseract
from collections import deque
from pathlib import Path
import threading
from pynput import keyboard
from datetime import datetime
import os

# =========================
# SCREENSHOT MANAGER
# =========================
class ScreenshotManager:
    """Captures and saves screenshots for analysis"""
    
    def __init__(self, output_dir="screenshots"):
        self.output_dir = output_dir
        self.last_screenshot_time = time.time()
        self.screenshot_interval = 0.5
        Path(output_dir).mkdir(exist_ok=True)
    
    def should_take_screenshot(self):
        """Check if enough time has passed"""
        return time.time() - self.last_screenshot_time >= self.screenshot_interval
    
    def save_screenshot(self, mask, radar, capture, prefix=""):
        """Save all three screenshots with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        mask_path = os.path.join(self.output_dir, f"{timestamp}_mask.png")
        radar_path = os.path.join(self.output_dir, f"{timestamp}_radar.png")
        capture_path = os.path.join(self.output_dir, f"{timestamp}_capture.png")
        
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(radar_path, radar)
        cv2.imwrite(capture_path, capture)
        
        self.last_screenshot_time = time.time()
        
        return {
            'mask': mask_path,
            'radar': radar_path,
            'capture': capture_path,
            'timestamp': timestamp
        }


# =========================
# OVERLAY COMPOSITOR
# =========================
class OverlayCompositor:
    """Overlays mask, radar, and capture for fact-checking"""
    
    @staticmethod
    def create_overlay(mask, radar, capture, stats=None):
        """Create side-by-side overlay of all three captures"""
        h_cap, w_cap = capture.shape[:2]
        h_mask, w_mask = mask.shape[:2]
        h_radar, w_radar = radar.shape[:2]
        
        # Resize to consistent height
        target_height = min(h_cap, h_mask, h_radar, 400)
        
        capture_resized = cv2.resize(capture, (int(w_cap * target_height / h_cap), target_height))
        mask_resized = cv2.resize(mask, (int(w_mask * target_height / h_mask), target_height))
        radar_resized = cv2.resize(radar, (int(w_radar * target_height / h_radar), target_height))
        
        # Ensure all have same height
        h1, w1 = capture_resized.shape[:2]
        h2, w2 = mask_resized.shape[:2]
        h3, w3 = radar_resized.shape[:2]
        
        max_h = max(h1, h2, h3)
        
        # Pad to same height
        capture_resized = cv2.copyMakeBorder(capture_resized, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT)
        mask_resized = cv2.copyMakeBorder(mask_resized, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT)
        radar_resized = cv2.copyMakeBorder(radar_resized, 0, max_h - h3, 0, 0, cv2.BORDER_CONSTANT)
        
        # Horizontal stack
        overlay = np.hstack([capture_resized, mask_resized, radar_resized])
        
        # Add labels
        cv2.putText(overlay, "CAPTURE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, "MASK", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, "RADAR", (w1 + w2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add stats if provided
        if stats:
            y_pos = 60
            for stat in stats:
                cv2.putText(overlay, stat, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos += 25
        
        return overlay


# =========================
# ACCURACY VERIFIER
# =========================
class AccuracyVerifier:
    """Verifies detection accuracy by comparing across frames"""
    
    def __init__(self, history_size=30):
        self.detection_history = deque(maxlen=history_size)
        self.false_positives = 0
        self.true_positives = 0
        self.false_negatives = 0
        self.class_accuracy = {}
    
    def record_detection(self, fish_id, class_id, area, confidence):
        """Record a detection"""
        self.detection_history.append({
            'id': fish_id,
            'class': class_id,
            'area': area,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def verify_persistence(self, current_fishes):
        """Check if detected fish persist across frames"""
        if not self.detection_history:
            return []
        
        current_ids = {f.id for f in current_fishes}
        recent_ids = {d['id'] for d in list(self.detection_history)[-10:]}
        
        verified = current_ids & recent_ids
        
        return list(verified)
    
    def get_class_accuracy(self, class_id):
        """Get accuracy for specific fish class"""
        class_detections = [d for d in self.detection_history if d['class'] == class_id]
        
        if not class_detections:
            return 0.0
        
        avg_confidence = np.mean([d['confidence'] for d in class_detections])
        return avg_confidence
    
    def analyze_mask_quality(self, mask, capture):
        """Analyze mask quality for accuracy improvements"""
        # Count motion pixels
        motion_pixels = np.count_nonzero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
        
        # Ideal motion ratio: 5-30%
        if motion_ratio < 0.05:
            return "TOO CONSERVATIVE: Increase threshold"
        elif motion_ratio > 0.30:
            return "TOO AGGRESSIVE: Decrease threshold"
        else:
            return "OPTIMAL: Mask tuned well"


# =========================
# HOTKEY LISTENER (Global Input Handler)
# =========================
class HotkeyListener:
    """Global hotkey listener using pynput for reliable key detection"""
    
    def __init__(self):
        self.shift_pressed = False
        self.plus_pressed = False
        self.minus_pressed = False
        self.q_pressed = False
        self.shift_toggled = False
        self.listener = None
        self.callbacks = {}
        
    def on_press(self, key):
        """Handle key press"""
        try:
            if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                if not self.shift_pressed:
                    self.shift_pressed = True
                    self.shift_toggled = True
                    if 'shift' in self.callbacks:
                        self.callbacks['shift']()
            
            if key == keyboard.Key.equal or str(key) == "'+'":
                if 'plus' in self.callbacks:
                    self.callbacks['plus']()
            
            if key == keyboard.Key.minus or str(key) == "'-'":
                if 'minus' in self.callbacks:
                    self.callbacks['minus']()
            
            if hasattr(key, 'char'):
                if key.char == 'q' or key.char == 'Q':
                    if 'q' in self.callbacks:
                        self.callbacks['q']()
                        
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release"""
        try:
            if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.shift_pressed = False
        except AttributeError:
            pass
    
    def start(self):
        """Start listening for hotkeys"""
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
    
    def stop(self):
        """Stop listening for hotkeys"""
        if self.listener:
            self.listener.stop()
    
    def register_callback(self, key, callback):
        """Register callback for key event"""
        self.callbacks[key] = callback
    
    def check_shift_toggled(self):
        """Check if shift was toggled and reset"""
        if self.shift_toggled:
            self.shift_toggled = False
            return True
        return False


# =========================
# WINDOW POSITIONING
# =========================
class WindowManager:
    """Manages separate GUI windows with proper positioning"""
    def __init__(self):
        self.windows = {}
        screen_w, screen_h = pyautogui.size()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.fullscreen_windows = {}
        
    def create(self, name, x, y, width=500, height=500):
        """Create and position a window"""
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        cv2.moveWindow(name, x, y)
        self.windows[name] = (x, y, width, height)
        return name
    
    def toggle_fullscreen(self, name):
        """Toggle window to fullscreen"""
        if name not in self.fullscreen_windows:
            self.fullscreen_windows[name] = False
        
        if not self.fullscreen_windows[name]:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.fullscreen_windows[name] = True
        else:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            self.fullscreen_windows[name] = False
    
    def show(self, name, image):
        """Display image in window"""
        if image is not None and image.size > 0:
            cv2.imshow(name, image)


# =========================
# CHROME WINDOW DETECTION
# =========================
def get_chrome_region():
    """Auto-detect Chrome window and get right-half region"""
    try:
        import win32gui
        
        def enum_windows(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and 'chrome' in win32gui.GetWindowText(hwnd).lower():
                results.append(hwnd)
        
        results = []
        win32gui.EnumWindows(enum_windows, results)
        
        if results:
            hwnd = results[0]
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top
            
            return {
                "left": left + width // 2,
                "top": top,
                "width": width // 2,
                "height": height,
                "window": hwnd
            }
    except:
        pass
    
    sct = mss.mss()
    monitor = sct.monitors[1]
    return {
        "left": monitor["width"] // 2,
        "top": 0,
        "width": monitor["width"] // 2,
        "height": monitor["height"],
        "window": None
    }


# =========================
# MOTION DETECTION ENGINE
# =========================
class MotionDetector:
    """Detects only moving pixels, ignores static background and oscillations"""
    
    def __init__(self, width, height, adapt_rate=5):
        self.width = width
        self.height = height
        self.adapt_rate = adapt_rate
        self.prev_frame = None
        self.motion_history = deque(maxlen=60)
        self.base_threshold = 15
        self.target_brightness = 127
        self.current_brightness = 127
        self.oscillation_suppression = 0.0
        
    def update_adapt_rate(self, rate):
        """Update adaptation speed (0-100)"""
        self.adapt_rate = max(0.1, rate)
    
    def detect_motion(self, frame):
        """Frame differencing for motion-only detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray.copy()
        
        _, motion_mask = cv2.threshold(diff, self.base_threshold, 255, cv2.THRESH_BINARY)
        
        motion_pixels = np.count_nonzero(motion_mask)
        self.motion_history.append(motion_pixels)
        
        return motion_mask
    
    def detect_oscillation(self):
        """Detect 0.5-2 sec oscillating patterns (UI overlays)"""
        if len(self.motion_history) < 30:
            return False
        
        recent = list(self.motion_history)
        
        for period in range(15, 61):
            if len(recent) < period * 2:
                continue
            
            first_half = recent[:period]
            second_half = recent[period:period*2]
            
            if len(first_half) == len(second_half):
                corr = np.corrcoef(first_half, second_half)[0, 1]
                if corr > 0.85:
                    self.oscillation_suppression = 1.0
                    return True
        
        self.oscillation_suppression = max(0, self.oscillation_suppression - 0.05)
        return False
    
    def adapt_threshold(self, motion_mask):
        """Auto-adapt threshold based on motion brightness"""
        motion_pixels = motion_mask > 0
        
        if np.any(motion_pixels):
            self.current_brightness = np.mean(self.prev_frame[motion_pixels])
            delta = (self.target_brightness - self.current_brightness) * (self.adapt_rate / 100.0)
            self.base_threshold = max(5, self.base_threshold + delta)


# =========================
# INTERSECTION DETECTOR
# =========================
class IntersectionDetector:
    """Detects intersection points of small fish trajectories"""
    
    def __init__(self):
        self.small_fish_history = deque(maxlen=100)
        self.intersection_points = []
        self.last_intersection_shot_time = time.time()
        self.min_shot_interval = 0.3
    
    def detect_intersections(self, small_fishes):
        """Calculate intersection points for small fish trajectories"""
        self.intersection_points = []
        
        if len(small_fishes) < 2:
            return 0
        
        for i in range(len(small_fishes)):
            for j in range(i + 1, len(small_fishes)):
                f1 = small_fishes[i]
                f2 = small_fishes[j]
                
                x1, y1 = f1.x + f1.w // 2, f1.y + f1.h // 2
                x2, y2 = f2.x + f2.w // 2, f2.y + f2.h // 2
                
                if f1.speed < 0.5 or f2.speed < 0.5:
                    continue
                
                intersection = self._calculate_intersection(
                    x1, y1, f1.vx, f1.vy,
                    x2, y2, f2.vx, f2.vy
                )
                
                if intersection:
                    self.intersection_points.append({
                        'point': intersection['point'],
                        'distance': intersection['distance'],
                        'confidence': intersection['confidence'],
                        'fish_ids': (f1.id, f2.id)
                    })
        
        return len(self.intersection_points)
    
    def _calculate_intersection(self, x1, y1, vx1, vy1, x2, y2, vx2, vy2):
        """Calculate if two trajectories intersect"""
        min_dist = float('inf')
        intersection_point = None
        
        for t in range(1, 21):
            proj_x1 = x1 + vx1 * t
            proj_y1 = y1 + vy1 * t
            proj_x2 = x2 + vx2 * t
            proj_y2 = y2 + vy2 * t
            
            dist = np.sqrt((proj_x2 - proj_x1)**2 + (proj_y2 - proj_y1)**2)
            
            if dist < min_dist:
                min_dist = dist
                intersection_point = ((proj_x1 + proj_x2) / 2, (proj_y1 + proj_y2) / 2)
        
        if min_dist < 30 and intersection_point:
            return {
                'point': intersection_point,
                'distance': min_dist,
                'confidence': max(0, 1.0 - min_dist / 50.0)
            }
        
        return None
    
    def get_intersection_points(self):
        """Return list of all detected intersection points"""
        return self.intersection_points
    
    def should_fire_intersection_shot(self):
        """Check if enough time has passed for next intersection shot"""
        return time.time() - self.last_intersection_shot_time >= self.min_shot_interval
    
    def record_intersection_shot(self):
        """Record that we fired at an intersection point"""
        self.last_intersection_shot_time = time.time()


# =========================
# BLOB TRACKER & CLUSTERING
# =========================
class Fish:
    """Represents a single fish target"""
    def __init__(self, x, y, w, h, fish_id):
        self.id = fish_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.vx = 0
        self.vy = 0
        self.speed = 0
        self.heading = 0
        self.class_id = self._classify()
        self.history = deque(maxlen=10)
        self.history.append((x, y))
        self.age = 0
        self.last_seen = time.time()
        self.shots = 0
    
    def _classify(self):
        """Classify fish by size into 10+ classes"""
        if self.area < 50:
            return 0
        elif self.area < 100:
            return 1
        elif self.area < 200:
            return 2
        elif self.area < 400:
            return 3
        elif self.area < 600:
            return 4
        elif self.area < 900:
            return 5
        elif self.area < 1200:
            return 6
        elif self.area < 1800:
            return 7
        elif self.area < 2500:
            return 8
        else:
            return 9
    
    def update(self, x, y, w, h):
        """Update position and velocity"""
        self.vx = x - self.x
        self.vy = y - self.y
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        
        if self.speed > 0:
            self.heading = np.arctan2(self.vy, self.vx) * 180 / np.pi
        
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.class_id = self._classify()
        self.history.append((x, y))
        self.age += 1
        self.last_seen = time.time()
    
    def is_same_trajectory(self, other, dist_thresh=30, angle_thresh=15, speed_thresh=0.5):
        """Check if two blobs are same fish (same trajectory)"""
        dist = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if dist > dist_thresh:
            return False
        
        if self.speed < 2 or other.speed < 2:
            return dist < dist_thresh * 0.5
        
        angle_diff = abs(self.heading - other.heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > angle_thresh:
            return False
        
        speed_ratio = min(self.speed, other.speed) / max(self.speed, other.speed, 0.1)
        if speed_ratio < (1.0 - speed_thresh):
            return False
        
        return True


class BlobTracker:
    """Tracks blobs and merges same-trajectory fragments into single fish"""
    
    def __init__(self):
        self.fishes = []
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections, motion_mask):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        valid_detections = []
        for x, y, w, h in detections:
            if w < 3 or h < 3:
                continue
            
            cx, cy = x + w // 2, y + h // 2
            if motion_mask[min(cy, motion_mask.shape[0]-1), min(cx, motion_mask.shape[1]-1)] > 128:
                valid_detections.append((x, y, w, h))
        
        matched = set()
        
        for x, y, w, h in valid_detections:
            best_match = None
            best_dist = float('inf')
            
            for fish in self.fishes:
                dist = np.sqrt((fish.x - x)**2 + (fish.y - y)**2)
                if dist < best_dist and dist < 50:
                    best_match = fish
                    best_dist = dist
            
            if best_match:
                best_match.update(x, y, w, h)
                matched.add(best_match.id)
            else:
                new_fish = Fish(x, y, w, h, self.next_id)
                self.fishes.append(new_fish)
                self.next_id += 1
        
        self.fishes = [f for f in self.fishes if f.id in matched or f.age < 5]
        self._merge_trajectories()
        
        return self.fishes
    
    def _merge_trajectories(self):
        """Merge fish with identical trajectory into one"""
        merged = []
        used = set()
        
        for i, f1 in enumerate(self.fishes):
            if f1.id in used:
                continue
            
            group = [f1]
            used.add(f1.id)
            
            for j, f2 in enumerate(self.fishes):
                if j <= i or f2.id in used:
                    continue
                
                if f1.is_same_trajectory(f2):
                    group.append(f2)
                    used.add(f2.id)
            
            group_sorted = sorted(group, key=lambda f: f.area, reverse=True)
            keeper = group_sorted[0]
            
            for absorbed in group_sorted[1:]:
                keeper.shots += absorbed.shots
            
            merged.append(keeper)
        
        self.fishes = merged


# =========================
# AGGRESSION CONTROL SYSTEM (-100% to +100%)
# =========================
class AggressionController:
    """Controls firing aggression from -100% (conservative) to +100% (constant fire)"""
    
    def __init__(self):
        self.aggression_level = 0  # 0% = baseline balanced
        self.max_aggression = 100  # 100% = constant fire
        self.min_aggression = -100  # -100% = purely conservative
        self.firing_threshold = 0.7
        self.increment = 10  # ±10% per keypress
        self.force_intersection_shots = False
    
    def increase(self):
        """Increase aggression (press +)"""
        self.aggression_level = min(self.max_aggression, self.aggression_level + self.increment)
        
        # Calculate multiplier: -100% = 0.1x, 0% = 1.0x, +100% = 10.0x
        if self.aggression_level >= 0:
            multiplier = 1.0 + (self.aggression_level / 100.0) * 9.0  # 1.0 to 10.0
        else:
            multiplier = 1.0 + (self.aggression_level / 100.0)  # 1.0 to 0.1
        
        adjusted_threshold = max(0.1, self.firing_threshold / multiplier)
        
        if self.aggression_level >= 50:
            self.force_intersection_shots = True
        
        return self.aggression_level, multiplier, adjusted_threshold
    
    def decrease(self):
        """Decrease aggression (press -)"""
        self.aggression_level = max(self.min_aggression, self.aggression_level - self.increment)
        
        if self.aggression_level >= 0:
            multiplier = 1.0 + (self.aggression_level / 100.0) * 9.0
        else:
            multiplier = 1.0 + (self.aggression_level / 100.0)
        
        adjusted_threshold = max(0.1, self.firing_threshold / multiplier)
        
        if self.aggression_level < 50:
            self.force_intersection_shots = False
        
        return self.aggression_level, multiplier, adjusted_threshold
    
    def get_firing_confidence_multiplier(self):
        """Get multiplier for confidence calculations"""
        if self.aggression_level >= 0:
            return 1.0 + (self.aggression_level / 100.0) * 9.0
        else:
            return 1.0 + (self.aggression_level / 100.0)
    
    def get_adjusted_threshold(self):
        """Get firing threshold adjusted by aggression"""
        multiplier = self.get_firing_confidence_multiplier()
        return max(0.1, self.firing_threshold / multiplier)
    
    def is_intersection_forcing_active(self):
        """Check if forced intersection shots are enabled"""
        return self.force_intersection_shots
    
    def get_fire_rate_multiplier(self):
        """Get fire rate multiplier based on aggression"""
        if self.aggression_level >= 100:
            return 10.0  # Constant fire
        elif self.aggression_level >= 0:
            return 1.0 + (self.aggression_level / 100.0)
        else:
            return 0.1 + (self.aggression_level / 100.0) * 0.9  # Down to 0.1


# =========================
# AUTOFIRE TOGGLE SYSTEM
# =========================
class AutofireController:
    """Controls autofire system with Shift hotkey toggle"""
    
    def __init__(self):
        self.autofire_enabled = False
        self.last_toggle_time = time.time()
        self.toggle_cooldown = 0.2
    
    def can_toggle(self):
        """Check if enough time has passed since last toggle"""
        return time.time() - self.last_toggle_time >= self.toggle_cooldown
    
    def toggle(self):
        """Toggle autofire on/off"""
        if self.can_toggle():
            self.autofire_enabled = not self.autofire_enabled
            self.last_toggle_time = time.time()
            return True
        return False
    
    def is_enabled(self):
        """Check if autofire is currently enabled"""
        return self.autofire_enabled
    
    def get_status_text(self):
        """Get human-readable status"""
        return "✅ ON" if self.autofire_enabled else "❌ OFF"
    
    def get_status_color(self):
        """Get color for status display (BGR)"""
        return (0, 255, 0) if self.autofire_enabled else (0, 0, 255)


# =========================
# SAFE CLICK VALIDATOR
# =========================
class SafeClickValidator:
    """Ensures clicks stay within game window"""
    
    def __init__(self, region):
        self.region = region
        self.safe_margin = 10
    
    def validate_click(self, screen_x, screen_y):
        """Check if click is within window bounds"""
        left = self.region["left"] + self.safe_margin
        right = self.region["left"] + self.region["width"] - self.safe_margin
        top = self.region["top"] + self.safe_margin
        bottom = self.region["top"] + self.region["height"] - self.safe_margin
        
        if left <= screen_x <= right and top <= screen_y <= bottom:
            return True, (screen_x, screen_y)
        
        # Clamp to safe bounds
        clamped_x = max(left, min(right, screen_x))
        clamped_y = max(top, min(bottom, screen_y))
        
        return False, (clamped_x, clamped_y)


# =========================
# MAIN APPLICATION v7.0 - VISION VERIFICATION
# =========================
def main():
    print("=" * 90)
    print("🎮 HYBRID FISH DETECTION SYSTEM v7.0 - VISION VERIFICATION & OVERLAY")
    print("=" * 90)
    print("✨ NEW FEATURES:")
    print("  • 📸 Screenshots every 0.5s (Mask, Radar, Capture)")
    print("  • 🔍 Overlay compositor for fact-checking")
    print("  • ✅ Accuracy verifier & mask quality analyzer")
    print("  • 📊 Aggression range: -100% (conservative) to +100% (constant fire)")
    print("  • 🔒 Safe click validator (clicks stay in window)")
    print("  • 🖥️ HUD visible in fullscreen + hotkeys work globally")
    print("=" * 90)
    print("🎮 HOTKEY CONTROLS:")
    print("  • SHIFT: Toggle AUTOFIRE ON/OFF")
    print("  • +: Increase Aggression (+10%)")
    print("  • -: Decrease Aggression (-10%)")
    print("  • Q: Quit Application")
    print("  • F: Toggle Fullscreen Overlay")
    print("=" * 90)
    
    # Setup
    region = get_chrome_region()
    print(f"📍 Capturing from: {region}")
    print(f"⏳ Initializing systems...")
    
    sct = mss.mss()
    wm = WindowManager()
    motion_detector = MotionDetector(region["width"], region["height"])
    tracker = BlobTracker()
    intersection_detector = IntersectionDetector()
    aggression_controller = AggressionController()
    autofire_controller = AutofireController()
    hotkey_listener = HotkeyListener()
    screenshot_manager = ScreenshotManager()
    accuracy_verifier = AccuracyVerifier()
    safe_click_validator = SafeClickValidator(region)
    
    # Create windows
    wm.create("VISUALIZATION", 50, 50, 800, 600)
    wm.create("OVERLAY_CHECKER", 900, 50, 1000, 600)
    wm.create("RADAR_DISPLAY", 50, 700, 400, 400)
    
    # Register hotkey callbacks
    def toggle_autofire():
        if autofire_controller.toggle():
            status = "✅ ON" if autofire_controller.is_enabled() else "❌ OFF"
            print(f"🔄 AUTOFIRE TOGGLED: {status}")
    
    def increase_agg():
        agg, mult, thresh = aggression_controller.increase()
        print(f"📈 Aggression: {agg:+3d}% | Multiplier: {mult:.1f}x | Threshold: {thresh:.2f}")
    
    def decrease_agg():
        agg, mult, thresh = aggression_controller.decrease()
        print(f"📉 Aggression: {agg:+3d}% | Multiplier: {mult:.1f}x | Threshold: {thresh:.2f}")
    
    hotkey_listener.register_callback('shift', toggle_autofire)
    hotkey_listener.register_callback('plus', increase_agg)
    hotkey_listener.register_callback('minus', decrease_agg)
    
    # Start hotkey listener
    hotkey_listener.start()
    print("✅ Hotkey Listener Started (pynput)")
    print("✅ Ready to accept input!")
    print("=" * 90)
    
    frame_count = 0
    start_time = time.time()
    intersection_shots_fired = 0
    normal_shots_fired = 0
    should_quit = False
    fullscreen_overlay = False
    
    try:
        while not should_quit:
            # Capture
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            if frame is None or frame.size == 0:
                continue
            
            h, w = frame.shape[:2]
            
            # Motion detection
            motion_mask = motion_detector.detect_motion(frame)
            
            if motion_mask is None or motion_mask.size == 0:
                continue
            
            # Check for oscillations
            is_oscillating = motion_detector.detect_oscillation()
            
            if is_oscillating:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Adapt threshold
            motion_detector.adapt_threshold(motion_mask)
            
            # Find contours
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for c in contours:
                x, y, w_c, h_c = cv2.boundingRect(c)
                area = w_c * h_c
                
                if area < 20:
                    continue
                
                aspect = w_c / max(h_c, 1)
                if aspect > 5 or aspect < 0.2:
                    continue
                
                detections.append((x, y, w_c, h_c))
            
            # Track blobs
            fishes = tracker.update(detections, motion_mask)
            
            # Small fish intersection detection
            small_fishes = [f for f in fishes if f.class_id <= 2]
            intersection_count = intersection_detector.detect_intersections(small_fishes)
            intersection_points = intersection_detector.get_intersection_points()
            
            # Record detections for accuracy
            for fish in fishes:
                confidence = 0.6 + (0.1 * max(0, 2 - fish.class_id))
                accuracy_verifier.record_detection(fish.id, fish.class_id, fish.area, confidence)
            
            # Firing logic
            fire_rate_mult = aggression_controller.get_fire_rate_multiplier()
            
            if autofire_controller.is_enabled():
                confidence_multiplier = aggression_controller.get_firing_confidence_multiplier()
                adjusted_threshold = aggression_controller.get_adjusted_threshold()
                
                # Forced intersection shots
                if aggression_controller.is_intersection_forcing_active() and intersection_points:
                    if intersection_detector.should_fire_intersection_shot():
                        best_intersection = max(intersection_points, key=lambda x: x['confidence'])
                        pt = best_intersection['point']
                        
                        screen_x = region["left"] + int(pt[0])
                        screen_y = region["top"] + int(pt[1])
                        
                        is_safe, (safe_x, safe_y) = safe_click_validator.validate_click(screen_x, screen_y)
                        
                        pyautogui.click(safe_x, safe_y)
                        intersection_detector.record_intersection_shot()
                        intersection_shots_fired += 1
                
                # Normal fish targeting
                else:
                    optimal_targets = []
                    for fish in small_fishes:
                        base_confidence = 0.6 + (0.1 * (2 - fish.class_id))
                        adjusted_confidence = min(1.0, base_confidence * confidence_multiplier)
                        
                        if adjusted_confidence >= adjusted_threshold:
                            optimal_targets.append((fish, adjusted_confidence))
                    
                    optimal_targets.sort(key=lambda x: -x[1])
                    
                    if optimal_targets and len(optimal_targets) >= 2:
                        target_fish = optimal_targets[0][0]
                        cx = int(target_fish.x + target_fish.w // 2)
                        cy = int(target_fish.y + target_fish.h // 2)
                        
                        screen_x = region["left"] + cx
                        screen_y = region["top"] + cy
                        
                        is_safe, (safe_x, safe_y) = safe_click_validator.validate_click(screen_x, screen_y)
                        
                        pyautogui.click(safe_x, safe_y)
                        normal_shots_fired += 1
            
            # === VISUALIZATION ===
            vis = frame.copy()
            
            class_colors = [
                (0, 255, 0), (50, 255, 0), (100, 255, 0), (150, 255, 0),
                (0, 255, 255), (0, 200, 255), (0, 128, 255), (0, 100, 200),
                (0, 0, 255), (255, 0, 0)
            ]
            
            for fish in fishes:
                color = class_colors[min(fish.class_id, 9)]
                cv2.rectangle(vis, (fish.x, fish.y), (fish.x + fish.w, fish.y + fish.h), color, 2)
                
                cx, cy = fish.x + fish.w // 2, fish.y + fish.h // 2
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                
                if fish.speed > 2:
                    px = cx + int(fish.vx * 3)
                    py = cy + int(fish.vy * 3)
                    cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 0, 0), 2, tipLength=0.3)
                
                label = f"ID:{fish.id} C:{fish.class_id} S:{fish.speed:.1f}"
                cv2.putText(vis, label, (fish.x, fish.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw intersection points
            for pt_data in intersection_points:
                pt = pt_data['point']
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (255, 255, 0), 2)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
                
                if aggression_controller.is_intersection_forcing_active() and len(intersection_points) > 0:
                    best = max(intersection_points, key=lambda x: x['confidence'])
                    if pt == best['point']:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 12, (0, 0, 255), 3)
                        cv2.putText(vis, "TARGET", (int(pt[0])-30, int(pt[1])-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # HUD overlay (visible in fullscreen)
            hud_height = 200
            hud = np.zeros((hud_height, vis.shape[1], 3), dtype=np.uint8)
            hud[:] = (30, 30, 30)
            
            fps = frame_count / (time.time() - start_time + 0.001)
            forcing_status = "🔴 FORCING" if aggression_controller.is_intersection_forcing_active() else "⚪ NORMAL"
            autofire_text = autofire_controller.get_status_text()
            autofire_color = autofire_controller.get_status_color()
            
            hud_stats = [
                f"FPS: {fps:.1f} | Agg: {aggression_controller.aggression_level:+3d}% | Mult: {aggression_controller.get_firing_confidence_multiplier():.1f}x",
                f"Targets: {len(fishes)} | Small: {len(small_fishes)} | Intersections: {intersection_count}",
                f"Threshold: {aggression_controller.get_adjusted_threshold():.2f} | {forcing_status}",
                f"AUTOFIRE: {autofire_text}",
                f"Int Shots: {intersection_shots_fired} | Normal: {normal_shots_fired}",
                f"🎮 HOTKEYS: SHIFT=Toggle  +/-=Agg  Q=Quit  F=Fullscreen"
            ]
            
            for i, stat in enumerate(hud_stats):
                color = (0, 255, 0) if "AUTOFIRE" in stat and autofire_controller.is_enabled() else (0, 0, 255) if "AUTOFIRE" in stat else (255, 255, 255)
                cv2.putText(hud, stat, (10, 25 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            vis_with_hud = np.vstack([hud, vis])
            wm.show("VISUALIZATION", vis_with_hud)
            
            # === RADAR ===
            radar_size = 300
            radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
            
            for fish in fishes:
                rx = int((fish.x / w) * radar_size)
                ry = int((fish.y / h) * radar_size)
                color = class_colors[min(fish.class_id, 9)]
                radius = 3 + fish.class_id
                cv2.circle(radar, (rx, ry), radius, color, -1)
            
            for pt_data in intersection_points:
                pt = pt_data['point']
                rx = int((pt[0] / w) * radar_size)
                ry = int((pt[1] / h) * radar_size)
                cv2.circle(radar, (rx, ry), 5, (255, 255, 0), 2)
                cv2.circle(radar, (rx, ry), 2, (0, 255, 255), -1)
            
            radar_text = [
                f"Agg: {aggression_controller.aggression_level:+3d}%",
                f"Int: {intersection_count}",
                f"Auto: {autofire_text}"
            ]
            
            for i, txt in enumerate(radar_text):
                cv2.putText(radar, txt, (10, 20 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            wm.show("RADAR_DISPLAY", radar)
            
            # === PERIODIC SCREENSHOTS FOR ANALYSIS ===
            if screenshot_manager.should_take_screenshot():
                overlay = OverlayCompositor.create_overlay(
                    cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR),
                    radar,
                    frame,
                    stats=[
                        f"Agg: {aggression_controller.aggression_level:+3d}% | Targets: {len(fishes)} | Int: {intersection_count}",
                        f"Thresh: {motion_detector.base_threshold:.1f} | Motion: {np.count_nonzero(motion_mask)//100}%",
                        f"Quality: {accuracy_verifier.analyze_mask_quality(motion_mask, frame)}"
                    ]
                )
                wm.show("OVERLAY_CHECKER", overlay)
            
            # === INPUT HANDLING ===
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                should_quit = True
                print("\n⏹️ Quit command received!")
            
            if key == ord('f') or key == ord('F'):
                fullscreen_overlay = not fullscreen_overlay
                wm.toggle_fullscreen("OVERLAY_CHECKER")
            
            # Check shift toggle
            if hotkey_listener.check_shift_toggled():
                pass  # Already handled by callback
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user (Ctrl+C)")
    finally:
        hotkey_listener.stop()
        print(f"\n📊 SESSION SUMMARY:")
        print(f"  Total Frames: {frame_count}")
        print(f"  Intersection Shots: {intersection_shots_fired}")
        print(f"  Normal Shots: {normal_shots_fired}")
        print(f"  Total Shots: {intersection_shots_fired + normal_shots_fired}")
        print(f"  Final Aggression: {aggression_controller.aggression_level:+3d}%")
        print(f"  Autofire Final: {autofire_controller.get_status_text()}")
        print(f"  Screenshots saved to: screenshots/")
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")


if __name__ == "__main__":
    main()