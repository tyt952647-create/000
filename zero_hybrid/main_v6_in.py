import cv2
import numpy as np
import mss
import time
import pyautogui
from collections import deque
from pathlib import Path
import threading
from pynput import keyboard

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
            # Shift key
            if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                if not self.shift_pressed:
                    self.shift_pressed = True
                    self.shift_toggled = True
                    if 'shift' in self.callbacks:
                        self.callbacks['shift']()
            
            # Plus/Equal key (for aggression increase)
            if key == keyboard.Key.equal or str(key) == "'+'":
                if 'plus' in self.callbacks:
                    self.callbacks['plus']()
            
            # Minus key (for aggression decrease)
            if key == keyboard.Key.minus or str(key) == "'-'":
                if 'minus' in self.callbacks:
                    self.callbacks['minus']()
            
            # Q key (quit)
            if hasattr(key, 'char'):
                if key.char == 'q' or key.char == 'Q':
                    if 'q' in self.callbacks:
                        self.callbacks['q']()
                        
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release"""
        try:
            # Shift key release
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
        
    def create(self, name, x, y, width=500, height=500):
        """Create and position a window"""
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        cv2.moveWindow(name, x, y)
        self.windows[name] = (x, y, width, height)
        return name
    
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
        import win32con
        
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
            
            # Capture right-half only
            return {
                "left": left + width // 2,
                "top": top,
                "width": width // 2,
                "height": height
            }
    except:
        pass
    
    # Fallback: use primary monitor's right half
    sct = mss.mss()
    monitor = sct.monitors[1]
    return {
        "left": monitor["width"] // 2,
        "top": 0,
        "width": monitor["width"] // 2,
        "height": monitor["height"]
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
# SMALL FISH INTERSECTION DETECTOR
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
# AGGRESSION CONTROL SYSTEM (1000% MAX)
# =========================
class AggressionController:
    """Controls firing aggression with 1000% max multiplier and intersection forcing"""
    
    def __init__(self):
        self.aggression_level = 100
        self.max_aggression = 1000
        self.min_aggression = 10
        self.firing_threshold = 0.7
        self.increment = 50
        self.force_intersection_shots = False
    
    def increase(self):
        """Increase aggression (press +)"""
        self.aggression_level = min(self.max_aggression, self.aggression_level + self.increment)
        multiplier = self.aggression_level / 100.0
        adjusted_threshold = max(0.1, self.firing_threshold / multiplier)
        
        if self.aggression_level >= 100:
            self.force_intersection_shots = True
        
        return self.aggression_level, adjusted_threshold
    
    def decrease(self):
        """Decrease aggression (press -)"""
        self.aggression_level = max(self.min_aggression, self.aggression_level - self.increment)
        multiplier = self.aggression_level / 100.0
        adjusted_threshold = max(0.1, self.firing_threshold / multiplier)
        
        if self.aggression_level < 100:
            self.force_intersection_shots = False
        
        return self.aggression_level, adjusted_threshold
    
    def get_firing_confidence_multiplier(self):
        """Get multiplier for confidence calculations"""
        return self.aggression_level / 100.0
    
    def get_adjusted_threshold(self):
        """Get firing threshold adjusted by aggression"""
        multiplier = self.aggression_level / 100.0
        return max(0.1, self.firing_threshold / multiplier)
    
    def is_intersection_forcing_active(self):
        """Check if forced intersection shots are enabled"""
        return self.force_intersection_shots


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
# MAIN APPLICATION v6.2 - HOTKEYS FIXED
# =========================
def main():
    print("=" * 80)
    print("🎮 HYBRID FISH DETECTION SYSTEM v6.2 - HOTKEYS FIXED WITH PYNPUT")
    print("=" * 80)
    print("✨ Features:")
    print("  • Small Fish Intersection Point Counter")
    print("  • Max Aggression: 1000% (10x multiplier)")
    print("  • Firing Threshold: 70% (30% lower than original)")
    print("  • 🔴 FORCED INTERSECTION SHOTS at 100%+ aggression")
    print("  • ⏸️ AUTOFIRE TOGGLE with Shift key (starts OFF)")
    print("  • 10+ Dynamic Fish Classes")
    print("=" * 80)
    print("🎮 HOTKEY CONTROLS:")
    print("  • SHIFT: Toggle AUTOFIRE ON/OFF")
    print("  • +/-: Adjust Aggression (10%-1000%)")
    print("  • Q: Quit Application")
    print("=" * 80)
    
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
    
    # Create windows
    wm.create("VISUALIZATION", 50, 50, 800, 600)
    wm.create("MASK CONTROLS", 900, 50, 500, 600)
    wm.create("RADAR DISPLAY", 50, 700, 400, 400)
    
    # Register hotkey callbacks
    hotkey_listener.register_callback('shift', lambda: autofire_controller.toggle())
    hotkey_listener.register_callback('plus', lambda: aggression_controller.increase())
    hotkey_listener.register_callback('minus', lambda: aggression_controller.decrease())
    
    # Start hotkey listener
    hotkey_listener.start()
    print("✅ Hotkey Listener Started (pynput)")
    print("✅ Ready to accept input!")
    print("=" * 80)
    
    frame_count = 0
    start_time = time.time()
    intersection_shots_fired = 0
    normal_shots_fired = 0
    should_quit = False
    
    try:
        while not should_quit:
            # Capture
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            if frame is None or frame.size == 0:
                print("⚠️ Frame capture failed, retrying...")
                time.sleep(0.1)
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
            
            # Find contours in motion mask
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
            
            # Firing logic with autofire toggle
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
                        
                        pyautogui.click(screen_x, screen_y)
                        intersection_detector.record_intersection_shot()
                        intersection_shots_fired += 1
                        
                        print(f"🔴 INTERSECTION SHOT #{intersection_shots_fired} at ({int(pt[0])}, {int(pt[1])}) | Conf: {best_intersection['confidence']:.2f}")
                
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
                        
                        pyautogui.click(screen_x, screen_y)
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
            
            # Stats
            fps = frame_count / (time.time() - start_time + 0.001)
            motion_pct = np.count_nonzero(motion_mask) // 100
            
            stats_y = 30
            forcing_status = "🔴 FORCING ON" if aggression_controller.is_intersection_forcing_active() else "⚪ NORMAL"
            autofire_color = autofire_controller.get_status_color()
            autofire_text = autofire_controller.get_status_text()
            
            stats = [
                f"FPS: {fps:.1f}",
                f"Targets: {len(fishes)} | Small: {len(small_fishes)}",
                f"🔄 Intersections: {intersection_count}",
                f"📊 Agg: {aggression_controller.aggression_level}% | {forcing_status}",
                f"🎯 Threshold: {aggression_controller.get_adjusted_threshold():.2f}",
                f"💥 Int Shots: {intersection_shots_fired} | Normal: {normal_shots_fired}",
                f"⏸️ AUTOFIRE: {autofire_text}",
                f"Motion: {motion_pct}% | Adapt: {motion_detector.base_threshold:.1f}"
            ]
            
            for i, stat in enumerate(stats):
                if "AUTOFIRE" in stat:
                    cv2.putText(vis, stat, (10, stats_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, autofire_color, 2)
                elif "FORCING ON" in stat:
                    cv2.putText(vis, stat, (10, stats_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(vis, stat, (10, stats_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            wm.show("VISUALIZATION", vis)
            
            # === MASK CONTROLS ===
            mask_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_display, f"Thresh: {motion_detector.base_threshold:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_display, f"Motion: {np.count_nonzero(motion_mask)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_display, f"Oscillating: {'YES' if is_oscillating else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_oscillating else (0, 255, 0), 2)
            cv2.putText(mask_display, f"Small Fish: {len(small_fishes)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(mask_display, f"Intersections: {intersection_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            forcing_color = (0, 0, 255) if aggression_controller.is_intersection_forcing_active() else (0, 255, 0)
            forcing_text = "FORCING: ON" if aggression_controller.is_intersection_forcing_active() else "FORCING: OFF"
            cv2.putText(mask_display, forcing_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, forcing_color, 2)
            
            autofire_status = f"AUTOFIRE: {autofire_text}"
            cv2.putText(mask_display, autofire_status, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, autofire_color, 2)
            
            cv2.putText(mask_display, f"🎮 HOTKEYS: SHIFT=Toggle  +/-=Agg  Q=Quit", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(mask_display, f"Int: {intersection_shots_fired} | Norm: {normal_shots_fired}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            wm.show("MASK CONTROLS", mask_display)
            
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
            
            cv2.putText(radar, f"Agg: {aggression_controller.aggression_level}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(radar, f"Int: {intersection_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            forcing_status_radar = "FORCE: ON" if aggression_controller.is_intersection_forcing_active() else "FORCE: OFF"
            forcing_color_radar = (0, 0, 255) if aggression_controller.is_intersection_forcing_active() else (0, 255, 0)
            cv2.putText(radar, forcing_status_radar, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, forcing_color_radar, 1)
            
            autofire_status_radar = f"AUTO: {autofire_text}"
            cv2.putText(radar, autofire_status_radar, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, autofire_color, 1)
            
            wm.show("RADAR DISPLAY", radar)
            
            # === INPUT HANDLING ===
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                should_quit = True
                print("\n⏹️ Quit command received!")
            
            # Check shift toggle
            if hotkey_listener.check_shift_toggled():
                if autofire_controller.toggle():
                    status = "✅ ON" if autofire_controller.is_enabled() else "❌ OFF"
                    print(f"🔄 AUTOFIRE TOGGLED: {status}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user (Ctrl+C)")
    finally:
        hotkey_listener.stop()
        print(f"\n📊 Session Summary:")
        print(f"  Total Frames: {frame_count}")
        print(f"  Total Intersection Shots: {intersection_shots_fired}")
        print(f"  Total Normal Shots: {normal_shots_fired}")
        print(f"  Total Shots: {intersection_shots_fired + normal_shots_fired}")
        print(f"  Final Aggression: {aggression_controller.aggression_level}%")
        print(f"  Autofire Final: {autofire_controller.get_status_text()}")
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")


if __name__ == "__main__":
    main()