import cv2
import numpy as np
import mss
import time
import pyautogui
import json
from collections import deque
from pathlib import Path
from ocr_scoring_system import OCRScoringSystem, MultiClassFishClassifier, BulletConservationManager

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
        self.adapt_rate = adapt_rate  # % per frame
        self.prev_frame = None
        self.motion_history = deque(maxlen=60)  # 2 sec @ 30fps
        self.base_threshold = 15  # Frame diff threshold
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
        
        # Absolute difference = motion
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray.copy()
        
        # Binary: only high-motion pixels
        _, motion_mask = cv2.threshold(diff, self.base_threshold, 255, cv2.THRESH_BINARY)
        
        # Track motion over time for oscillation detection
        motion_pixels = np.count_nonzero(motion_mask)
        self.motion_history.append(motion_pixels)
        
        return motion_mask
    
    def detect_oscillation(self):
        """Detect 0.5-2 sec oscillating patterns (UI overlays)"""
        if len(self.motion_history) < 30:
            return False
        
        # Look for patterns repeating every 15-60 frames (0.5-2 sec @ 30fps)
        recent = list(self.motion_history)
        
        for period in range(15, 61):
            if len(recent) < period * 2:
                continue
            
            # Compare motion amplitude at period intervals
            first_half = recent[:period]
            second_half = recent[period:period*2]
            
            # If oscillating, the two halves should be similar
            if len(first_half) == len(second_half):
                corr = np.corrcoef(first_half, second_half)[0, 1]
                if corr > 0.85:  # High correlation = oscillation
                    self.oscillation_suppression = 1.0
                    return True
        
        # Decay suppression
        self.oscillation_suppression = max(0, self.oscillation_suppression - 0.05)
        return False
    
    def adapt_threshold(self, motion_mask):
        """Auto-adapt threshold based on motion brightness"""
        motion_pixels = motion_mask > 0
        
        if np.any(motion_pixels):
            # Get brightness of moving pixels only
            self.current_brightness = np.mean(self.prev_frame[motion_pixels])
            
            # Slowly adapt toward middle gray (127)
            delta = (self.target_brightness - self.current_brightness) * (self.adapt_rate / 100.0)
            self.base_threshold = max(5, self.base_threshold + delta)


# =========================
# BLOB TRACKER & CLUSTERING
# =========================
class Fish:
    """Represents a single fish target with enhanced attributes"""
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
        self.kill_confidence = 0.0
    
    def _classify(self):
        """Classify fish by size (0-5 main + modifiers)"""
        if self.area < 100:
            return 0
        elif self.area < 300:
            return 1
        elif self.area < 600:
            return 2
        elif self.area < 1200:
            return 3
        elif self.area < 2500:
            return 4
        else:
            return 5
    
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
        # Distance check
        dist = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if dist > dist_thresh:
            return False
        
        # If either not moving, can't confirm trajectory
        if self.speed < 2 or other.speed < 2:
            return dist < dist_thresh * 0.5
        
        # Heading check
        angle_diff = abs(self.heading - other.heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > angle_thresh:
            return False
        
        # Speed check
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
        
        # Filter: only accept detections in motion regions
        valid_detections = []
        for x, y, w, h in detections:
            if w < 3 or h < 3:
                continue
            
            # Check if detection overlaps motion
            cx, cy = x + w // 2, y + h // 2
            if motion_mask[min(cy, motion_mask.shape[0]-1), min(cx, motion_mask.shape[1]-1)] > 128:
                valid_detections.append((x, y, w, h))
        
        # Match detections to existing fish
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
                # New fish
                new_fish = Fish(x, y, w, h, self.next_id)
                self.fishes.append(new_fish)
                self.next_id += 1
        
        # Remove unmatched old fish
        self.fishes = [f for f in self.fishes if f.id in matched or f.age < 5]
        
        # Merge same-trajectory fragments
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
            
            # Keep largest fish in group, absorb others
            group_sorted = sorted(group, key=lambda f: f.area, reverse=True)
            keeper = group_sorted[0]
            
            for absorbed in group_sorted[1:]:
                keeper.shots += absorbed.shots
            
            merged.append(keeper)
        
        self.fishes = merged


# =========================
# MAIN APPLICATION
# =========================
def main():
    print("🎮 ZERO HYBRID v4.0 - ENHANCED TARGETING & EFFICIENCY")
    print("=" * 60)
    
    # Setup
    region = get_chrome_region()
    print(f"📍 Capturing from: {region}")
    
    sct = mss.mss()
    wm = WindowManager()
    motion_detector = MotionDetector(region["width"], region["height"])
    tracker = BlobTracker()
    
    # Initialize learning systems
    ocr_system = OCRScoringSystem()
    classifier = MultiClassFishClassifier()
    bullet_manager = BulletConservationManager(cost_per_shot=1.0)
    
    # Create windows at specific positions
    wm.create("VISUALIZATION", 50, 50, 800, 600)
    wm.create("STATS", 900, 50, 500, 600)
    wm.create("RADAR DISPLAY", 50, 700, 400, 400)
    
    # Control parameters
    fire_aggression = 0.5  # 0=conservative, 1.0=aggressive
    last_aggression_update = time.time()
    
    print("🎯 Motion-based detection active")
    print("⚙️ OCR multiplier learning enabled")
    print("💥 Multi-class targeting (10+ classes)")
    print("🔫 Bullet conservation active")
    print("🔴 Controls: Q=quit, +=aggressive, -=conservative")
    print("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
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
                # Suppress oscillation by morphological closing
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.mologyclose(motion_mask, kernel, iterations=2)
            
            # Adapt threshold
            motion_detector.adapt_threshold(motion_mask)
            
            # Find contours in motion mask
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for c in contours:
                x, y, w_c, h_c = cv2.boundingRect(c)
                area = w_c * h_c
                
                # Filter by size and aspect ratio
                if area < 20:
                    continue
                
                aspect = w_c / max(h_c, 1)
                if aspect > 5 or aspect < 0.2:
                    continue
                
                detections.append((x, y, w_c, h_c))
            
            # Track blobs
            fishes = tracker.update(detections, motion_mask)
            
            # =========================
            # 🎯 INTELLIGENT TARGETING
            # =========================
            
            # Prioritize small fish for efficiency
            priority_targets = []
            
            for fish in fishes:
                # Multi-class classification
                class_info = classifier.get_class_info(fish.class_id)
                
                # Get OCR-learned multiplier
                multiplier, confidence = ocr_system.get_expected_multiplier(fish.class_id)
                
                # Estimate value
                base_value = 2 if fish.class_id < 2 else (5 if fish.class_id < 4 else 15)
                estimated_value = base_value * multiplier
                
                # Bullet conservation check
                can_fire = bullet_manager.should_fire(estimated_value, confidence * fire_aggression)
                
                if can_fire and fish.speed > 1:
                    priority_targets.append({
                        'fish': fish,
                        'value': estimated_value,
                        'confidence': confidence,
                        'class_name': class_info.get('name', f'Class{fish.class_id}')
                    })
            
            # Sort by value and fire top target
            if priority_targets:
                priority_targets.sort(key=lambda x: x['value'] * x['confidence'], reverse=True)
                target = priority_targets[0]['fish']
                
                # Calculate lead and fire
                lead_x = int(target.x + target.vx * 3)
                lead_y = int(target.y + target.vy * 3)
                
                screen_x = region["left"] + lead_x
                screen_y = region["top"] + lead_y
                
                pyautogui.click(screen_x, screen_y)
                bullet_manager.record_shot(priority_targets[0]['value'], 1.0, hit=True)
            
            # === VISUALIZATION ===
            vis = frame.copy()
            
            for i, fish in enumerate(fishes):
                # Draw box
                multiplier, conf = ocr_system.get_expected_multiplier(fish.class_id)
                color = (0, 255, 0) if conf > 0.6 else (100, 100, 100)
                cv2.rectangle(vis, (fish.x, fish.y), (fish.x + fish.w, fish.y + fish.h), color, 2)
                
                # Draw single center dot
                cx, cy = fish.x + fish.w // 2, fish.y + fish.h // 2
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw velocity vector
                if fish.speed > 2:
                    px = cx + int(fish.vx * 3)
                    py = cy + int(fish.vy * 3)
                    cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 0, 0), 2, tipLength=0.3)
                
                # Label with class and multiplier
                label = f"C:{fish.class_id} M:{multiplier:.1f}x Conf:{conf:.0%}"
                cv2.putText(vis, label, (fish.x, fish.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Stats
            fps = frame_count / (time.time() - start_time + 0.001)
            efficiency = bullet_manager.get_efficiency_ratio()
            agg_display = f"{'AGGRESSIVE' if fire_aggression > 0.7 else 'BALANCED' if fire_aggression > 0.3 else 'CONSERVATIVE'}"
            stats = f"FPS:{fps:.1f} | Targets:{len(fishes)} | Eff:{efficiency:.2f}x | {agg_display}"
            cv2.putText(vis, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            wm.show("VISUALIZATION", vis)
            
            # === STATS DISPLAY ===
            stats_display = np.zeros((600, 500, 3), dtype=np.uint8)
            
            y_offset = 30
            cv2.putText(stats_display, f"EFFICIENCY STATS", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 40
            
            cv2.putText(stats_display, f"Shots Fired: {bullet_manager.total_shots}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Total Gain: {bullet_manager.total_gain:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Total Cost: {bullet_manager.total_cost:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            cv2.putText(stats_display, f"Ratio: {efficiency:.2f}x", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if efficiency > 1.5 else (255, 100, 0), 2)
            y_offset += 40
            
            cv2.putText(stats_display, f"AGGRESSION: {fire_aggression:.0%}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 40
            cv2.putText(stats_display, f"Threshold: {bullet_manager.efficiency_threshold:.2f}x", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            wm.show("STATS", stats_display)
            
            # === RADAR ===
            radar_size = 300
            radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
            
            for fish in fishes:
                rx = int((fish.x / w) * radar_size)
                ry = int((fish.y / h) * radar_size)
                
                multiplier, conf = ocr_system.get_expected_multiplier(fish.class_id)
                radius = int(2 + conf * 8)  # Larger circle for high-confidence
                
                # Color by multiplier expectation
                if multiplier > 2.0:
                    color = (0, 0, 255)  # Red = high multiplier
                elif multiplier > 1.5:
                    color = (0, 255, 0)  # Green = medium multiplier
                else:
                    color = (255, 255, 0)  # Yellow = base
                
                cv2.circle(radar, (rx, ry), radius, color, -1)
            
            wm.show("RADAR DISPLAY", radar)
            
            # === INPUT HANDLING ===
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                fire_aggression = min(1.0, fire_aggression + 0.1)
                bullet_manager.efficiency_threshold = max(1.0, bullet_manager.efficiency_threshold - 0.1)
                print(f"📈 Aggression: {fire_aggression:.0%} | Threshold: {bullet_manager.efficiency_threshold:.2f}x")
            elif key == ord('-') or key == ord('_'):
                fire_aggression = max(0.0, fire_aggression - 0.1)
                bullet_manager.efficiency_threshold = min(3.0, bullet_manager.efficiency_threshold + 0.1)
                print(f"📉 Aggression: {fire_aggression:.0%} | Threshold: {bullet_manager.efficiency_threshold:.2f}x")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        cv2.destroyAllWindows()
        print(f"✅ Final Efficiency: {bullet_manager.get_efficiency_ratio():.2f}x")
        print("✅ System shutdown complete")


if __name__ == "__main__":
    main()