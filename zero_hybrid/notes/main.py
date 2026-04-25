import cv2
import numpy as np
import mss
import time
import pyautogui
from collections import deque
from pathlib import Path

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
        """Classify fish by size"""
        if self.area < 100:
            return 0
        elif self.area < 300:
            return 1
        elif self.area < 600:
            return 2
        elif self.area < 1200:
            return 3
        else:
            return 4
    
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
    print("🎮 HYBRID FISH DETECTION SYSTEM v3.0")
    print("=" * 50)
    
    # Setup
    region = get_chrome_region()
    print(f"📍 Capturing from: {region}")
    
    sct = mss.mss()
    wm = WindowManager()
    motion_detector = MotionDetector(region["width"], region["height"])
    tracker = BlobTracker()
    
    # Create windows at specific positions
    wm.create("VISUALIZATION", 50, 50, 800, 600)
    wm.create("MASK CONTROLS", 900, 50, 500, 600)
    wm.create("RADAR DISPLAY", 50, 700, 400, 400)
    
    print("🎯 Motion-based detection active")
    print("⚙️ Auto-tuning enabled")
    print("🔴 Press Q to quit")
    print("=" * 50)
    
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
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Adapt threshold
            motion_detector.adapt_threshold(motion_mask)
            
            # Find contours in motion mask
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                
                # Filter by size and aspect ratio
                if area < 20:
                    continue
                
                aspect = w / max(h, 1)
                if aspect > 5 or aspect < 0.2:
                    continue
                
                detections.append((x, y, w, h))
            
            # Track blobs
            fishes = tracker.update(detections, motion_mask)
          
            # === VISUALIZATION ===
            vis = frame.copy()
            
            for fish in fishes:
                # Draw box
                color = (0, 255, 0) if fish.speed > 2 else (100, 100, 100)
                cv2.rectangle(vis, (fish.x, fish.y), (fish.x + fish.w, fish.y + fish.h), color, 2)
                
                # Draw single center dot
                cx, cy = fish.x + fish.w // 2, fish.y + fish.h // 2
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw velocity vector
                if fish.speed > 2:
                    px = cx + int(fish.vx * 3)
                    py = cy + int(fish.vy * 3)
                    cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 0, 0), 2, tipLength=0.3)
                
                # Label
                label = f"ID:{fish.id} C:{fish.class_id} S:{fish.speed:.1f}"
                cv2.putText(vis, label, (fish.x, fish.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Stats
            fps = frame_count / (time.time() - start_time + 0.001)
            stats = f"FPS:{fps:.1f} | Targets:{len(fishes)} | Motion:{np.count_nonzero(motion_mask)//100}% | Adapt:{motion_detector.base_threshold:.1f}"
            cv2.putText(vis, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            wm.show("VISUALIZATION", vis)
            
            # === MASK CONTROLS ===
            mask_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_display, f"Thresh: {motion_detector.base_threshold:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_display, f"Motion Pixels: {np.count_nonzero(motion_mask)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_display, f"Oscillating: {'YES' if is_oscillating else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_oscillating else (0, 255, 0), 2)
            
            wm.show("MASK CONTROLS", mask_display)
            
            # === RADAR ===
            radar_size = 300
            radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
            
            for fish in fishes:
                rx = int((fish.x / w) * radar_size)
                ry = int((fish.y / h) * radar_size)
                
                # Color by class
                colors = [(0, 255, 0), (50, 255, 0), (0, 255, 255), (0, 128, 255), (0, 0, 255)]
                color = colors[min(fish.class_id, 4)]
                
                cv2.circle(radar, (rx, ry), 5, color, -1)
            
            wm.show("RADAR DISPLAY", radar)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                motion_detector.update_adapt_rate(motion_detector.adapt_rate + 1)
                print(f"📈 Adapt rate: {motion_detector.adapt_rate}%")
            elif key == ord('-') or key == ord('_'):
                motion_detector.update_adapt_rate(motion_detector.adapt_rate - 1)
                print(f"📉 Adapt rate: {motion_detector.adapt_rate}%")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")


if __name__ == "__main__":
    main()