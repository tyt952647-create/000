import cv2
import numpy as np
import mss
import time
import pyautogui
from collections import deque

from enhanced_ai import (
    FishMemory, ShotTracker, AdaptiveStrategy,
    ObservationLearner, GameClassifier, StatePersistence
)
from priority_selector import PrioritySelector

# [Previous classes: WindowManager, get_chrome_region, MotionDetector, Fish, BlobTracker]
# Keep all as before...

class WindowManager:
    def __init__(self):
        self.windows = {}
        screen_w, screen_h = pyautogui.size()
        self.screen_w = screen_w
        self.screen_h = screen_h
    
    def create(self, name, x, y, width=500, height=500):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        cv2.moveWindow(name, x, y)
        self.windows[name] = (x, y, width, height)
    
    def show(self, name, image):
        if image is not None and image.size > 0:
            cv2.imshow(name, image)

def get_chrome_region():
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
            return {"left": left + width // 2, "top": top, "width": width // 2, "height": height}
    except:
        pass
    sct = mss.mss()
    monitor = sct.monitors[1]
    return {"left": monitor["width"] // 2, "top": 0, "width": monitor["width"] // 2, "height": monitor["height"]}

class MotionDetector:
    def __init__(self, width, height, adapt_rate=5):
        self.width = width
        self.height = height
        self.adapt_rate = adapt_rate
        self.prev_frame = None
        self.motion_history = deque(maxlen=60)
        self.base_threshold = 15
        self.target_brightness = 127
    
    def update_adapt_rate(self, rate):
        self.adapt_rate = max(0.1, rate)
    
    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray.copy()
        _, motion_mask = cv2.threshold(diff, self.base_threshold, 255, cv2.THRESH_BINARY)
        self.motion_history.append(np.count_nonzero(motion_mask))
        return motion_mask
    
    def detect_oscillation(self):
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
                    return True
        return False
    
    def adapt_threshold(self, motion_mask):
        motion_pixels = motion_mask > 0
        if np.any(motion_pixels):
            brightness = np.mean(self.prev_frame[motion_pixels])
            delta = (self.target_brightness - brightness) * (self.adapt_rate / 100.0)
            self.base_threshold = max(5, self.base_threshold + delta)

class Fish:
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
        if self.area < 100: return 0
        elif self.area < 300: return 1
        elif self.area < 600: return 2
        elif self.area < 1200: return 3
        else: return 4
    
    def update(self, x, y, w, h):
        self.vx = x - self.x
        self.vy = y - self.y
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        if self.speed > 0:
            self.heading = np.arctan2(self.vy, self.vx) * 180 / np.pi
        self.x, self.y, self.w, self.h = x, y, w, h
        self.area = w * h
        self.class_id = self._classify()
        self.history.append((x, y))
        self.age += 1
        self.last_seen = time.time()
    
    def is_same_trajectory(self, other, dist_thresh=30, angle_thresh=15, speed_thresh=0.5):
        dist = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if dist > dist_thresh: return False
        if self.speed < 2 or other.speed < 2: return dist < dist_thresh * 0.5
        angle_diff = abs(self.heading - other.heading)
        if angle_diff > 180: angle_diff = 360 - angle_diff
        if angle_diff > angle_thresh: return False
        speed_ratio = min(self.speed, other.speed) / max(self.speed, other.speed, 0.1)
        return speed_ratio >= (1.0 - speed_thresh)

class BlobTracker:
    def __init__(self):
        self.fishes = []
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections, motion_mask):
        self.frame_count += 1
        valid_detections = []
        for x, y, w, h in detections:
            if w < 3 or h < 3: continue
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
                self.fishes.append(Fish(x, y, w, h, self.next_id))
                self.next_id += 1
        
        self.fishes = [f for f in self.fishes if f.id in matched or f.age < 5]
        self._merge_trajectories()
        return self.fishes
    
    def _merge_trajectories(self):
        merged = []
        used = set()
        for i, f1 in enumerate(self.fishes):
            if f1.id in used: continue
            group = [f1]
            used.add(f1.id)
            for j, f2 in enumerate(self.fishes):
                if j <= i or f2.id in used: continue
                if f1.is_same_trajectory(f2):
                    group.append(f2)
                    used.add(f2.id)
            group = sorted(group, key=lambda f: f.area, reverse=True)
            merged.append(group[0])
        self.fishes = merged

def main():
    print("🎮 HYBRID FISH DETECTION + BALANCED AI v6.5")
    print("=" * 70)
    print("✅ BALANCED STRATEGIC FIRING")
    print("   • Fire with CONFIDENCE on proven patterns")
    print("   • 65%+ historical success = FIRE without hesitation")
    print("   • Avoid random waste - every shot must have purpose")
    print("   • 30s mandatory: Pick best high-confidence target only")
    print("=" * 70)
    
    region = get_chrome_region()
    sct = mss.mss()
    wm = WindowManager()
    motion_detector = MotionDetector(region["width"], region["height"])
    tracker = BlobTracker()
    
    fish_memory = FishMemory('fish_memory.json')
    shot_tracker = ShotTracker('shot_history.json')
    strategy = AdaptiveStrategy(initial_shot_cost=1)
    priority_selector = PrioritySelector(fish_memory, shot_tracker)
    game_classifier = GameClassifier('game_profiles.json')
    persistence = StatePersistence('system_state.json')
    
    initial_score = persistence.get_score_state()
    current_score = initial_score
    
    wm.create("VISUALIZATION", 50, 50, 900, 650)
    wm.create("MASK CONTROLS", 1000, 50, 500, 650)
    wm.create("RADAR", 50, 750, 400, 400)
    wm.create("STATS", 500, 750, 500, 400)
    
    frame_count = 0
    start_time = time.time()
    shots_fired = 0
    last_target = None
    
    try:
        while True:
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            if frame is None or frame.size == 0:
                time.sleep(0.1)
                continue
            
            h, w = frame.shape[:2]
            
            motion_mask = motion_detector.detect_motion(frame)
            if motion_mask is None or motion_mask.size == 0:
                continue
            
            if motion_detector.detect_oscillation():
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            motion_detector.adapt_threshold(motion_mask)
            
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for c in contours:
                x, y, w_rect, h_rect = cv2.boundingRect(c)
                area = w_rect * h_rect
                if area < 20: continue
                aspect = w_rect / max(h_rect, 1)
                if aspect > 5 or aspect < 0.2: continue
                detections.append((x, y, w_rect, h_rect))
            
            fishes = tracker.update(detections, motion_mask)
            
            if fishes:
                ranked_targets, aggression, must_fire = priority_selector.rank_targets(
                    fishes, region, current_score
                )
                
                if ranked_targets and strategy.should_fire(current_score):
                    target = ranked_targets[0]
                    fish = target['fish']
                    burst_size = target['burst_size']
                    lead_x, lead_y = target['burst_target']
                    expected_gain = target['expected_gain']
                    confidence = target['confidence']
                    target_type = target['type']
                    
                    for burst_idx in range(burst_size):
                        shot_cost = strategy.adjust_shot_cost(burst_idx, burst_size)
                        pyautogui.click(region["left"] + lead_x, region["top"] + lead_y)
                        shots_fired += 1
                        current_score -= shot_cost
                        
                        shot_tracker.record_shot(
                            fish.id, fish.class_id, fish.area,
                            fish.x, fish.y, fish.vx, fish.vy, 'shot'
                        )
                        
                        if burst_idx < burst_size - 1:
                            time.sleep(0.03)
                    
                    strategy.reset_burst()
                    priority_selector.record_shot_fired(expected_gain, burst_size)
                    last_target = fish
                    
                    gain_str = f"+{expected_gain:.1f}" if expected_gain > 0 else f"{expected_gain:.1f}"
                    print(f"🎯 T{target['priority_tier']} [{target_type}] "
                          f"Class {fish.class_id} | Conf:{confidence:.0%} | "
                          f"Gain:{gain_str} | Shots:{shots_fired} | Score:{current_score}")
            
            # === VISUALIZATION ===
            vis = frame.copy()
            for fish in fishes:
                color = (0, 255, 0) if fish.speed > 2 else (100, 100, 100)
                cv2.rectangle(vis, (fish.x, fish.y), (fish.x + fish.w, fish.y + fish.h), color, 2)
                cx, cy = fish.x + fish.w // 2, fish.y + fish.h // 2
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                if fish.speed > 2:
                    px = cx + int(fish.vx * 5)
                    py = cy + int(fish.vy * 5)
                    cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 0, 0), 2, tipLength=0.3)
                cv2.putText(vis, f"ID:{fish.id} C:{fish.class_id} S:{fish.speed:.1f}", 
                           (fish.x, fish.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            fps = frame_count / (time.time() - start_time + 0.001)
            cv2.putText(vis, f"FPS:{fps:.1f} | Targets:{len(fishes)} | Score:{current_score} | Shots:{shots_fired}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            wm.show("VISUALIZATION", vis)
            
            # === MASK ===
            mask_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_display, f"Threshold: {motion_detector.base_threshold:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            wm.show("MASK CONTROLS", mask_display)
            
            # === RADAR ===
            radar = np.zeros((300, 300, 3), dtype=np.uint8)
            for fish in fishes:
                rx = int((fish.x / w) * 300)
                ry = int((fish.y / h) * 300)
                colors = [(0, 255, 0), (50, 255, 0), (0, 255, 255), (0, 128, 255), (0, 0, 255)]
                cv2.circle(radar, (rx, ry), 5, colors[min(fish.class_id, 4)], -1)
            wm.show("RADAR DISPLAY", radar)
            
            # === STATS ===
            stats_img = np.zeros((400, 500, 3), dtype=np.uint8)
            stats_img[:] = (30, 30, 30)
            y = 30
            cv2.putText(stats_img, "BALANCED FIRING SYSTEM", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            y += 35
            cv2.putText(stats_img, f"Score: {current_score:.1f} ({current_score - initial_score:+.1f})", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 25
            cv2.putText(stats_img, f"Shots: {shots_fired}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 25
            
            urgency = priority_selector.get_fire_urgency()
            time_since = time.time() - priority_selector.last_shot_time
            urgency_color = [(0, 255, 0), (0, 165, 255), (0, 0, 255)][urgency]
            urgency_text = ["NORMAL", "PREP", "MUST FIRE"][urgency]
            cv2.putText(stats_img, f"Time Since Fire: {int(time_since)}s | {urgency_text}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, urgency_color, 2)
            y += 30
            
            stats = priority_selector.get_stats()
            cv2.putText(stats_img, f"Efficiency: {stats['efficiency']:.2f} pts/shot | Waste: {stats['waste_rate']:.0%}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y += 25
            cv2.putText(stats_img, f"Memory: {len(fish_memory.fish_classes)} | History: {len(shot_tracker.session_shots)}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
            wm.show("AI STATS", stats_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped")
    finally:
        persistence.save_state({'score': current_score, 'shots_fired': shots_fired})
        fish_memory.save()
        shot_tracker.save()
        cv2.destroyAllWindows()
        
        stats = priority_selector.get_stats()
        print("\n" + "=" * 70)
        print(f"📊 Score: {current_score} ({current_score:.1f} ({current_score - initial_score:+.1f})")
        print(f"🎯 Shots: {shots_fired}")
        print(f"💰 Efficiency: {stats['efficiency']:.2f} pts/shot")
        print(f"🚫 Waste Rate: {stats['waste_rate']:.1%}")
        print("=" * 70)

if __name__ == "__main__":
    main()