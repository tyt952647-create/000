import cv2
import numpy as np
import pyautogui

def burst_fire(x, y, shots=3):
    for _ in range(shots):
        pyautogui.click(x, y)


class Blob:
    """Enhanced blob with trajectory tracking"""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.cls = 0
        self.aspect_ratio = w / max(h, 1)
        self.compactness = (w * h) / (max(w, h) ** 2)
        self.confidence = 0.5
        self.vx = 0  # velocity x
        self.vy = 0  # velocity y
        self.blob_id = None
        self.last_seen = None


def merge_nearby_blobs(detections, merge_threshold=25):
    """Merge overlapping or very close detections to avoid duplicates"""
    if not detections:
        return []
    
    merged = []
    used = set()
    
    for i, d1 in enumerate(detections):
        if i in used:
            continue
        
        x1, y1, w1, h1 = d1
        cx1, cy1 = x1 + w1//2, y1 + h1//2
        
        # Find all blobs close to this one
        group = [i]
        for j, d2 in enumerate(detections):
            if j <= i or j in used:
                continue
            
            x2, y2, w2, h2 = d2
            cx2, cy2 = x2 + w2//2, y2 + h2//2
            
            # Check distance between centroids
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            
            if dist < merge_threshold:
                group.append(j)
        
        # Merge group
        xs = [detections[idx][0] for idx in group]
        ys = [detections[idx][1] for idx in group]
        ws = [detections[idx][2] for idx in group]
        hs = [detections[idx][3] for idx in group]
        
        merged_x = min(xs)
        merged_y = min(ys)
        merged_w = max([xs[k] + ws[k] for k in range(len(xs))]) - merged_x
        merged_h = max([ys[k] + hs[k] for k in range(len(ys))]) - merged_y
        
        merged.append((merged_x, merged_y, merged_w, merged_h))
        used.update(group)
    
    return merged


def classify_fish(blob):
    """Classify fish type based on morphological features and size"""
    area = blob.area
    aspect = blob.aspect_ratio
    compact = blob.compactness
    
    # Class 0: Tiny < 100 px²
    if area < 100:
        blob.cls = 0
        blob.confidence = 0.8
    # Class 1: Small 100-300 px² (elongated, streamlined)
    elif area < 300 and aspect > 1.2:
        blob.cls = 1
        blob.confidence = 0.75
    # Class 2: Medium 300-600 px² (balanced)
    elif area < 600:
        blob.cls = 2
        blob.confidence = 0.7
    # Class 3: Large 600-1200 px² (larger, elongated)
    elif area < 1200 and aspect > 0.8:
        blob.cls = 3
        blob.confidence = 0.75
    # Class 4: Huge > 1200 px²
    else:
        blob.cls = 4
        blob.confidence = 0.8


def update_blobs(detections):
    """Process detections and classify fish - motion-based only"""
    
    # First merge nearby detections to avoid duplicates
    detections = merge_nearby_blobs(detections, merge_threshold=20)
    
    blobs = []
    
    for d in detections:
        if len(d) < 4:
            continue
        
        x, y, w, h = d
        
        # Stricter size filtering
        if w < 5 or h < 5:
            continue
        
        # Reject extremely elongated or thin objects (noise)
        aspect = w / max(h, 1)
        if aspect > 5 or aspect < 0.2:
            continue
        
        blob = Blob(x, y, w, h)
        classify_fish(blob)
        blobs.append(blob)
    
    if not blobs:
        return [], [], [0, 0, 0, 0, 0]
    
    # Find single best intersection point per distinct fish
    intersections = []
    
    # Group blobs by proximity
    blob_groups = []
    used = set()
    
    for i, b1 in enumerate(blobs):
        if i in used:
            continue
        
        group = [b1]
        used.add(i)
        
        for j, b2 in enumerate(blobs):
            if j <= i or j in used:
                continue
            
            # Only group blobs of same or adjacent classes
            if abs(b1.cls - b2.cls) <= 1:
                dist = np.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2)
                
                # Adaptive distance based on fish size
                max_dist = min(b1.area, b2.area) ** 0.5 * 1.5
                
                if dist < max_dist:
                    group.append(b2)
                    used.add(j)
        
        blob_groups.append(group)
    
    # Generate one intersection point per fish group (centroid)
    for group in blob_groups:
        if len(group) > 0:
            cx = sum(b.x + b.w//2 for b in group) // len(group)
            cy = sum(b.y + b.h//2 for b in group) // len(group)
            intersections.append((cx, cy))
    
    # Count by class
    counts = [0] * 5
    for b in blobs:
        counts[b.cls] += 1
    
    return blobs, intersections, counts
