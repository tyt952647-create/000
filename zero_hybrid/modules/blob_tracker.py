import numpy as np

class Blob:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.cls = 0

def update_blobs(detections):
    blobs = []

    for d in detections:
        if len(d) < 4:
            continue

        x, y, w, h = d

        if w < 5 or h < 5:
            continue

        blobs.append(Blob(x, y, w, h))

    if not blobs:
        return [], [], [0]*5

    intersections = []

    for b in blobs:
        cx = b.x + b.w // 2
        cy = b.y + b.h // 2
        intersections.append((cx, cy))

    counts = [0]*5
    for b in blobs:
        counts[0] += 1

    return blobs, intersections, counts