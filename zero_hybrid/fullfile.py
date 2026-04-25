import cv2, numpy as np, time, mss

print("HYBRID BLOB AI SYSTEM")

# =========================
# CONFIG
# =========================
SCALE = 0.3
RADAR_SIZE = 300

SHOT_COST = 1
SAFE_RESERVE = 5
STOP_LOSS = -50

BURST_DELAY = 0.03

# =========================
# INIT
# =========================
sct = mss.mss()
monitor = sct.monitors[1]

screen_w = monitor["width"]
screen_h = monitor["height"]

score_bank = 50
profit = 0

prev_frame = None

# =========================
# FISH CLASS
# =========================
class Blob:
    def __init__(self,x,y,area):
        self.x=x; self.y=y
        self.vx=0; self.vy=0
        self.area=area
        self.last_seen=time.time()
        self.shots=0

blobs=[]

# =========================
# TRACKING
# =========================
def update_blobs(detections):
    global blobs, score_bank, profit

    new=[]
    now=time.time()

    for cx,cy,area in detections:
        found=False
        for b in blobs:
            if abs(b.x-cx)<25 and abs(b.y-cy)<25:
                b.vx=cx-b.x
                b.vy=cy-b.y
                b.x=cx; b.y=cy
                b.area=area
                b.last_seen=now
                new.append(b)
                found=True
                break
        if not found:
            new.append(Blob(cx,cy,area))

    # kill detection
    for b in blobs:
        if (now - b.last_seen) > 0.4 and b.shots > 0:
            gain = estimate_value(b.area)
            score_bank += gain
            profit += gain
            b.shots = 0

    blobs = new
    return blobs

# =========================
# SIZE + VALUE
# =========================
def estimate_value(area):
    if area < 150: return 2
    elif area < 400: return 5
    else: return 15

def estimate_hp(area):
    if area < 150: return 1
    elif area < 400: return 3
    else: return 6

# =========================
# CLUSTER
# =========================
def cluster_info(b, bs):
    xs=ys=vxs=vys=count=0

    for o in bs:
        if abs(b.x-o.x)<70 and abs(b.y-o.y)<70:
            xs+=o.x; ys+=o.y
            vxs+=o.vx; vys+=o.vy
            count+=1

    if count==0:
        return b.x,b.y,b.vx,b.vy,1

    return xs//count, ys//count, vxs/count, vys/count, count

# =========================
# INTERCEPT
# =========================
def predict(x,y,vx,vy):
    return int(x+vx*4), int(y+vy*4)

# =========================
# BURST FIRE
# =========================
def burst(px,py,n):
    for i in range(n):
        shoot(px,py)
        if i<n-1:
            time.sleep(BURST_DELAY)

def shoot(x,y):
    pass  # hook your controller

# =========================
# RADAR
# =========================
def draw_radar(bs):
    radar = np.zeros((RADAR_SIZE,RADAR_SIZE,3), dtype=np.uint8)

    for b in bs:
        rx = int((b.x/screen_w)*RADAR_SIZE)
        ry = int((b.y/screen_h)*RADAR_SIZE)

        # 🎨 multi-color precision (more buckets)
        if b.area < 100: color=(0,255,0)
        elif b.area < 200: color=(50,255,0)
        elif b.area < 400: color=(0,255,255)
        elif b.area < 800: color=(0,128,255)
        else: color=(0,0,255)

        _,_,_,_,c = cluster_info(b, bs)
        r = 2 + min(c,4)

        cv2.circle(radar,(rx,ry),r,color,-1)

    cv2.imshow("RADAR", radar)

# =========================
# MAIN LOOP
# =========================
while True:

    if profit <= STOP_LOSS:
        print("STOP LOSS")
        break

    frame = np.array(sct.grab(monitor))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    small = cv2.resize(gray,(0,0),fx=SCALE,fy=SCALE)

    if prev_frame is None:
        prev_frame = small
        continue

    diff = cv2.absdiff(prev_frame, small)
    _,th = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections=[]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue

        x,y,w,h = cv2.boundingRect(c)

        # 🔥 multi-point precision sampling
        cx = x + w//2
        cy = y + h//2

        detections.append((
            int(cx/SCALE),
            int(cy/SCALE),
            area
        ))

    prev_frame = small

    bs = update_blobs(detections)

    # prioritize clusters + small blobs
    bs_sorted = sorted(bs, key=lambda b: (b.area, -cluster_info(b,bs)[4]))

    for b in bs_sorted[:6]:

        cx,cy,vx,vy,c = cluster_info(b,bs)

        hp = estimate_hp(b.area)
        value = estimate_value(b.area)

        if value <= hp:
            continue

        if score_bank <= SAFE_RESERVE:
            continue

        px,py = predict(cx,cy,vx,vy)

        # 🔥 cluster burst logic
        if c >= 4:
            shots = 3
        elif c == 3:
            shots = 2
        elif c == 2:
            shots = 1
        else:
            shots = 0

        if shots <= 0:
            continue

        burst(px,py,shots)

        score_bank -= shots * SHOT_COST
        profit -= shots * SHOT_COST

        b.shots += shots

    draw_radar(bs)

    if cv2.waitKey(1) & 0xFF == 27:
        break