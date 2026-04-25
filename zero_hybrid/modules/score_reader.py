import cv2
import pytesseract

REGION = (1100, 1020, 300, 120)  # adjust if needed

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150,255,cv2.THRESH_BINARY)
    return cv2.resize(th, None, fx=2, fy=2)

def get_player_score(frame):
    x,y,w,h = REGION
    crop = frame[y:y+h, x:x+w]

    if crop.size == 0:
        return 0

    proc = preprocess(crop)

    text = pytesseract.image_to_string(
        proc,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )

    try:
        return int("".join(filter(str.isdigit, text)))
    except:
        return 0