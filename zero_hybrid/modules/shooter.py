import pyautogui
import time

def burst_fire(x, y, shots=5):
    pyautogui.moveTo(x, y)

    for _ in range(shots):
        pyautogui.click()
        time.sleep(0.02)