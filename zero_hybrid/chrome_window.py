"""
Chrome window detector - finds and tracks Chrome window for adaptive screen capture
"""
import pyautogui
import mss
import numpy as np
import time

class ChromeWindowCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.last_check = 0
        self.check_interval = 1.0  # Check every 1 second
        self.monitor = None
        self.chrome_bounds = None
        self._find_chrome_window()
    
    def _find_chrome_window(self):
        """Find Chrome window - platform specific"""
        try:
            import pygetwindow as gw
            
            # Try to find Chrome window
            chrome_windows = gw.getWindowsWithTitle('Chrome')
            if chrome_windows:
                w = chrome_windows[0]
                self.chrome_bounds = {
                    'left': w.left,
                    'top': w.top,
                    'width': w.width,
                    'height': w.height
                }
                return True
        except:
            pass
        
        # Fallback: use primary monitor
        self.monitor = self.sct.monitors[1]
        return False
    
    def _get_right_half(self, bounds):
        """Extract right half of bounds"""
        return {
            'left': bounds['left'] + bounds['width'] // 2,
            'top': bounds['top'],
            'width': bounds['width'] // 2,
            'height': bounds['height']
        }
    
    def get_capture_region(self):
        """Get current capture region - right half only"""
        now = time.time()
        
        # Periodically refresh Chrome window position (handles move/resize)
        if now - self.last_check > self.check_interval:
            self._find_chrome_window()
            self.last_check = now
        
        bounds = self.chrome_bounds or self.monitor
        return self._get_right_half(bounds)
    
    def grab(self):
        """Grab right half of Chrome window"""
        region = self.get_capture_region()
        img = np.array(self.sct.grab(region))
        return img, region
