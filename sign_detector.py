import cv2
import os

class SignDetector:
    def __init__(self, cascade_path='stop_data.xml'):
        '''
        Uses Haar Cascades to detect stop signs.
        You must download the Haar cascade for Stop Signs and provide it here.
        Example: https://github.com/opencv/opencv/blob/master/data/haarcascades/
        '''
        self.cascade = cv2.CascadeClassifier()
        if os.path.exists(cascade_path):
            self.cascade.load(cascade_path)
            print(f"Loaded Haarcascade from {cascade_path}")
        else:
            print(f"Warning: Cascade file {cascade_path} not found. Sign detection disabled.")

    def detect(self, frame):
        '''
        Returns 'STOP' if a stop sign is detected, else None.
        '''
        if self.cascade.empty():
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect signs
        signs = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(signs) > 0:
            # Check bounding box sizes, only trigger if it's large enough (close enough)
            for (x, y, w, h) in signs:
                if w > 50 and h > 50:
                    return "STOP"
        
        return None
