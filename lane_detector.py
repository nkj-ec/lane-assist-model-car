import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # We assume a standard 640x480 resolution input from the cam
        self.width = 640
        self.height = 480
        pass
        
    def process(self, frame):
        '''
        Processes the frame, detects lanes, and returns a steering logic
        value (-1.0 to 1.0) and the annotated frame.
        '''
        # 1. Grayscale & Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 3. Create Region of Interest
        # We only care about the lower half of the road
        mask = np.zeros_like(edges)
        h, w = edges.shape
        polygon = np.array([[
            (0, h),
            (w, h),
            (w, int(h * 0.5)),
            (0, int(h * 0.5))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        
        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(
            cropped_edges, 1, np.pi/180, 
            threshold=50, maxLineGap=20, minLineLength=40
        )
        
        steering_offset = 0.0 # -1.0 is full left, 1.0 is full right
        
        if lines is not None:
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Avoid divide by zero
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Classify by slope
                if slope < -0.3: # Left lane
                    left_lines.append((slope, intercept))
                elif slope > 0.3: # Right lane
                    right_lines.append((slope, intercept))
            
            # Average the lines
            left_avg = np.average(left_lines, axis=0) if left_lines else None
            right_avg = np.average(right_lines, axis=0) if right_lines else None
            
            mid_x = w // 2
            y_eval = h # We evaluate the difference at the very bottom of the screen
            
            # 5. Calculate steering offset
            if left_avg is not None and right_avg is not None:
                # Both lanes detected
                left_x = (y_eval - left_avg[1]) / left_avg[0]
                right_x = (y_eval - right_avg[1]) / right_avg[0]
                center_of_lane = (left_x + right_x) / 2
                pixel_offset = center_of_lane - mid_x
                steering_offset = pixel_offset / (w / 2)
            
            elif left_avg is not None:
                # Only left lane detected, estimate by adding lane width
                left_x = (y_eval - left_avg[1]) / left_avg[0]
                center_of_lane = left_x + (w // 3) # Assumption of lane width
                pixel_offset = center_of_lane - mid_x
                steering_offset = pixel_offset / (w / 2)
                
            elif right_avg is not None:
                # Only right lane detected, estimate
                right_x = (y_eval - right_avg[1]) / right_avg[0]
                center_of_lane = right_x - (w // 3)
                pixel_offset = center_of_lane - mid_x
                steering_offset = pixel_offset / (w / 2)
        
        # Cap the steering offset at [-1, 1]
        steering_offset = max(min(steering_offset, 1.0), -1.0)
        
        return steering_offset, cropped_edges
