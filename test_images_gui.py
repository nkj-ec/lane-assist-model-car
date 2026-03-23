import cv2
import os
import sys

from lane_detector import LaneDetector
from yolo_detector import YoloDetector

def main():
    test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    
    if not os.path.exists(test_dir):
        print(f"Creating directory: {test_dir}")
        os.makedirs(test_dir)
        
    # List all image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No valid images found in '{test_dir}'.")
        print("Please add some (.jpg, .png, etc.) and run this script again.")
        return
        
    print(f"Found {len(image_files)} images in '{test_dir}'.")

    # Initialize detectors
    # Set use_birds_eye=True to see the lane warp as in main.py
    lane_detector = LaneDetector(use_birds_eye=True)
    yolo_detector = YoloDetector()

    print("Controls:")
    print("  'n' or any other key : Next image")
    print("  'q'                  : Quit")

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"Processing: {img_file}")
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Resize to 640x480 as expected by the detector
        frame = cv2.resize(frame, (640, 480))

        # Run YOLO detection
        sign, obstacle_detected, pedestrian_detected, annotated_frame = yolo_detector.detect(frame)
        
        # Run Lane detection
        steering_offset, final_frame = lane_detector.process(annotated_frame)
        
        # Overlay YOLO detections state
        status_text = []
        if pedestrian_detected:
            status_text.append("PEDESTRIAN")
        if obstacle_detected:
            status_text.append("OBSTACLE")
        if sign == "STOP":
            status_text.append("STOP SIGN")
            
        if status_text:
            text_str = " | ".join(status_text)
            # Base position for text
            text_pos = (20, 80)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size to draw a background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text_str, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(final_frame, 
                          (text_pos[0] - 5, text_pos[1] - text_height - 5), 
                          (text_pos[0] + text_width + 5, text_pos[1] + baseline + 5), 
                          (0, 0, 0), cv2.FILLED)
            
            # Draw text
            cv2.putText(final_frame, text_str, text_pos, font, font_scale, (0, 0, 255), thickness)

        cv2.imshow('Test Image Result', final_frame)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()
