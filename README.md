# Raspberry Pi 5 Autonomous Lane Assist Car

This project contains the Python code required to run a model car with a Raspberry Pi 5, an L298N motor driver, and a camera (Pi Camera V2/V3 or USB WebCam).

## Features
- **Motor Control**: Interfacing the L298N via `gpiozero` which is natively compatible with the Pi 5's RP1 I/O chip.
- **Lane Detection**: Canny Edge detection + Hough Line Transform using `OpenCV`.
- **Obstacle & Traffic Sign Detection**: Unified detection using an `ultralytics` YOLOv8n object detection model.

## Hardware Wiring
### L298N to Raspberry Pi 5
- **INT1**: GPIO 24  
- **INT2**: GPIO 23  
- **ENA**:  GPIO 25 (PWM capable)  
- **INT3**: GPIO 22  
- **INT4**: GPIO 27  
- **ENB**:  GPIO 17 (PWM capable)  

*(You can easily change these pins in `main.py` when initializing `Car()`)*

## Environment Setup

1. **Enable Camera Interfaces**: Ensure your camera is enabled via `sudo raspi-config` (Or automatically works dynamically on Pi 5 via `rpicam`/`libcamera`). We are using `Picamera2` (the official Python rpicam library) to capture frames natively.
   
2. **Install Dependencies**:
```bash
# Strongly recommended to create a virtual environment first
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```


3. **Install YOLOv8**:
   The code relies on the `ultralytics` package for object detection. Ensure it's installed via:
   ```bash
   pip install ultralytics
   ```
   > **Note**: The first time you run the script, `ultralytics` will automatically download the `yolov8n.pt` model weights (approx. 6MB).


## Running the Code
Run the code headless or via a terminal:
```bash
python main.py
```

> **Note**: Displaying `cv2.imshow()` on the Pi consumes CPU and decreases FPS. If operating entirely autonomously, it's recommended to leave it commented out in `main.py`.
