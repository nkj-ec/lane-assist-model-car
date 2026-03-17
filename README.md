# Raspberry Pi 5 Autonomous Lane Assist Car

This project contains the Python code required to run a model car with a Raspberry Pi 5, an L298N motor driver, and a camera (Pi Camera V2/V3 or USB WebCam).

## Features
- **Motor Control**: Interfacing the L298N via `gpiozero` which is natively compatible with the Pi 5's RP1 I/O chip.
- **Lane Detection**: Morphological transformations (Top-Hat/Black-Hat) + Polynomial sliding-window fit using `OpenCV`.
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
On modern Raspberry Pi OS (Bookworm and later), the system protects the global Python environment. Running a standard `pip install` will result in an "externally managed environment" error.

You have three options to install the required libraries:

**Option A - The easiest way (forcing pip):**
You can bypass the environment protection by adding `--break-system-packages`. Note that `opencv-python` is required instead of `opencv-python-headless` so you can view the GUI windows:
```bash
pip install -r requirements.txt --break-system-packages
```

**Option B - Using apt (Safest for system packages):**
Install the core libraries using the OS package manager, and only use pip for `ultralytics`.
```bash
sudo apt update
sudo apt install python3-opencv python3-numpy python3-gpiozero python3-lgpio
pip install ultralytics --break-system-packages
```

**Option C - Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Install YOLOv8**:
If you didn't install `ultralytics` in the previous step, do so now:
```bash
pip install ultralytics --break-system-packages
```
> **Note**: The first time you run the script, `ultralytics` will automatically download the lightweight `yolov8n.pt` model weights (approx. 6MB).

## Configuration
### Bird's Eye View
The lane detector supports two perspective modes, toggled via `use_birds_eye` in `main.py`:
- **`True`** (default): Warps the camera feed into a top-down view before detecting lanes. This gives better results on straight and gently curving tracks.
- **`False`**: Processes the image from the camera's native perspective. Useful if the warp distorts your specific track layout.

## Running the Code
Run the code headless or via a terminal:
```bash
python main.py
```

> **Note**: Displaying `cv2.imshow()` on the Pi consumes CPU and decreases FPS. If operating entirely autonomously, it's recommended to leave it commented out in `main.py`.
