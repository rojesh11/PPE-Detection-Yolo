# Object Detection GUI

This repository contains a Python script that implements an Object Detection GUI using Tkinter and OpenCV. The GUI allows users to perform object detection on video files or live camera feed using the YOLO (You Only Look Once) object detection model.

## Features

- Select a video file for object detection.
- Start live object detection from the camera.
- Display real-time object detection results with bounding boxes and labels.
- Adjust confidence threshold for filtering detections.

## Dependencies

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Tkinter (usually comes pre-installed with Python)
- Ultralytics YOLO (installation instructions [here](https://github.com/ultralytics/yolov5))

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Object-Detection-GUI.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Object-Detection-GUI
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the script:

    ```bash
    python object_detection_gui.py
    ```

5. In the GUI window, you can:
    - Click on "Select Video for Detection" to choose a video file for object detection.
    - Click on "Start Live Detection" to begin real-time object detection from the camera.
    - Adjust the confidence threshold as needed.

6. Press 'q' or 'Esc' key to exit the application.

## Acknowledgments

- This project utilizes the YOLO object detection model implemented by Ultralytics. Check out their GitHub repository: [Ultralytics YOLO](https://github.com/ultralytics/yolov5).

---

You can customize this README according to your preferences and add any additional information or instructions as needed.
