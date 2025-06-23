# CodeAlpha_ObjectDetectionTracking

This project performs real-time object detection and tracking using YOLO and SORT.

## Requirements
- OpenCV
- Numpy
- Pretrained YOLOv3 weights and config files
- coco.names file
- filterpy (for SORT)

## Steps
1. `1_video_input.py` — Basic webcam video capture.
2. `2_object_detection_yolo.py` — Object detection using YOLOv3.
3. `3_tracking_sort.py` — Integrate SORT tracker for object tracking (placeholder).

## Usage
Run the scripts in order. Download YOLO files from:
- yolov3.weights: https://pjreddie.com/media/files/yolov3.weights
- yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
- coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names
