# INF2009-P1-T02
## Installation

Get mediapipe installed on terminal

```bash
pip install mediapipe
```
Download the poselandmark detection model on VSC terminal:

```bash
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile "pose_landmarker.task"
```
To download the poselandmark detection model on RPI: 
```bash
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```
