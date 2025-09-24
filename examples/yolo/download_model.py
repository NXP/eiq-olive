from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')
model.export(format='onnx')
os.remove('yolov8n.pt')

