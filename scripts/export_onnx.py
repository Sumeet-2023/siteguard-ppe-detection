"""Export the trained yolo11s checkpoint to ONNX FP32."""
import argparse
from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("runs/ppe/yolo11s/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    YOLO(str(a.weights)).export(
        format="onnx", imgsz=a.imgsz, opset=13, simplify=True, dynamic=False,
    )
