"""Zero-shot baseline: does the COCO-pretrained backbone already localise
people well, even without helmet semantics?
"""
from ultralytics import YOLO

if __name__ == "__main__":
    m = YOLO("yolo11s.pt").val(data="configs/data_shwd.yaml", split="test", classes=[0])
    print(f"COCO person-class mAP50 on our test set: {m.box.map50:.4f}")
