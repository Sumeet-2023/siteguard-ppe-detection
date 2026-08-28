import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11s")
ap.add_argument("--data", default="configs/data_shwd.yaml")
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

YOLO(f"{a.model}.pt").train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    project="runs/ppe", name=a.model,
    seed=1337, deterministic=True, patience=20, cos_lr=True,
    # augmentation tuned for this domain
    hsv_v=0.5,          # worksites have brutal lighting variation
    degrees=5.0,        # heads are upright; big rotations are unrealistic
    scale=0.5,          # heads appear at wildly different distances
    fliplr=0.5,
    flipud=0.0,         # never vertical-flip a construction site
    close_mosaic=10,    # disable mosaic for the last 10 epochs
    copy_paste=0.1,     # cheap help for the minority helmet class
)
