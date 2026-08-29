"""Run the trained model over the test set, score each image by how badly its
predictions disagree with ground truth (missed boxes, extra boxes, wrong
class), and save the worst N as annotated images for manual captioning.

Not a formal metric -- just a fast way to surface genuine failure cases
instead of eyeballing random validation-batch mosaics.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_gt(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, w, h = (float(v) for v in line.split())
        x1, y1 = (xc - w / 2) * img_w, (yc - h / 2) * img_h
        x2, y2 = (xc + w / 2) * img_w, (yc + h / 2) * img_h
        boxes.append((int(cls_id), np.array([x1, y1, x2, y2])))
    return boxes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1 = max(a[0], b[0]), max(a[1], b[1])
    xa2, ya2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def score_image(gt: list[tuple[int, np.ndarray]], preds, iou_thresh=0.5) -> tuple[int, int, int]:
    """Returns (missed, extra, wrong_class) counts via greedy IoU matching."""
    matched_gt: set[int] = set()
    pred_to_gt: dict[int, int] = {}
    for pi, (pcls, pbox) in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, (gcls, gbox) in enumerate(gt):
            if gi in matched_gt:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_thresh:
            matched_gt.add(best_gi)
            pred_to_gt[pi] = best_gi

    wrong_class = sum(
        1 for pi, (pcls, pbox) in enumerate(preds)
        if pi in pred_to_gt and pcls != gt[pred_to_gt[pi]][0]
    )
    missed = len(gt) - len(matched_gt)
    extra = len(preds) - len(pred_to_gt)
    return missed, extra, wrong_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("runs/ppe/yolo11s/weights/best.pt"))
    ap.add_argument("--data-root", type=Path, default=Path("data/shwd_final"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/failure_cases"))
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    model = YOLO(str(a.weights))
    img_dir = a.data_root / "images" / a.split
    label_dir = a.data_root / "labels" / a.split

    scored = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = load_gt(label_dir / f"{img_path.stem}.txt", w, h)

        res = model.predict(img, imgsz=640, conf=a.conf, verbose=False)[0]
        preds = [(int(c), box) for box, c in
                 zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())]

        missed, extra, wrong = score_image(gt, preds)
        badness = 2 * missed + extra + 2 * wrong
        if badness > 0:
            scored.append((badness, missed, extra, wrong, img_path, res))

    scored.sort(key=lambda r: -r[0])
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(scored)}/{len(list(img_dir.glob('*.jpg')))} test images had at least one error")
    print(f"Saving top {a.top_n} worst cases to {a.out_dir}\n")

    for rank, (badness, missed, extra, wrong, img_path, res) in enumerate(scored[:a.top_n]):
        out_path = a.out_dir / f"failcase_{rank:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), res.plot())
        print(f"{rank:2d}. {img_path.name}: missed={missed} extra={extra} "
              f"wrong_class={wrong} -> {out_path.name}")


if __name__ == "__main__":
    main()
