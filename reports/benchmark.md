# Benchmarks

## Phase 3 — Zero-shot COCO baseline

Ran `yolo11s.pt` (COCO-pretrained, untouched) against our honest test split (3,419 images,
8,916 ground-truth boxes), filtering to COCO class 0 (`person`).

| Metric | Value |
|---|---|
| mAP50 | 0.0045 |
| Precision | 0.0151 |
| Recall | 0.279 |

**Interpretation:** this is not "the backbone fails to localise people" — recall of 0.28 shows
COCO's person detector does find people in these images. The near-zero mAP50 is a geometry
mismatch: COCO `person` boxes are full-body, while SHWD ground truth boxes are tight head/helmet
regions. A full-body box essentially never reaches IoU ≥ 0.5 against a head-sized box, so mAP
collapses even though detection is "working." This is itself the finding motivating Phase 4:
PPE compliance needs boxes trained on head-region semantics, not repurposed person detection.

## Phase 4/5 — Trained models

*(pending — GPU training runs via `notebooks/colab_train.ipynb`, see README)*
