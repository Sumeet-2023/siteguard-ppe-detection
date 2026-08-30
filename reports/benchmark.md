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

Trained on Colab (T4 GPU), 80 epochs, imgsz=640, on the honest pHash split (15,952 train /
3,418 val / 3,419 test, 5.4% duplicates removed). Evaluated locally on CPU (this machine has no
GPU) against the held-out test split — accuracy numbers are unaffected by device, latency numbers
are CPU-only.

| Model | Size (MB) | mAP50 | mAP50-95 | AP50 helmet | AP50 head | CPU p50 (ms) | CPU FPS |
|---|---|---|---|---|---|---|---|
| yolo11n | 5.5 | 0.9487 | 0.5795 | 0.9665 | 0.9309 | 82.59 | 10.1 |
| yolo11s | 19.2 | 0.9619 | 0.6000 | 0.9738 | 0.9500 | 205.22 | 4.8 |

Both models comfortably outperform the zero-shot baseline (as expected — they were trained for
this task), and per-class AP shows the imbalance effect predicted in Phase 4: `head` (the majority
class) still trails `helmet` on both models, though `copy_paste` augmentation kept the gap modest
(~3.5pt on yolo11n, ~2.4pt on yolo11s) rather than the more severe ~19pt gap the original SHWD's
12:1 imbalance would suggest — this mirror's dataset is less extreme (~2:1 head:helmet).

yolo11s wins on every accuracy metric but is 2.5x slower on CPU (205ms vs 83ms) and 3.5x larger.
For a CPU-only edge deployment, yolo11n's speed/accuracy tradeoff is compelling — it's within
1.3pt mAP50 of yolo11s at less than half the latency.

**Note:** `mAP50` here is on unseen `test` images, computed with `model.val()`; it differs slightly
from `results.csv`'s in-training `val` split numbers (0.950/0.961) because those were measured
against the validation split during/after training, not the held-out test split.

## Phase 7 — PyTorch vs ONNX (yolo11s)

Same weights, same test split, PyTorch checkpoint vs its FP32 ONNX export (opset 13,
simplified), both on CPU.

| Variant | Size (MB) | mAP50 | mAP50-95 | CPU p50 (ms) | CPU FPS |
|---|---|---|---|---|---|
| PyTorch | 19.2 | 0.9619 | 0.6000 | 185.62 | 5.4 |
| ONNX FP32 | 37.9 | 0.9595 | 0.5929 | 146.10 | 6.8 |

~27% CPU latency reduction (186ms → 146ms) for a 0.24pt mAP50 drop — a real, close-to-free
deployment win, though the ONNX file itself is ~2x larger on disk (37.9 vs 19.2 MB) since it isn't
using any of PyTorch's weight compression. `models/best.onnx` (used by the FastAPI service) is
this ONNX export.

## Phase 6 — Cross-dataset generalisation (SH17, zero-shot)

`yolo11s` (trained only on SHWD, no fine-tuning) evaluated against the entire SH17 dataset
(8,099 images) — 17 classes remapped down to our `{helmet, head}` scheme via
`scripts/prepare_cross_dataset.py`, per the maintainer's own `sh17.yaml`
(github.com/ahmadmughees/SH17dataset), verified against instance counts on the actual download
(927 helmet + 11,985 head = 12,912 boxes retained; 63,082 dropped from the other 15 classes).
1,565/8,099 images end up with zero helmet/head boxes after remapping (pure negatives).

| Train → Test | mAP50 | mAP50-95 | AP50 helmet | AP50 head |
|---|---|---|---|---|
| SHWD → SHWD (in-domain) | 0.9619 | 0.6000 | 0.9738 | 0.9500 |
| SHWD → SH17 (cross-domain, zero-shot) | 0.5241 | 0.2702 | 0.2850 | 0.7631 |

A 43.8pt mAP50 drop crossing domains — this is the expected result, not a failure of the model.
Notably the drop is not uniform across classes: `head` holds up reasonably (0.95 → 0.76, -19pt)
since "a person's head" generalises across photo styles, but `helmet` collapses (0.97 → 0.29,
-68.8pt). SH17's images (Pexels stock photography across many industries — farming, firefighting,
medical, general labor) show far more headwear variety than SHWD's construction-site-focused
"hard hat or nothing" framing, so the model's helmet detector — tuned to SHWD's specific hard-hat
appearance — doesn't transfer to whatever counts as "helmet" in SH17's broader label set. This
matches the pattern already seen in the failure-case analysis (Phase 5): the model leans on
color/shape cues for "helmet" that don't generalise well outside its training distribution.

## Phase 6 — "Own photos" substitute set (web-sourced, not phone-shot)

**Read the caveat before the number.** The project spec wants 60-80 personally-shot phone photos
with independently-verified ground truth. No camera/site access was available here, so this is a
much smaller substitute: 9 CC-licensed photos from Wikimedia Commons (chosen over Pexels
specifically, since SH17 already draws from Pexels), labeled via **model-assisted** review — I
verified/corrected the model's own draft predictions rather than annotating from scratch. That
means any head the model missed outright isn't in the ground truth at all, so **recall/mAP here is
likely optimistic** and not directly comparable to the SHWD or SH17 numbers above, which used
independently-sourced ground truth. Full methodology, licensing, and per-image notes:
`reports/own_photos/README.md`.

| Metric | Value |
|---|---|
| Images / boxes | 9 / 19 (16 helmet, 3 head) |
| mAP50 | 0.8539 |
| mAP50-95 | 0.8183 |
| Precision | 0.9603 |
| Recall | 0.8330 |
| AP50 helmet | 0.9438 |
| AP50 head | 0.7640 |

What's actually trustworthy from this exercise (the precision-side findings, not the headline
mAP): the review caught 2 false positives (a fan and a glove, both misfired as "helmet" at low
confidence) and 1 real misclassification — a soft yellow cloth cap on a 1940s factory worker,
called "helmet" at 0.89 confidence. That's the same color/shape over-generalization already found
in Phase 5's failure cases and Phase 6's SH17 eval, now showing up a third time on genuinely novel
images. Three independent pieces of evidence for the same finding is a real pattern, not noise.

## Naive vs honest split

*(pending — would require retraining on a random image-level split for comparison; not yet run)*
