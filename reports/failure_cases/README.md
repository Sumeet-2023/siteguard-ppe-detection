# Failure cases

Selected automatically by `scripts/find_failure_cases.py`: ran `yolo11s` (conf=0.25) over the
full test split (3,419 images), greedily IoU-matched predictions to ground truth, and ranked
images by a weighted count of missed/extra/misclassified boxes. 1,202/3,419 test images (35%) had
at least one error at this confidence threshold — these are the 8 worst.

| # | Image | missed | extra | wrong class | Caption |
|---|---|---|---|---|---|
| 0 | `failcase_00_helm_006285.jpg` | 38 | 6 | 0 | Large auditorium crowd shot from a distance — most of the missed boxes are heads under ~15px, well below what a 640px-input model resolves reliably. Classic small-object failure. |
| 1 | `failcase_01_helm_003937.jpg` | 13 | 26 | 0 | Classroom CCTV, tightly packed seated rows — the model fires multiple overlapping boxes on the same head instead of one, inflating false positives faster than it misses anything. |
| 2 | `failcase_02_helm_005603.jpg` | 14 | 23 | 0 | US lecture hall shot from behind — same duplicate-box pattern as #1, this time on back-of-head views with laptops partially occluding hairlines. |
| 3 | `failcase_03_helm_006276.jpg` | 14 | 16 | 0 | Conference/gala seating, dense front-facing rows — duplicate boxes again; this failure mode (over-detection on tightly packed heads) recurs across very different venues/lighting, suggesting it's an NMS/scale issue rather than a domain-specific one. |
| 4 | `failcase_04_helm_004630.jpg` | 12 | 16 | 0 | Same CCTV classroom camera series as #1 — confirms the duplicate-box pattern is systematic for this camera's viewing angle on packed bench seating, not a one-off. |
| 5 | `failcase_05_helm_003876.jpg` | 11 | 16 | 0 | Same recurring dense-seating duplicate-box pattern. |
| 6 | `failcase_06_helm_001784.jpg` | 16 | 3 | 1 | Black-and-white historical protest photo — out-of-domain image entirely. The model still finds most heads, and correctly tags a light-colored hard hat as `helmet` (0.82-0.86 conf), but the grayscale/vintage domain shift costs recall on partially obscured heads in the crowd. |
| 7 | `failcase_07_helm_006893.jpg` | 8 | 3 | 9 | Culinary school kitchen — white chef toques get misclassified as `helmet` nine times. Strong evidence the model leans on color/shape (white, rounded, head-top) rather than construction-specific texture, so it doesn't discriminate PPE helmets from other white headwear. |

**Two real, distinct failure modes emerge**, not just "the model isn't perfect":

1. **Duplicate/overlapping boxes on densely packed heads** (#1-5, the majority of these cases) —
   consistent across camera angles, lighting, and venues, so this reads as an NMS-threshold or
   scale-handling issue rather than a data problem. Worth revisiting `iou` in NMS or adding more
   dense-crowd training examples with tight ground-truth spacing.
2. **Color/shape-based over-generalization of "helmet"** (#6, #7) — any pale, rounded, head-top
   object (a light hard hat in a B&W photo, a chef's toque) gets called a helmet. Expected, since
   the training data likely has little headwear diversity outside actual hard hats — this is
   exactly the kind of failure a label audit (see README's "what I would do next") would surface
   more of.
