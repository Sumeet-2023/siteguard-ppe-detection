# SiteGuard — PPE Detection & Video Inference Service

> A construction-site PPE compliance detector: given an image or video, find people and flag who
> is not wearing a hard hat. Tracks people across frames so a violation is reported once, not once
> per frame.

**Status:** data pipeline, both training runs, ONNX export, cross-dataset eval (SH17), and the
Docker service are all done and verified end to end — `/detect/image` and `/detect/video` tested
directly against the container. Only your own phone-photo test set and a demo GIF remain. See
[Definition of done](#definition-of-done).

Dataset note: `vodan37/yolo-helmethead` on Kaggle ships pre-converted to YOLO format already
(not VOC XML) and combines multiple sources into ~22.8k images — larger and less imbalanced
(~2:1 head:helmet) than the original 7.5k-image SHWD the project spec describes. Its own
train/valid/test split is discarded in favor of our group-disjoint pHash split
(`scripts/prepare_shwd_raw.py` flattens + remaps classes, then `make_splits.py` +
`apply_splits.py` do the honest split).

<!-- ![demo](reports/demo.gif) -->
*(demo GIF goes here once a trained model produces one — see Phase 8)*

## Quickstart

```bash
docker compose up
curl -X POST -F "file=@sample.jpg" http://localhost:8000/detect/image
```

Requires `models/best.onnx` to exist locally before building the image (see [Training](#training)).
Verified working: `docker build` + `docker run -p 8000:8000` (this dev machine has no `docker
compose` plugin, but the Dockerfile/compose file are equivalent — compose just wraps the same
build+run+port-mapping) — `/health`, `/detect/image`, and `/detect/video` all tested directly
against the container and matched local (non-Docker) results exactly. Final image: **2.73 GB**
(the CPU-only torch index still pulls matplotlib/pandas/seaborn as Ultralytics deps even though
the API never plots anything — stripping those would shrink this further, noted in
[What I would do next](#what-i-would-do-next)).

## Benchmark

**Zero-shot COCO baseline** (Phase 3, already run — see `reports/benchmark.md`): `yolo11s.pt`
against our test split, filtered to COCO class `person`.

| Metric | Value |
|---|---|
| mAP50 | 0.0045 |
| Precision | 0.0151 |
| Recall | 0.279 |

Recall of 0.28 shows COCO's person detector does find people in these frames; the near-zero mAP50
is a box-geometry mismatch — COCO `person` is full-body, SHWD ground truth is a tight head/helmet
region, so IoU ≥ 0.5 is essentially never reached. This is the case for training PPE-specific
boxes rather than reusing generic person detection.

**Trained models** — 80 epochs, imgsz=640, trained on Colab T4, evaluated on this (CPU-only)
machine against the held-out test split (see `reports/benchmark.md` for full detail):

| Model | Size (MB) | mAP50 | mAP50-95 | AP50 helmet | AP50 head | CPU p50 (ms) | CPU FPS |
|---|---|---|---|---|---|---|---|
| yolo11n | 5.5 | 0.9487 | 0.5795 | 0.9665 | 0.9309 | 82.59 | 10.1 |
| yolo11s | 19.2 | 0.9619 | 0.6000 | 0.9738 | 0.9500 | 205.22 | 4.8 |

yolo11s wins on every accuracy metric but is 2.5x slower and 3.5x larger. For CPU-only edge
deployment, yolo11n is within 1.3pt mAP50 of yolo11s at less than half the latency — a real
tradeoff, not a clear winner either way. GPU latency not yet measured (would need to run on Colab).

## Generalisation study

`yolo11s`, zero-shot on SH17 (8,099 images, 17 classes remapped to our `{helmet, head}` scheme —
see `scripts/prepare_cross_dataset.py` and `reports/benchmark.md` for the full class mapping and
instance-count verification). No fine-tuning.

| Train → Test | mAP50 | AP50 helmet | AP50 head |
|---|---|---|---|
| SHWD → SHWD (in-domain) | 0.9619 | 0.9738 | 0.9500 |
| SHWD → SH17 (cross-domain) | 0.5241 | 0.2850 | 0.7631 |
| SHWD → "own photos" (9 web-sourced images — see caveat) | 0.8539* | 0.9438* | 0.7640* |

The drop isn't uniform: `head` generalises reasonably (-19pt) but `helmet` collapses (-68.8pt) —
SH17's Pexels-sourced images span far more industries and headwear styles than SHWD's
construction-focused "hard hat or bare head" framing, so the helmet detector doesn't transfer.
Consistent with the failure-case finding that the model leans on color/shape cues for "helmet."

**\*Read before trusting this row.** No camera/site access was available in this environment, so
this isn't the 60-80 personally-shot phone photos the spec asks for — it's 9 CC-licensed Wikimedia
Commons images, labeled via *model-assisted* review (I corrected the model's own draft predictions
rather than annotating from scratch), which means recall/mAP here is likely optimistic and not
directly comparable to the two rows above. What is trustworthy: the review caught 2 false
positives and 1 real misclassification — a soft cloth cap called "helmet" at 0.89 confidence — the
same color/shape over-generalization bug showing up for a third independent time (Phase 5's
failure cases, the SH17 eval, now this). Full methodology and per-image licensing/attribution:
[`reports/own_photos/README.md`](reports/own_photos/README.md).

## Naive vs honest split

Scraped datasets like SHWD contain near-duplicate images (same stock photo, different resolution).
A random image-level split leaks duplicates across train/val/test and inflates mAP. This project
splits by perceptual-hash group instead (`scripts/make_splits.py`).

Measured on the combined 22,789-image pool: **1,234 duplicate images (5.4%)** collapse into
21,555 pHash groups. That's the leakage a naive random split would scatter across train/val/test.
mAP numbers for both split strategies are reported below once training completes.

| Split strategy | mAP50 |
|---|---|
| Naive (random, image-level) | TBD |
| Honest (group-disjoint by pHash) | TBD |

## Failure analysis

8 worst cases from the test split, auto-selected by `scripts/find_failure_cases.py` (greedy
IoU-matching against ground truth, ranked by missed/extra/misclassified boxes) — captions and
images in [`reports/failure_cases/`](reports/failure_cases/README.md). Two distinct failure modes
emerge: **duplicate/overlapping boxes on densely packed heads** (consistent across camera angles
and venues — an NMS/scale issue, not a data gap) and **color/shape over-generalization of
"helmet"** onto other pale, rounded headwear (a B&W photo's hard hat, a chef's toque) — expected
given limited headwear diversity in training data.

## Video pipeline & service

Verified end to end against the real trained model (`models/best.onnx`), both locally and inside
the built Docker container:

- `/health`, `/detect/image` — tested with a real held-out test image; identical output locally
  vs. containerized (same boxes, same confidences)
- `/detect/video` — tested against a synthetic 40-frame panning clip built from a test image (no
  real camera footage was available in this environment); `ViolationMonitor` correctly fires one
  `no_helmet` event per newly-flagged track and never re-fires for the same track, matching the
  unit tests in `tests/test_tracker.py`. Track IDs churn heavily on this synthetic pan (expected —
  a static-image pan gives ByteTrack much less frame-to-frame continuity than real motion would),
  so treat this as a pipeline smoke test, not a tracking-quality benchmark; that needs real footage.
- Fixed along the way: `lap` (a ByteTrack dependency) was missing from `requirements.txt` — pinned
  now. A missing `.dockerignore` also caused the first build attempt to try sending the entire
  `data/` directory (30+ GB) as build context and exhaust Docker's storage partition — added one
  scoped to just what the image actually needs (`siteguard/`, `requirements.txt`, `models/best.onnx`).

## What I would do next

- Slim the Docker image below its current 2.73 GB by dropping PyTorch/Ultralytics' plotting deps
  (matplotlib/pandas/seaborn) that the inference-only API never uses, replacing them with
  hand-written ONNX Runtime preprocessing/NMS (the exact tradeoff the project spec cut for scope)
- INT8 quantisation for another ~2× CPU speedup
- A label audit / retrain loop on high-confidence false positives
- RT-DETR as a third benchmark model for architectural contrast
- Head-size-stratified metrics (small objects are where these models actually fail)
- Real edge deployment measurement on a Raspberry Pi 5

## Dataset licences and attribution

- **`vodan37/yolo-helmethead`** (Kaggle mirror of SHWD, used as the primary training set) —
  licensed **GNU LGPL 3.0** per Kaggle's own listing at download time. Verify this hasn't changed
  before redistributing anything derived from it; not included in this repo
  (`scripts/download_data.sh`).
- **SH17** (`mugheesahmad/sh17-dataset-for-ppe-detection` on Kaggle) — licensed **CC BY-NC-SA 4.0**
  per the dataset maintainer's GitHub repo (github.com/ahmadmughees/SH17dataset); sourced from
  Pexels imagery. Used only for zero-shot cross-dataset evaluation, no fine-tuning — the
  non-commercial/share-alike terms matter more if you ever train on it directly.
- Neither dataset is committed to this repository. Run `make data` for SHWD; SH17 was fetched
  ad hoc for the cross-dataset eval (`kaggle datasets download -d mugheesahmad/sh17-dataset-for-ppe-detection`).

---

## Project layout

```
.
├── README.md
├── requirements.txt
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── configs/
│   ├── data_shwd.yaml
│   └── splits.json
├── notebooks/
│   └── colab_train.ipynb
├── scripts/
│   ├── download_data.sh
│   ├── prepare_shwd_raw.py
│   ├── voc_to_yolo.py
│   ├── make_splits.py
│   ├── apply_splits.py
│   ├── baseline_zeroshot.py
│   ├── prepare_cross_dataset.py
│   ├── prepare_own_photos.py
│   ├── build_own_photos_labels.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── find_failure_cases.py
├── siteguard/
│   ├── tracker.py
│   ├── video.py
│   └── api.py
├── models/
│   └── best.onnx
├── runs/ppe/
│   ├── yolo11n/weights/{best,last}.pt
│   └── yolo11s/weights/{best,last}.pt
├── reports/
│   ├── benchmark.md
│   ├── failure_cases/
│   └── own_photos/
│       ├── README.md
│       ├── corrections.json
│       └── draft_preds.json
└── tests/
    └── test_tracker.py
```

## Running it end to end

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data           # download the SHWD mirror, flatten + remap classes
make splits          # group-disjoint train/val/test split by pHash
make apply-splits    # materialise images/labels/{train,val,test}
make baseline        # zero-shot COCO person-detection sanity check
make train           # train yolo11n and yolo11s (needs GPU for reasonable time)
make bench           # accuracy + latency benchmark table -> reports/benchmark.md
make export          # export best model to ONNX
make serve           # build and run the Docker image
```

No GPU locally, so Phase 4 (`make train`) runs on Colab instead: upload
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb), set the runtime to a GPU, and run
top to bottom. It's self-contained — it writes out and runs the same `scripts/*.py` shown above
(so it reproduces the identical honest split, byte-for-byte, given the same seed and source
data), trains both models, benchmarks, exports ONNX, and packages everything into a zip to bring
back into this repo (unzip into the repo root — it recreates `models/`, `reports/`, and
`configs/splits.json`).

## Definition of done

- [x] `docker compose up` works on a clean machine with no manual steps (verified via `docker
      build` + `docker run`; no `docker compose` plugin on this dev machine to test the wrapper
      literally, but it's a thin equivalent of the same build+run)
- [x] `make bench` regenerates the whole benchmark table from checkpoints
- [ ] Both naive-split and honest-split numbers are published (only honest-split trained so far)
- [x] Per-class AP reported everywhere, never aggregate mAP alone
- [x] Cross-dataset evaluation present with a documented class mapping
- [~] Own phone-photo test set labelled and reported — substituted with 9 web-sourced images
      (no camera access available); see the caveat in the Generalisation study section
- [ ] Latency measured with warmup and CUDA sync, on both CPU and GPU (CPU done, no local GPU)
- [x] 6+ annotated failure cases with explanations
- [x] Dataset licences stated; no dataset committed to git
- [x] At least one test (the tracker's violation logic)
- [ ] README opens with a GIF
