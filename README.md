# SiteGuard — PPE Detection & Video Inference Service

> Detects people in images/video and flags who isn't wearing a hard hat. Tracks people across
> frames so each violation is reported once, not once per frame.

**Status:** Data pipeline, both training runs, ONNX export, cross-dataset eval, and the Docker
service are all done and verified. Only the naive-split comparison and a demo GIF are missing.
See [Definition of done](#definition-of-done).

<!-- ![demo](reports/demo.gif) -->
*(demo GIF goes here once a trained model produces one)*

## Quickstart

```bash
docker compose up
curl -X POST -F "file=@sample.jpg" http://localhost:8000/detect/image
```

Requires `models/best.onnx` to exist locally before building the image. Verified working (build +
run + `/health`, `/detect/image`, `/detect/video`) — details in
[Video pipeline & service](#video-pipeline--service). Image size: **2.73 GB**.

## Dataset

Training data is `vodan37/yolo-helmethead` on Kaggle — a mirror of SHWD, but already in YOLO
format (not VOC XML as the original spec assumed) and larger (~22.8k images, ~2:1 head:helmet)
than the original 7.5k-image SHWD. Its pre-made split is discarded in favor of our own
group-disjoint pHash split (`scripts/prepare_shwd_raw.py`, `make_splits.py`, `apply_splits.py`).

## Benchmark

**Zero-shot COCO baseline** — `yolo11s.pt`, untrained on this task, against our test split:

| Metric | Value |
|---|---|
| mAP50 | 0.0045 |
| Precision | 0.0151 |
| Recall | 0.279 |

Recall of 0.28 shows COCO's person detector does find people. The near-zero mAP50 is a box shape
mismatch — COCO `person` boxes cover the full body, but ground truth here is a tight head/helmet
box, so they rarely overlap enough to count. This is why the task needs its own trained model.

**Trained models** — 80 epochs, imgsz=640, trained on Colab T4, evaluated here on CPU:

| Model | Size (MB) | mAP50 | mAP50-95 | AP50 helmet | AP50 head | CPU p50 (ms) | CPU FPS |
|---|---|---|---|---|---|---|---|
| yolo11n | 5.5 | 0.9487 | 0.5795 | 0.9665 | 0.9309 | 82.59 | 10.1 |
| yolo11s | 19.2 | 0.9619 | 0.6000 | 0.9738 | 0.9500 | 205.22 | 4.8 |

yolo11s wins on accuracy but is 2.5x slower and 3.5x larger. For CPU-only edge use, yolo11n is
within 1.3pt mAP50 of yolo11s at less than half the latency — a real tradeoff either way. GPU
latency isn't measured yet.

## Generalisation study

`yolo11s`, zero-shot on SH17 (8,099 images, 17 classes mapped down to our `{helmet, head}`
scheme — see `scripts/prepare_cross_dataset.py`). No fine-tuning.

| Train → Test | mAP50 | AP50 helmet | AP50 head |
|---|---|---|---|
| SHWD → SHWD (in-domain) | 0.9619 | 0.9738 | 0.9500 |
| SHWD → SH17 (cross-domain) | 0.5241 | 0.2850 | 0.7631 |
| SHWD → "own photos" (9 web-sourced images*) | 0.8539 | 0.9438 | 0.7640 |

The drop isn't even: `head` holds up (-19pt) but `helmet` collapses (-68.8pt). SH17 covers far
more industries and headwear styles than SHWD's narrow "hard hat or bare head" framing, so the
helmet detector doesn't transfer. This matches the failure-case finding below: the model leans on
color/shape, not real PPE features.

**\*About that third row:** no camera was available, so this isn't the 60-80 personally-shot phone
photos the spec wants — it's 9 CC-licensed Wikimedia images, labeled by correcting the model's own
predictions rather than annotating from scratch. That makes recall/mAP here optimistic and not
directly comparable to the rows above. What *is* trustworthy: the review caught 2 false positives
and one real error — a cloth cap called "helmet" at 0.89 confidence, the same color/shape bias
seen twice already. Full details: [`reports/own_photos/README.md`](reports/own_photos/README.md).

## Naive vs honest split

Scraped datasets like SHWD contain near-duplicate images. A random split leaks duplicates across
train/val/test and inflates accuracy. This project splits by perceptual-hash group instead
(`scripts/make_splits.py`): **1,234 of 22,789 images (5.4%) are duplicates**, now kept in the same
split. Comparison numbers below need a second training run on a naive split, not yet done.

| Split strategy | mAP50 |
|---|---|
| Naive (random, image-level) | TBD |
| Honest (group-disjoint by pHash) | TBD |

## Failure analysis

8 worst cases from the test split, picked by `scripts/find_failure_cases.py` (matches predictions
to ground truth, ranks by missed/extra/wrong-class boxes). Captions and images in
[`reports/failure_cases/`](reports/failure_cases/README.md). Two failure modes stand out:
**duplicate boxes on packed heads** (same bug across many camera angles — an NMS issue, not a data
gap) and **"helmet" over-applied to any pale, round headwear** (a B&W photo's hard hat, a chef's
toque) — expected given how little headwear variety is in the training data.

## Video pipeline & service

Verified against the real model (`models/best.onnx`), locally and in Docker:

- `/health`, `/detect/image` — same output locally and in the container
- `/detect/video` — tested with a synthetic panning clip (no real footage was available).
  `ViolationMonitor` fires one `no_helmet` event per newly-flagged track and never repeats,
  matching `tests/test_tracker.py`. Track IDs churn a lot on this synthetic clip since a still-image
  pan gives far less continuity than real motion — treat this as a smoke test, not a tracking
  benchmark.
- Two bugs fixed along the way: `lap` (a ByteTrack dependency) was missing from
  `requirements.txt`; a missing `.dockerignore` made the first build try to send the whole 30+ GB
  `data/` folder as build context. Both fixed.

## What I would do next

- Slim the Docker image below 2.73 GB by dropping unused plotting deps (matplotlib/pandas/seaborn)
  and hand-writing ONNX preprocessing/NMS
- INT8 quantisation for another ~2x CPU speedup
- A label audit / retrain loop on high-confidence false positives
- RT-DETR as a third benchmark model
- Head-size-stratified metrics (small objects are where these models fail most)
- Real edge deployment numbers on a Raspberry Pi 5

## Dataset licences

- **`vodan37/yolo-helmethead`** (Kaggle, primary training set) — **GNU LGPL 3.0** per Kaggle's
  listing. Not included in this repo (`scripts/download_data.sh` fetches it).
- **SH17** (`mugheesahmad/sh17-dataset-for-ppe-detection` on Kaggle) — **CC BY-NC-SA 4.0**,
  sourced from Pexels. Used only for zero-shot eval, no fine-tuning.
- Neither dataset is committed here. `make data` fetches SHWD; SH17 was a one-off download.

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

make data            # download the SHWD mirror, flatten + remap classes
make splits          # group-disjoint train/val/test split by pHash
make apply-splits    # materialise images/labels/{train,val,test}
make baseline        # zero-shot COCO person-detection sanity check
make train           # train yolo11n and yolo11s (needs a GPU for reasonable time)
make bench           # accuracy + latency table -> reports/benchmark.md
make export          # export best model to ONNX
make serve           # build and run the Docker image
```

No local GPU, so `make train` runs on Colab instead: upload
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb), set the runtime to GPU, and run it
top to bottom. It writes out and runs the same scripts shown above, so it reproduces the same
split given the same seed and source data. It trains both models, benchmarks, exports ONNX, and
zips everything for you to bring back into this repo.

## Definition of done

- [x] `docker compose up` works with no manual steps (verified via `docker build` + `docker run`;
      no compose plugin on this dev machine, but it's a thin wrapper over the same build+run)
- [x] `make bench` regenerates the benchmark table from checkpoints
- [ ] Naive-split and honest-split numbers both published (only honest-split trained so far)
- [x] Per-class AP reported everywhere, never aggregate mAP alone
- [x] Cross-dataset evaluation with a documented class mapping
- [~] Own phone-photo test set — substituted with 9 web-sourced images (no camera available); see
      the caveat above
- [ ] Latency measured with warmup and CUDA sync, CPU and GPU (CPU done, no local GPU)
- [x] 6+ annotated failure cases with explanations
- [x] Dataset licences stated; no dataset committed to git
- [x] At least one test (the tracker's violation logic)
- [ ] README opens with a GIF
