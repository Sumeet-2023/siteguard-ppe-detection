# SiteGuard — PPE Detection & Video Inference Service

> A construction-site PPE compliance detector: given an image or video, find people and flag who
> is not wearing a hard hat. Tracks people across frames so a violation is reported once, not once
> per frame.

**Status:** data pipeline done (honest split verified: 22,789 images → 21,555 pHash groups, 5.4%
duplicates caught), zero-shot baseline recorded. Training pending — this machine has no GPU, so
Phase 4 runs via [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb). See
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

**Trained models** *(pending GPU training — see `notebooks/colab_train.ipynb`)*

| Model | Params | mAP50 | AP50 helmet | AP50 head | CPU p50 (ms) | GPU p50 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|
| yolo11n | ~2.6M | TBD | TBD | TBD | TBD | TBD | TBD |
| yolo11s | ~9.4M | TBD | TBD | TBD | TBD | TBD | TBD |

## Generalisation study

*(populate after Phase 6 — cross-dataset eval on SH17 and your own phone photos)*

| Train → Test | mAP50 |
|---|---|
| SHWD → SHWD (in-domain) | TBD |
| SHWD → SH17 (cross-domain) | TBD |
| SHWD → my own photos | TBD |

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

*(6–8 annotated failure cases go in `reports/failure_cases/`, each with a one-line caption —
populate after Phase 5)*

## What I would do next

- INT8 quantisation for another ~2× CPU speedup
- A label audit / retrain loop on high-confidence false positives
- RT-DETR as a third benchmark model for architectural contrast
- Head-size-stratified metrics (small objects are where these models actually fail)
- Real edge deployment measurement on a Raspberry Pi 5

## Dataset licences and attribution

- **SHWD** (Safety Helmet Wearing Dataset) — primary training set. Verify licence terms for your
  use case before redistribution; not included in this repo (`scripts/download_data.sh`).
- **SH17** — used only for zero-shot cross-dataset evaluation, no fine-tuning. Verify licence terms
  separately.
- Neither dataset is committed to this repository. Run `make data` to fetch SHWD.

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
│   └── data_shwd.yaml
├── scripts/
│   ├── download_data.sh
│   ├── voc_to_yolo.py
│   ├── make_splits.py
│   ├── apply_splits.py
│   ├── baseline_zeroshot.py
│   ├── prepare_cross_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── siteguard/
│   ├── tracker.py
│   ├── video.py
│   └── api.py
├── reports/
│   ├── benchmark.md
│   └── failure_cases/
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

- [ ] `docker compose up` works on a clean machine with no manual steps
- [ ] `make bench` regenerates the whole benchmark table from checkpoints
- [ ] Both naive-split and honest-split numbers are published
- [ ] Per-class AP reported everywhere, never aggregate mAP alone
- [ ] Cross-dataset evaluation present with a documented class mapping
- [ ] Own phone-photo test set labelled and reported
- [ ] Latency measured with warmup and CUDA sync, on both CPU and GPU
- [ ] 6+ annotated failure cases with explanations
- [ ] Dataset licences stated; no dataset committed to git
- [ ] At least one test (the tracker's violation logic)
- [ ] README opens with a GIF
