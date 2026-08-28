#!/usr/bin/env bash
# Downloads SHWD (Safety Helmet Wearing Dataset) via Kaggle CLI.
# SHWD mirrors live in Google Drive folders that go stale -- Kaggle is the
# most stable mirror at the time of writing. Verify the dataset's licence
# before use; several PPE datasets are research-use-only.
set -euo pipefail

DATA_DIR="data/shwd"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/annotations" ] && [ -d "$DATA_DIR/images" ]; then
    echo "SHWD already present at $DATA_DIR, skipping download."
    exit 0
fi

if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle" >&2
    echo "Then configure ~/.kaggle/kaggle.json with your API token." >&2
    echo "" >&2
    echo "Manual alternative: download the Safety-Helmet-Wearing-Dataset" >&2
    echo "and place VOC XML annotations in $DATA_DIR/annotations" >&2
    echo "and images in $DATA_DIR/images" >&2
    exit 1
fi

kaggle datasets download -d vodan37/yolo-helmethead -p "$DATA_DIR" --unzip

echo "Downloaded to $DATA_DIR."
echo "This mirror ships pre-converted to YOLO format (not VOC XML) at:"
echo "  $DATA_DIR/helm/helm/images/{train,valid,test}"
echo "  $DATA_DIR/helm/helm/labels/{train,valid,test}"
echo "with classes {0: head, 1: helmet} and its own (naive, random) split."
echo "Run scripts/prepare_shwd_raw.py to flatten it and remap classes to"
echo "our {0: helmet, 1: head} convention before running make_splits.py."
