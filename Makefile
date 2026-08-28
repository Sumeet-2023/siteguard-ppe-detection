.PHONY: data splits apply-splits baseline train bench export serve test

data:
	bash scripts/download_data.sh
	python scripts/voc_to_yolo.py --ann-dir data/shwd/annotations --out-dir data/shwd/labels

splits:
	python scripts/make_splits.py --img-dir data/shwd/images

apply-splits:
	python scripts/apply_splits.py \
		--img-dir data/shwd/images --label-dir data/shwd/labels \
		--out-root data/shwd

baseline:
	python scripts/baseline_zeroshot.py

train:
	python scripts/train.py --model yolo11n --epochs 80
	python scripts/train.py --model yolo11s --epochs 80

bench:
	python scripts/evaluate.py --runs-dir runs/ppe

export:
	python scripts/export_onnx.py

serve:
	docker build -t siteguard . && docker run -p 8000:8000 siteguard

test:
	pytest tests/ -v
