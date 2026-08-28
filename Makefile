.PHONY: data splits apply-splits baseline train bench export serve test

data:
	bash scripts/download_data.sh
	python scripts/prepare_shwd_raw.py --mirror-root data/shwd/helm/helm --out-root data/shwd_raw

splits:
	python scripts/make_splits.py --img-dir data/shwd_raw/images

apply-splits:
	python scripts/apply_splits.py --splits configs/splits.json \
		--img-dir data/shwd_raw/images --label-dir data/shwd_raw/labels \
		--out-root data/shwd_final

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
