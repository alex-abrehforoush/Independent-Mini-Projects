PY ?= .venv/bin/python
MODEL ?= Qwen/Qwen3-VL-2B-Instruct
RUN ?= runs/zeroshot_2b

.PHONY: setup data tasks tasks-wrist eval test

setup:            ## create venv + install (hand_tracking_toolkit comes from GitHub)
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" opencv-python-headless webdataset
	.venv/bin/pip install --no-deps git+https://github.com/facebookresearch/hand_tracking_toolkit

data:             ## stream all 1516 HOT3D-Clips (Aria) into data/hot3d (~1.5 h, ~2 GB kept)
	$(PY) scripts/00_fetch_hot3d.py

tasks:            ## hand-mesh contact labels -> 4-frame tile tasks in data/tasks
	$(PY) scripts/01_build_tasks.py

tasks-wrist:      ## same tasks from the wrist-distance baseline labels
	$(PY) scripts/01_build_tasks.py --method wrist --out data/tasks_wrist

eval:             ## score MODEL (+ optional ADAPTER) on the val split
	$(PY) scripts/02_evaluate.py --model $(MODEL) $(if $(ADAPTER),--adapter $(ADAPTER)) --out $(RUN)

test:
	$(PY) -m pytest -q
