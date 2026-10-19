PY ?= .venv/bin/python
RUN ?= runs/g2p_base

.PHONY: setup data train eval-g2p calibrate eval-audio serve simulate listen-report test

setup:            ## create venv + install
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" transformers fastapi "uvicorn[standard]" prometheus_client httpx

data:             ## download HomoRich / SentenceBench / Mana voice / recognizer / 2 Mana-TTS shards, build tagging data
	$(PY) scripts/00_download.py
	$(PY) scripts/01_prepare_g2p_data.py

train:            ## train the streaming G2P tagger (~2h on a GTX 1650)
	$(PY) scripts/02_train_g2p.py --out $(RUN)

calibrate:        ## fit + validate recognizer projection on human Mana-TTS speech
	$(PY) scripts/03_calibrate_recognizer.py

eval-g2p:         ## text-level accuracy, lookahead trade-off, latency
	$(PY) scripts/04_eval_g2p.py --run $(RUN)

eval-audio:       ## audio-level pronunciation evaluation of full TTS systems
	$(PY) scripts/05_eval_tts_audio.py --run $(RUN)

serve:            ## run the service on :8000 (listening test at /listen)
	MALAYER_RUN=$(RUN) .venv/bin/uvicorn malayer.serve.app:app --port 8000

simulate:         ## replay normal + shifted traffic and streaming sessions through the service
	MALAYER_RUN=$(RUN) $(PY) scripts/06_simulate_traffic.py

listen-report:    ## analyze listening-test votes
	$(PY) scripts/07_analyze_listening.py

test:
	$(PY) -m pytest -q
