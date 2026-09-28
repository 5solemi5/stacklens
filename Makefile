PY := .venv/bin/python

.PHONY: setup data pipeline pattern risk stack advisor summary app readme test all

setup:
	python3 -m venv .venv && $(PY) -m pip install -e ".[dev,advisor,app]"

data:
	mkdir -p data/raw && curl -L -o data/raw/LSWMD.pkl https://huggingface.co/datasets/lslattery/wafer-defect-detection/resolve/main/LSWMD.pkl

pipeline:  ; $(PY) scripts/run_pipeline.py
pattern:   ; $(PY) scripts/train_pattern.py
risk:      ; $(PY) scripts/train_risk.py
stack:     ; $(PY) scripts/run_stack.py
advisor:   ; $(PY) scripts/run_advisor.py
summary:   ; $(PY) scripts/summarize.py && $(PY) scripts/export_app_data.py
readme:    ; $(PY) scripts/build_readme.py
app:       ; .venv/bin/streamlit run app/streamlit_app.py
test:      ; .venv/bin/ruff check . && $(PY) -m pytest -q

all: pipeline pattern risk stack summary advisor summary readme
