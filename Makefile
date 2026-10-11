PY := .venv/Scripts/python.exe
ifeq ($(OS),)
$(error This Makefile assumes Windows paths; on macOS/Linux set PY=.venv/bin/python)
endif

.PHONY: help install samples ingest api ui test lint eval clean

help:
	@echo "install   create the venv and install requirements"
	@echo "samples   generate the three sample papers"
	@echo "ingest    ingest data/sample_papers into a fresh index"
	@echo "api       run the FastAPI backend on :8000"
	@echo "ui        run the Streamlit front end on :8501"
	@echo "test      run the test suite"
	@echo "eval      run the full evaluation and write a report"
	@echo "clean     delete the venv, index and results"

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

samples:
	$(PY) scripts/make_sample_papers.py

ingest: samples
	$(PY) scripts/ingest.py data/sample_papers --reset

api:
	$(PY) -m uvicorn app.main:app --reload --port 8000

ui:
	$(PY) -m streamlit run ui/streamlit_app.py --server.port 8501

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check app eval scripts tests ui

eval:
	$(PY) -m eval.run --modes rag agent

clean:
	rm -rf .venv data/chroma data/registry.db eval/results
