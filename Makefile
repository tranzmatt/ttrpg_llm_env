# TTRPG LLM Project Makefile

PYTHON := python3.12
VENV := ttrpg_llm_env
VENV_ACTIVATE := source $(VENV)/bin/activate

# Use scripts/ for all python scripts
SCRIPTDIR := scripts
EXTRACTOR_SCRIPT := $(SCRIPTDIR)/pdf_extractor.py
TRAIN_SCRIPT := $(SCRIPTDIR)/train_ttrpg_llm.py
INFER_SCRIPT := $(SCRIPTDIR)/gm_inference.py
SETUP_SCRIPT := $(SCRIPTDIR)/setup_environment_updated.py
MERGE_SCRIPT := $(SCRIPTDIR)/merge_safetensors.py

ENV_DONE := .env_setup_done
PDF_DONE := .pdf_processed
MODEL_DONE := .model_trained
MERGED_DONE := .model_merged
GGUF_DONE := .model_gguf

LLAMA_CPP := llama.cpp
GGUF_CONVERT := $(LLAMA_CPP)/convert_hf_to_gguf.py

MERGED_MODEL := trained_ttrpg_gm_merged/merged_model.safetensors
GGUF_MODEL := trained_ttrpg_gm_merged/model.gguf

all: help

help:
	@echo "TTRPG LLM Pipeline Targets:"
	@echo "  setup             - Creates Python venv and installs dependencies"
	@echo "  extractor         - Extracts training data from PDFs"
	@echo "  train             - Fine-tunes the model with your data"
	@echo "  merge_safetensors - Merge sharded safetensors into one file"
	@echo "  convert_gguf      - Convert merged safetensors to GGUF for Ollama"
	@echo "  infer             - Runs the interactive GM using your trained model"
	@echo "  clean             - Removes generated and cache files"

setup: $(ENV_DONE)
$(ENV_DONE):
	$(PYTHON) $(SETUP_SCRIPT)
	@touch $(ENV_DONE)
	@echo "✓ Environment setup completed. Activate venv with: source $(VENV)/bin/activate"

extractor: setup $(PDF_DONE)
$(PDF_DONE):
	$(VENV_ACTIVATE); $(PYTHON) $(EXTRACTOR_SCRIPT)
	@touch $(PDF_DONE)
	@echo "✓ PDF extraction complete."

train: setup extractor $(MODEL_DONE)
$(MODEL_DONE):
	$(PYTHON) $(TRAIN_SCRIPT)
	@touch $(MODEL_DONE)
	@echo "✓ Model training complete."

merge_safetensors: train $(MERGED_DONE)
$(MERGED_DONE):
	$(PYTHON) $(MERGE_SCRIPT) \
		trained_ttrpg_gm_merged/model-00001-of-00003.safetensors \
		trained_ttrpg_gm_merged/model-00002-of-00003.safetensors \
		trained_ttrpg_gm_merged/model-00003-of-00003.safetensors \
		-o $(MERGED_MODEL)
	@touch $(MERGED_DONE)
	@echo "✓ Safetensors shards merged."

convert_gguf: merge_safetensors $(GGUF_DONE)
$(GGUF_DONE):
	$(PYTHON) $(GGUF_CONVERT) \
		trained_ttrpg_gm_merged \
		--outfile $(GGUF_MODEL) \
		--outtype f16
	@touch $(GGUF_DONE)
	@echo "✓ Model converted to GGUF format ready for Ollama."

infer: setup train
	$(VENV_ACTIVATE); $(PYTHON) $(INFER_SCRIPT)

clean:
	rm -f $(ENV_DONE) $(PDF_DONE) $(MODEL_DONE) $(MERGED_DONE) $(GGUF_DONE)
	rm -rf ttrpg_llm_env
	rm -rf trained_ttrpg_gm trained_models datasets exports logs
	find . -name '*.pyc' -delete
	@echo "✓ Cleaned up build and intermediate files."

.PHONY: all help setup extractor train infer merge_safetensors convert_gguf clean

