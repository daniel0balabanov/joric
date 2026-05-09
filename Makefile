PYTHON        := .venv/bin/python
PIP           := .venv/bin/pip
GPT_SOVITS    := $(HOME)/GPT-SoVITS
TTS_PORT      := 9880

NVIDIA_LIB    := $(shell .venv/bin/python -c "import site,os; print(site.getsitepackages()[0]+'/nvidia/cublas/lib')" 2>/dev/null)
export LD_LIBRARY_PATH := $(NVIDIA_LIB):$(LD_LIBRARY_PATH)

# Run args (override on command line)
MODEL         ?=

# Voice training args (override on command line)
VOICE_DIR     ?= ./voices/myvoice
VOICE_NAME    ?= myvoice
VOICE_LANG    ?= en
VOICE_BATCH   ?= 4

.PHONY: venv run tts-server train-voice help

help:
	@echo "Targets:"
	@echo "  make venv          Create virtualenv and install dependencies"
	@echo "  make tts-server    Start GPT-SoVITS TTS server on port $(TTS_PORT)"
	@echo "  make run           Start joric voice chat (TTS server must be running)"
	@echo "  make train-voice   Fine-tune a new voice (set VOICE_DIR and VOICE_NAME)"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run MODEL=gemma4:e4b"
	@echo "  make run MODEL=gemma3:12b ARGS='--no-tts'"
	@echo "  make train-voice VOICE_DIR=./voices/myvoice VOICE_NAME=myvoice VOICE_LANG=en"

venv:
	python3 -m venv .venv
	$(PIP) install -e '.'

run:
	$(PYTHON) main.py $(if $(MODEL),--model $(MODEL),) $(ARGS)

ollama-cpu:
	@echo "Restarting Ollama in CPU-only mode (frees GPU for TTS)..."
	@pkill -x ollama 2>/dev/null || true
	@sleep 1
	OLLAMA_NUM_GPU=0 ollama serve &
	@sleep 2
	@echo "Ollama running on CPU."

tts-server:
	cd $(GPT_SOVITS) && \
	TORCH_BLAS_PREFER_CUBLASLT=1 $(GPT_SOVITS)/.venv/bin/python api_v2.py -a 0.0.0.0 -p $(TTS_PORT)

train-voice:
	$(PYTHON) train_voice.py \
		--name "$(VOICE_NAME)" \
		--voice-dir "$(VOICE_DIR)" \
		--language "$(VOICE_LANG)" \
		--batch-size "$(VOICE_BATCH)"
