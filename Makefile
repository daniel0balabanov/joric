PYTHON        := .venv/bin/python
PIP           := .venv/bin/pip
GPT_SOVITS    := $(HOME)/GPT-SoVITS
TTS_PORT      := 9880

# Voice training args (override on command line)
VOICE_DIR     ?= ./voices/myvoice
VOICE_NAME    ?= myvoice
VOICE_LANG    ?= en

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
	@echo "  make run ARGS='--no-tts'"
	@echo "  make train-voice VOICE_DIR=./voices/myvoice VOICE_NAME=myvoice VOICE_LANG=en"

venv:
	python3 -m venv .venv
	$(PIP) install -e '.'

run:
	$(PYTHON) main.py $(ARGS)

tts-server:
	cd $(GPT_SOVITS) && \
	$(GPT_SOVITS)/.venv/bin/python api_v2.py -a 0.0.0.0 -p $(TTS_PORT)

train-voice:
	$(PYTHON) train_voice.py \
		--name "$(VOICE_NAME)" \
		--voice-dir "$(VOICE_DIR)" \
		--language "$(VOICE_LANG)"
