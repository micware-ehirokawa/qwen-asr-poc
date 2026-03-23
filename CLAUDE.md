# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Japanese speech-to-text CLI tool powered by Qwen3-ASR-1.7B. Accepts wav/mp3/mp4 files and outputs transcribed Japanese text. Distributed as a pip-installable Python package.

## Language

This project is developed in Japanese. All documentation, commit messages, and comments should be written in Japanese.

## Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run CLI
qwen3-asr transcribe <audio_file>
```

## Architecture

- `src/qwen3_asr/cli.py` — Click-based CLI. Entry point: `main`. Validation + orchestration.
- `src/qwen3_asr/transcriber.py` — Model loading (`load_model`) and inference (`transcribe`). No class wrapper, plain functions.
- `src/qwen3_asr/exceptions.py` — Custom exceptions with Japanese messages. All inherit `Qwen3AsrError`.
- CLI catches `Qwen3AsrError` and outputs to stderr. stdout is reserved for transcription results only.

## Key Design Decisions

- `language="Japanese"` is hardcoded (Japanese-only scope)
- GPU: `bfloat16` / CPU: `float32` (bfloat16 on CPU causes hangs on some hardware)
- `Qwen3ASRModel` is imported locally inside `load_model()` to avoid heavy import at CLI startup
- `click.Path(exists=True)` is NOT used — validation is done manually for Japanese error messages
