# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Japanese speech-to-text CLI tool powered by Qwen3-ASR-1.7B. Accepts wav/mp3/mp4 files and outputs transcribed Japanese text. Distributed as a pip-installable Python package.

## Language

This project is developed in Japanese. All documentation, commit messages, and comments should be written in Japanese.

## Architecture

- **CLI entry point**: `qwen3-asr transcribe <file>` — single command, takes an audio file path
- **Package format**: pyproject.toml-based, pip-installable
- **Model**: Qwen3-ASR-1.7B (local inference only, no API calls)
- **Scope**: CLI only — no web UI, API server, streaming, or multi-language support
