#!/usr/bin/env bash
# Sarashina Discord Bot を起動する
cd "$(dirname "$0")"
exec .venv/bin/python bot/main.py
