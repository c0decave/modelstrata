#!/usr/bin/env bash
# Führt die gesamte Testsuite aus (stdlib unittest, keine Dependencies).
set -e
cd "$(dirname "$0")"
python3 -m unittest discover -p 'test_*.py' -v
