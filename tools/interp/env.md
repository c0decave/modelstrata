# Interpretability Setup

This folder is the phase-2 staging area. The helper modules are dependency-light
and testable with numpy; real model runs can add a separate host environment:

```bash
python -m venv .venv-interp
. .venv-interp/bin/activate
pip install torch transformers accelerate
# Optional, depending on architecture support:
pip install transformer-lens nnsight
```

Recommended first targets:

- Qwen3/Qwen2 small checkpoints for hook-shape and logit-lens validation.
- Short prompts only on CPU.
- Cache activations to separate artifacts; do not store harmful generated text.

