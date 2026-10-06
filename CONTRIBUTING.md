# Contributing

Kin Audio welcomes reproducible research, implementation improvements, datasets with clear
provenance, and critical evaluation.

## Before opening a change

- Keep re-instrumentation and generative performance translation as separate contracts.
- Do not add audio, MIDI, presets, checkpoints, or weights without an explicit compatible license.
- Do not report preservation from training examples or random chunks of a composition used in
  training.
- Prefer a small reproducible experiment over an untraceable large run.

## Development

```bash
mise install
mise exec -- uv sync --extra cpu --extra dev
mise exec -- uv run pytest
mise exec -- uv run ruff check .
```

A model change should include the configuration, dataset version, split method, seed, objective
metrics, and listening artifacts needed to examine the claim. Tests should cover observable model or
data invariants rather than implementation text.

## Dataset contributions

Each source needs:

- stable identifier and provenance;
- license and required attribution;
- renderer, instrument, or recording method;
- sample rate and channel layout;
- split-group identifier for preventing composition leakage.

Material from proprietary sample libraries is not accepted without written permission covering
redistribution and model training.
