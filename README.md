# Kin Audio

Open research and tooling for **content-preserving neural audio re-instrumentation**.
The musician supplies the notes, chords, timing, dynamics, and phrasing. The model changes the
sound-producing body.

Kin Audio is deliberately not a text-to-song generator. Its central question is measurable:

> Can a model turn a performed phrase into another instrument or texture without replacing the
> performance?

## Status

Early research. The repository currently contains a complete monophonic baseline:

- deterministic pitch, loudness, and voicing extraction;
- a compact DDSP-style harmonic renderer;
- a license-clean paired smoke-corpus generator;
- reproducible training, inference, and structural evaluation commands;
- a read-only experiment dashboard.

The synthetic corpus validates the pipeline; it is not evidence of real-recording quality. The next
benchmark is openly licensed real audio, beginning with StarNet. The first intended musical model is
monophonic voice/guitar/synth to a learned choir-like body.

## Research contract

Kin separates two tasks that are often conflated:

1. **Re-instrumentation** preserves notes, timing, voicing, and phrasing while changing timbre.
2. **Performance translation** preserves selected structure while generating missing musical
   information, such as turning drums into a pitched bass line.

Every release must state which contract it implements. Preservation claims require held-out metrics
and public listening examples.

## Quick start

Prerequisites: `mise` or Python 3.11–3.13.

```bash
mise install
mise exec -- uv sync --extra cpu --extra dev
mise exec -- uv run kin-audio generate-smoke --output data/smoke --examples 48
mise exec -- uv run kin-audio train \
  --manifest data/smoke/manifest.jsonl \
  --output runs/smoke-v1 \
  --epochs 8
mise exec -- uv run kin-audio infer \
  --checkpoint runs/smoke-v1/checkpoint.pt \
  --input data/smoke/00000/source.wav \
  --output runs/smoke-v1/source_render.wav
mise exec -- uv run kin-audio evaluate \
  --source data/smoke/00000/source.wav \
  --output runs/smoke-v1/source_render.wav \
  --report runs/smoke-v1/preservation.json
```

Choose exactly one accelerator extra: `cpu` for development and CI or `cu130` for CUDA 13 cloud
training. The lockfile resolves each from the official PyTorch package index. ROCm remains a
platform-specific environment because AMD officially supports this GPU on Ubuntu/RHEL, not Arch.

Launch the local dashboard:

```bash
mise exec -- uv run kin-dashboard
```

The dashboard reads `project.json` and completed runs under `runs/`. It does not mutate training
state or expose write operations.

## Model

The baseline makes pitch preservation structural:

```text
source audio
    │
    ├── fundamental frequency
    ├── loudness
    └── voicing
            │
            ▼
 recurrent harmonic-control network
            │
            ▼
 anti-aliased differentiable oscillator bank
            │
            ▼
 target-body audio
```

The network predicts time-varying harmonic distributions and gain. It does not predict pitch. That
constraint makes the model inspectable and provides a useful lower bound before less constrained
codec or diffusion models are introduced.

Current limitations are explicit:

- monophonic pitched input only;
- harmonic synthesis without a learned stochastic/noise branch;
- dependency-light autocorrelation pitch extraction intended as a baseline;
- no released musical checkpoint yet.

## Data policy

Code, datasets, and model weights are independently licensed artifacts.

Public Kin datasets must include provenance and an explicit redistribution license for every source.
Proprietary DAW packs and sample libraries are excluded unless their owners grant written permission
for dataset publication and model training. Synthetic data must record the renderer, version, preset
hash, random seed, MIDI source, and effects chain.

Train, validation, and test splits are made by underlying composition or performance—not random audio
chunks—to prevent the same MIDI arrangement from leaking between splits.

Initial external datasets:

- [StarNet](https://zenodo.org/records/6917099), paired re-instrumentation audio, CC BY 4.0;
- [NSynth](https://magenta.withgoogle.com/datasets/nsynth), isolated annotated notes, CC BY 4.0;
- [Slakh2100](https://zenodo.org/records/4599666), aligned MIDI and stems, CC BY 4.0.

## Evaluation

A useful model must satisfy both sides:

- **content preservation:** pitch error, voicing overlap, onset deviation, chroma/voicing agreement;
- **target quality:** held-out reconstruction, target-instrument similarity, artifacts, and human
  listening preference.

A model that sounds convincing while replacing the notes has failed re-instrumentation. A model that
preserves MIDI perfectly but sounds like a poor sampler has also failed musically.

## Project layout

```text
src/kin_audio/       model, training, inference, evaluation, dashboard
 tests/              behavior and invariant tests
 project.json        machine-readable current research state
 data/               generated or downloaded corpora; never committed
 runs/               checkpoints, metrics, and listening artifacts; never committed
```

## License

Code is licensed under Apache-2.0. Dataset and model releases will declare their own licenses and
source attributions.
