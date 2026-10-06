# Kin Audio

Open research and tooling for **content-preserving neural audio re-instrumentation**.
The musician supplies the notes, chords, timing, dynamics, and phrasing. The model changes the
sound-producing body.

Kin Audio is deliberately not a text-to-song generator. Its central question is measurable:

> Can a model turn a performed phrase into another instrument or texture without replacing the
> performance?

## Status

Early research. The primary path is now a paired audio-to-audio model that accepts both chords and
monophonic phrases:

- a complex-spectrogram U-Net baseline that preserves the complete source spectrum;
- aligned mono and polyphonic smoke-corpus generation;
- a monophonic DDSP renderer retained as an inspectable control, not the release target;
- reproducible training, inference, structural evaluation, and mobile listening tools;
- an experiment dashboard with mobile recording, upload, and listening controls.

The synthetic corpora validate the pipelines; they are not evidence of real-recording quality. The
next benchmark is openly licensed paired audio from StarNet. The intended musical model is
polyphonic from the beginning, with monophonic performances included in the same training
distribution.

## Research contract

Kin separates two tasks that are often conflated:

1. **Re-instrumentation** preserves notes, timing, voicing, and phrasing while changing timbre.
2. **Performance translation** preserves selected structure while generating missing musical
   information, such as turning drums into a pitched bass line.

Every release must state which contract it implements. Preservation claims require held-out metrics
and public listening examples.

## Product contract

The intended interaction has three meaningful inputs:

1. **Performance audio** supplies the notes, chords, timing, dynamics, and phrasing.
2. **Body reference audio** demonstrates the instrument or texture that should perform it.
3. **Transformation strength** moves continuously from the dry performance to the referenced body.

The engine must not need separate monophonic and polyphonic modes. At strength zero it must return
the dry input exactly. Named presets and future text guidance should resolve into the same
target-body representation rather than create parallel generation paths.

## Quick start

Prerequisites: `mise` or Python 3.11–3.13.

```bash
mise install
mise exec -- uv sync --extra cpu --extra dev
mise exec -- uv run kin-audio generate-reference-smoke \
  --output data/reference-smoke \
  --examples 48
mise exec -- uv run kin-audio train-reference \
  --manifest data/reference-smoke/manifest.jsonl \
  --output runs/reference-smoke-v1 \
  --epochs 8
mise exec -- uv run kin-audio infer-reference \
  --checkpoint runs/reference-smoke-v1/checkpoint.pt \
  --input data/reference-smoke/00000/source.wav \
  --reference data/reference-smoke/00001/target-warm_strings.wav \
  --strength 0.8 \
  --output runs/reference-smoke-v1/source_render.wav
```

Choose exactly one accelerator extra: `cpu` for development and CI or `cu130` for CUDA 13 cloud
training. The lockfile resolves each from the official PyTorch package index. ROCm remains a
platform-specific environment because AMD officially supports this GPU on Ubuntu/RHEL, not Arch.

Launch the local dashboard:

```bash
mise exec -- uv run kin-dashboard
```

The dashboard reads `project.json` and completed runs under `runs/`. It accepts a performance plus
a preset or uploaded body reference, and exposes a structural dry-to-target strength control.
Inputs are limited to 15 seconds in this research surface. Prompt renders are temporary; the
dashboard does not retrain models or mutate experiment state.

## Model

The primary engine keeps polyphonic information and separates musical content from target-body
guidance:

```text
performance audio ──► complex STFT ──► residual spectral U-Net ──► output audio
                                             ▲
                                             │ FiLM conditioning
body reference audio ──► style encoder ──────┘
                                             ▲
transformation strength ─────────────────────┘
```

The performance can contain a melody, chord, or overlapping voices. The independent reference
recording demonstrates the desired instrument or texture without needing matching notes. Its
smoothed spectral envelope is applied structurally, so the engine cannot silently ignore the body
reference; a learned style encoder then conditions the U-Net bottleneck for refinement. Strength
scales both paths and returns the input tensor exactly at zero.

An explicit frequency-position channel lets the model learn absolute formant regions rather than
incorrectly assuming that every spectral transformation is frequency-translation invariant.
Source, reference, and target examples are split by paired performance group so no performance or
its reference crosses the train/validation boundary.

The explicit-pitch DDSP renderer remains a diagnostic lower bound. It is useful for proving pitch
preservation and understanding failures, but it cannot represent chords and is no longer the main
model direction.

Current limitations are explicit:

- the polyphonic U-Net is a baseline before codec diffusion or flow matching;
- training currently uses short 16 kHz segments;
- objective polyphonic note/voicing evaluation still needs the real-data benchmark;
- no musical checkpoint has been released.

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
