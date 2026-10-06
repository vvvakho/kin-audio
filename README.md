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

## Quick start

Prerequisites: `mise` or Python 3.11–3.13.

```bash
mise install
mise exec -- uv sync --extra cpu --extra dev
mise exec -- uv run kin-audio generate-polyphonic-smoke \
  --output data/polyphonic-smoke \
  --examples 48
mise exec -- uv run kin-audio train-polyphonic \
  --manifest data/polyphonic-smoke/manifest.jsonl \
  --output runs/polyphonic-smoke-v1 \
  --epochs 8
mise exec -- uv run kin-audio infer-polyphonic \
  --checkpoint runs/polyphonic-smoke-v1/checkpoint.pt \
  --input data/polyphonic-smoke/00000/source.wav \
  --output runs/polyphonic-smoke-v1/source_render.wav
```

Choose exactly one accelerator extra: `cpu` for development and CI or `cu130` for CUDA 13 cloud
training. The lockfile resolves each from the official PyTorch package index. ROCm remains a
platform-specific environment because AMD officially supports this GPU on Ubuntu/RHEL, not Arch.

Launch the local dashboard:

```bash
mise exec -- uv run kin-dashboard
```

The dashboard reads `project.json` and completed runs under `runs/`. It can record or upload up to
15 seconds and render the performance through a polyphonic checkpoint. Prompt renders are temporary;
the dashboard does not retrain models or mutate experiment state.

## Model

The primary baseline keeps polyphonic information rather than reducing the source to one pitch
track:

```text
source audio: melody, chord, or overlapping voices
                    │
                    ▼
             complex STFT
                    │
                    ▼
       residual spectral U-Net
                    │
                    ▼
                inverse STFT
                    │
                    ▼
             target-body audio
```

The network predicts a complex spectral residual around the source. Its output projection is not
zero-initialized: a zero residual blocked useful gradients from reaching the encoder and made the
early smoke checkpoint behave almost exactly like an identity transform. An explicit
frequency-position channel also lets the model learn absolute formant regions rather than
incorrectly assuming that every spectral transformation is frequency-translation invariant. Paired
source/target training teaches the timbre change. Mono and polyphonic inputs use the same
architecture and objective.

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
