import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from uuid import uuid4

import gradio as gr
import soundfile as sf

from .inference import render_polyphonic_file


def _project_root() -> Path:
    configured = os.environ.get("KIN_AUDIO_ROOT")
    return Path(configured).resolve() if configured else Path.cwd().resolve()


def _project_markdown(root: Path) -> str:
    state_path = root / "project.json"
    if not state_path.exists():
        return "Project state is not available."
    state = json.loads(state_path.read_text())
    milestones = "\n".join(
        f"- **{item['name']}** — {item['status']}: {item['evidence']}"
        for item in state["milestones"]
    )
    return (
        f"## {state['name']}\n\n"
        f"**Mission:** {state['mission']}\n\n"
        f"**Current focus:** {state['current_focus']}\n\n"
        f"### Milestones\n{milestones}"
    )


def _dataset_rows(root: Path) -> list[list[Any]]:
    registry_path = root / "datasets" / "registry.json"
    if not registry_path.exists():
        return []
    registry = json.loads(registry_path.read_text())
    return [
        [
            dataset["name"],
            dataset["license"],
            dataset["role"],
            dataset["status"],
        ]
        for dataset in registry["datasets"]
    ]


def _scan_runs(root: Path) -> tuple[list[list[Any]], list[str]]:
    rows: list[list[Any]] = []
    names: list[str] = []
    runs_root = root / "runs"
    if not runs_root.exists():
        return rows, names
    summaries = sorted(
        runs_root.glob("*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summaries:
        run_name = summary_path.parent.name
        summary = json.loads(summary_path.read_text())
        rows.append(
            [
                run_name,
                summary.get("status", "unknown"),
                summary.get("device", "unknown"),
                summary.get("epochs", 0),
                round(float(summary.get("best_validation_loss", 0.0)), 5),
                round(float(summary.get("duration_seconds", 0.0)), 1),
            ]
        )
        names.append(run_name)
    return rows, names


def _polyphonic_runs(root: Path) -> list[str]:
    _, names = _scan_runs(root)
    return [
        name
        for name in names
        if json.loads((root / "runs" / name / "summary.json").read_text()).get(
            "model_type"
        )
        == "polyphonic-spectral-unet"
    ]


def _render_prompt(
    root: Path,
    run_name: str | None,
    input_path: str | None,
) -> tuple[str | None, str]:
    if run_name not in _polyphonic_runs(root):
        return None, "Select a polyphonic checkpoint."
    if not input_path:
        return None, "Record or upload an input performance first."
    duration = sf.info(input_path).duration
    if duration > 15.0:
        return None, f"Input is {duration:.1f}s; this research demo currently accepts up to 15s."
    if duration <= 0.0:
        return None, "The input audio is empty."
    checkpoint = root / "runs" / run_name / "checkpoint.pt"
    output = Path(gettempdir()) / f"kin-audio-{uuid4().hex}.wav"
    render_polyphonic_file(checkpoint, input_path, output)
    return str(output), f"Rendered {duration:.1f}s with `{run_name}`."


def _run_artifacts(
    root: Path,
    run_name: str | None,
) -> tuple[str, str | None, str | None, str | None]:
    if not run_name:
        return "No completed run selected.", None, None, None
    run = root / "runs" / run_name
    summary_path = run / "summary.json"
    comparison_path = run / "comparison.json"
    if not summary_path.exists() or not comparison_path.exists():
        return "The selected run has no complete listening artifact.", None, None, None
    summary = json.loads(summary_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    details = (
        f"### {run_name}\n"
        f"- Validation loss: `{summary['best_validation_loss']:.5f}`\n"
        f"- Parameters: `{summary['parameter_count']:,}`\n"
        f"- Device: `{summary['device']}`\n"
        f"- Training time: `{summary['duration_seconds']:.1f}s`\n"
    )
    preservation_path = run / "preservation.json"
    if preservation_path.exists():
        preservation = json.loads(preservation_path.read_text())
        pitch_error = preservation["median_absolute_pitch_error_cents"]
        pitch_error_label = "n/a" if pitch_error is None else f"{pitch_error:.1f} cents"
        details += (
            f"- Median pitch error: `{pitch_error_label}`\n"
            f"- Voicing overlap: `{preservation['voicing_iou']:.1%}`\n"
            f"- Pitch frames within 50 cents: "
            f"`{preservation['pitch_frames_within_50_cents']:.1%}`\n"
        )
    content_type = comparison.get("content_type")
    if content_type:
        voices = comparison.get("voices")
        voice_label = f", {voices} voices" if voices else ""
        details += f"- Listening example: `{content_type}{voice_label}`\n"
    source_name = comparison.get("source")
    source = run / source_name if source_name else None
    prediction = run / comparison["prediction"]
    target = run / comparison["target"]
    return details, str(source) if source else None, str(prediction), str(target)


def build_dashboard(root: Path | None = None) -> gr.Blocks:
    root = root or _project_root()
    rows, names = _scan_runs(root)
    initial = names[0] if names else None

    with gr.Blocks(title="Kin Audio Lab") as app:
        gr.Markdown("# Kin Audio Lab\nContent-preserving neural re-instrumentation research.")
        project = gr.Markdown(_project_markdown(root))
        gr.Markdown("## Dataset registry")
        gr.Dataframe(
            headers=["Dataset", "License", "Role", "Status"],
            value=_dataset_rows(root),
            interactive=False,
        )
        gr.Markdown("## Training runs")
        runs = gr.Dataframe(
            headers=["Run", "Status", "Device", "Epochs", "Validation loss", "Seconds"],
            value=rows,
            interactive=False,
        )
        with gr.Row():
            selected = gr.Dropdown(choices=names, value=initial, label="Listening run")
            refresh = gr.Button("Refresh")
        details = gr.Markdown()
        with gr.Row():
            source = gr.Audio(label="Input audio", interactive=False)
            prediction = gr.Audio(label="Model output", interactive=False)
            target = gr.Audio(label="Target reference", interactive=False)

        gr.Markdown(
            "## Try the model\n"
            "Record or upload up to 15 seconds. This is audio-conditioned transfer, not a "
            "text-prompt model."
        )
        prompt_models = _polyphonic_runs(root)
        prompt_model = gr.Dropdown(
            choices=prompt_models,
            value=prompt_models[0] if prompt_models else None,
            label="Polyphonic checkpoint",
        )
        prompt_input = gr.Audio(
            label="Your performance",
            sources=["upload", "microphone"],
            type="filepath",
        )
        prompt_button = gr.Button("Transform audio", variant="primary")
        prompt_status = gr.Markdown()
        prompt_output = gr.Audio(label="Your model output", interactive=False)

        def refresh_state() -> tuple[
            str,
            list[list[Any]],
            gr.Dropdown,
            gr.Dropdown,
        ]:
            refreshed_rows, refreshed_names = _scan_runs(root)
            value = refreshed_names[0] if refreshed_names else None
            refreshed_prompt_models = _polyphonic_runs(root)
            return (
                _project_markdown(root),
                refreshed_rows,
                gr.Dropdown(choices=refreshed_names, value=value),
                gr.Dropdown(
                    choices=refreshed_prompt_models,
                    value=refreshed_prompt_models[0] if refreshed_prompt_models else None,
                ),
            )

        refresh.click(refresh_state, outputs=[project, runs, selected, prompt_model])
        selected.change(
            lambda name: _run_artifacts(root, name),
            inputs=selected,
            outputs=[details, source, prediction, target],
        )
        app.load(
            lambda: _run_artifacts(root, initial),
            outputs=[details, source, prediction, target],
        )
        prompt_button.click(
            lambda run, audio: _render_prompt(root, run, audio),
            inputs=[prompt_model, prompt_input],
            outputs=[prompt_output, prompt_status],
        )
    return app


def main() -> None:
    root = _project_root()
    port = int(os.environ.get("KIN_DASHBOARD_PORT", "7860"))
    host = os.environ.get("KIN_DASHBOARD_HOST", "127.0.0.1")
    build_dashboard(root).launch(
        server_name=host,
        server_port=port,
        show_error=True,
        allowed_paths=[str(root / "runs")],
    )


if __name__ == "__main__":
    main()
