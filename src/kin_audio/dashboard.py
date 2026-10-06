import json
import os
from pathlib import Path
from typing import Any

import gradio as gr


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
    for summary_path in sorted(runs_root.glob("*/summary.json"), reverse=True):
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


def _run_artifacts(root: Path, run_name: str | None) -> tuple[str, str | None, str | None]:
    if not run_name:
        return "No completed run selected.", None, None
    run = root / "runs" / run_name
    summary_path = run / "summary.json"
    comparison_path = run / "comparison.json"
    if not summary_path.exists() or not comparison_path.exists():
        return "The selected run has no complete listening artifact.", None, None
    summary = json.loads(summary_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    details = (
        f"### {run_name}\n"
        f"- Validation loss: `{summary['best_validation_loss']:.5f}`\n"
        f"- Parameters: `{summary['parameter_count']:,}`\n"
        f"- Device: `{summary['device']}`\n"
        f"- Training time: `{summary['duration_seconds']:.1f}s`\n"
    )
    prediction = run / comparison["prediction"]
    target = run / comparison["target"]
    return details, str(prediction), str(target)


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
            prediction = gr.Audio(label="Model output", interactive=False)
            target = gr.Audio(label="Held-out target", interactive=False)

        def refresh_state() -> tuple[str, list[list[Any]], gr.Dropdown]:
            refreshed_rows, refreshed_names = _scan_runs(root)
            value = refreshed_names[0] if refreshed_names else None
            return (
                _project_markdown(root),
                refreshed_rows,
                gr.Dropdown(choices=refreshed_names, value=value),
            )

        refresh.click(refresh_state, outputs=[project, runs, selected])
        selected.change(
            lambda name: _run_artifacts(root, name),
            inputs=selected,
            outputs=[details, prediction, target],
        )
        app.load(
            lambda: _run_artifacts(root, initial),
            outputs=[details, prediction, target],
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
