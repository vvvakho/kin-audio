import argparse
import json
from pathlib import Path

from .data import (
    generate_polyphonic_smoke_dataset,
    generate_reference_smoke_dataset,
    generate_smoke_dataset,
)
from .inference import (
    evaluate_preservation,
    render_file,
    render_polyphonic_file,
    render_reference_file,
)
from .model import RendererConfig
from .polyphonic import PolyphonicConfig
from .reference import ReferenceConfig
from .train import TrainConfig, train_model
from .train_polyphonic import train_polyphonic_model
from .train_reference import train_reference_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kin-audio",
        description="Content-preserving neural audio re-instrumentation research tools",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-smoke", help="create a license-clean smoke corpus")
    generate.add_argument("--output", type=Path, default=Path("data/smoke"))
    generate.add_argument("--examples", type=int, default=48)
    generate.add_argument("--seed", type=int, default=20261006)

    generate_polyphonic = commands.add_parser(
        "generate-polyphonic-smoke",
        help="create aligned mono and polyphonic smoke pairs",
    )
    generate_polyphonic.add_argument("--output", type=Path, default=Path("data/polyphonic-smoke"))
    generate_polyphonic.add_argument("--examples", type=int, default=48)
    generate_polyphonic.add_argument("--seed", type=int, default=20261006)

    generate_reference = commands.add_parser(
        "generate-reference-smoke",
        help="create multi-body reference-conditioned smoke triples",
    )
    generate_reference.add_argument("--output", type=Path, default=Path("data/reference-smoke"))
    generate_reference.add_argument("--examples", type=int, default=48)
    generate_reference.add_argument("--seed", type=int, default=20261006)

    train = commands.add_parser("train", help="train the harmonic baseline")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--device", default="auto")
    train.add_argument("--harmonics", type=int, default=48)
    train.add_argument("--hidden-size", type=int, default=128)

    train_polyphonic = commands.add_parser(
        "train-polyphonic",
        help="train the paired polyphonic audio baseline",
    )
    train_polyphonic.add_argument("--manifest", type=Path, required=True)
    train_polyphonic.add_argument("--output", type=Path, required=True)
    train_polyphonic.add_argument("--epochs", type=int, default=8)
    train_polyphonic.add_argument("--batch-size", type=int, default=4)
    train_polyphonic.add_argument("--learning-rate", type=float, default=3e-4)
    train_polyphonic.add_argument("--device", default="auto")
    train_polyphonic.add_argument("--base-channels", type=int, default=24)

    train_reference = commands.add_parser(
        "train-reference",
        help="train the reference-conditioned polyphonic engine",
    )
    train_reference.add_argument("--manifest", type=Path, required=True)
    train_reference.add_argument("--output", type=Path, required=True)
    train_reference.add_argument("--epochs", type=int, default=8)
    train_reference.add_argument("--batch-size", type=int, default=4)
    train_reference.add_argument("--learning-rate", type=float, default=3e-4)
    train_reference.add_argument("--device", default="auto")
    train_reference.add_argument("--base-channels", type=int, default=16)
    train_reference.add_argument("--style-dim", type=int, default=64)

    infer = commands.add_parser("infer", help="render a monophonic recording")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="auto")

    infer_polyphonic = commands.add_parser(
        "infer-polyphonic",
        help="render a mono or polyphonic recording",
    )
    infer_polyphonic.add_argument("--checkpoint", type=Path, required=True)
    infer_polyphonic.add_argument("--input", type=Path, required=True)
    infer_polyphonic.add_argument("--output", type=Path, required=True)
    infer_polyphonic.add_argument("--device", default="auto")

    infer_reference = commands.add_parser(
        "infer-reference",
        help="render audio toward a target-body reference",
    )
    infer_reference.add_argument("--checkpoint", type=Path, required=True)
    infer_reference.add_argument("--input", type=Path, required=True)
    infer_reference.add_argument("--reference", type=Path, required=True)
    infer_reference.add_argument("--output", type=Path, required=True)
    infer_reference.add_argument("--strength", type=float, default=1.0)
    infer_reference.add_argument("--device", default="auto")

    evaluate = commands.add_parser("evaluate", help="measure source-structure preservation")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate-smoke":
        manifest = generate_smoke_dataset(args.output, examples=args.examples, seed=args.seed)
        print(manifest)
    elif args.command == "generate-polyphonic-smoke":
        manifest = generate_polyphonic_smoke_dataset(
            args.output,
            examples=args.examples,
            seed=args.seed,
        )
        print(manifest)
    elif args.command == "generate-reference-smoke":
        manifest = generate_reference_smoke_dataset(
            args.output,
            examples=args.examples,
            seed=args.seed,
        )
        print(manifest)
    elif args.command == "train":
        summary = train_model(
            args.manifest,
            args.output,
            renderer_config=RendererConfig(
                harmonics=args.harmonics,
                hidden_size=args.hidden_size,
            ),
            train_config=TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            ),
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "train-polyphonic":
        summary = train_polyphonic_model(
            args.manifest,
            args.output,
            model_config=PolyphonicConfig(base_channels=args.base_channels),
            train_config=TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            ),
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "train-reference":
        summary = train_reference_model(
            args.manifest,
            args.output,
            model_config=ReferenceConfig(
                base_channels=args.base_channels,
                style_dim=args.style_dim,
            ),
            train_config=TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            ),
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "infer":
        print(render_file(args.checkpoint, args.input, args.output, device_name=args.device))
    elif args.command == "infer-polyphonic":
        print(
            render_polyphonic_file(
                args.checkpoint,
                args.input,
                args.output,
                device_name=args.device,
            )
        )
    elif args.command == "infer-reference":
        print(
            render_reference_file(
                args.checkpoint,
                args.input,
                args.reference,
                args.output,
                strength=args.strength,
                device_name=args.device,
            )
        )
    elif args.command == "evaluate":
        print(json.dumps(evaluate_preservation(args.source, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
