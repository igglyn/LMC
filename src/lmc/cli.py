"""Command-line entry points for the first LMC milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .corpus import normalize_jsonl, validate_pairs
from .runner import LocalTeacherRunner, TransformersGenerationBackend, read_generation_requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmc")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("normalize", "validate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--input", required=True)
        if command == "normalize":
            subparser.add_argument("--output", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--input", required=True)
    generate.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "normalize":
        print(f"normalized {normalize_jsonl(args.input, args.output)} trace(s)")
    elif args.command == "validate":
        print(f"validated {validate_pairs(args.input)} pair(s)")
    else:
        config = ExperimentConfig.from_path(args.config)
        requests = read_generation_requests(args.input)
        backend = TransformersGenerationBackend(config.model, config.seed)
        results = LocalTeacherRunner(backend, config.teacher).generate(requests)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"generated {len(results)} local teacher response(s)")
