from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import resolve_checkpoint_spec
from .peaks import call_peaks
from .predict import scan_genome


STRICT_PEAK_NOTE = (
    "Default peak calling is intentionally stricter than a calibrated classifier threshold: "
    "min score 0.70 plus local prominence filtering. Lower it only when recall is more important."
)


def add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--checkpoint-registry", help="JSON/YAML registry of released checkpoints.")
    parser.add_argument("--checkpoint-label", help="Checkpoint label from registry. Defaults to registry default.")
    parser.add_argument("--checkpoint", help="Direct checkpoint path. Must be used with --model-config.")
    parser.add_argument("--model-config", help="Direct resolved model config path. Must be used with --checkpoint.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--torch-dtype", default=None, help="Override model dtype from resolved config.")
    parser.add_argument("--window-size", type=int, default=30000, help="Sliding window size in bp. Default: 30000.")
    parser.add_argument("--stride", type=int, default=30000, help="Sliding window stride in bp. Default: 30000.")
    parser.add_argument("--batch-size", type=int, default=4, help="Prediction batch size.")
    parser.add_argument("--max-n-frac", type=float, default=0.20, help="Skip windows with larger N fraction.")
    parser.add_argument("--sequence-names", help="Comma-separated FASTA sequence names, e.g. chr1,chr2.")
    parser.add_argument("--regions", help="Comma-separated regions, e.g. chr1:0-1000000,chr2:50000-90000.")
    parser.add_argument("--regions-file", help="Text file with one sequence name or chrom:start-end per line.")
    parser.add_argument("--score", choices=["prob", "logit"], default="prob", help="Score written to bedGraph.")
    parser.add_argument("--track-bin-size", type=int, default=None, help="bedGraph center-bin size. Default: stride.")
    parser.add_argument("--output-prefix", default="izscan", help="Output file prefix.")
    parser.add_argument("--write-sequences", action="store_true", help="Include raw window sequence in TSV output.")
    parser.add_argument(
        "--no-terminal-window",
        dest="include_terminal_window",
        action="store_false",
        help="Do not add a final terminal window to cover each region end.",
    )
    parser.set_defaults(include_terminal_window=True)


def add_peak_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--score-track", required=True, help="Input bedGraph with chrom/start/end/score.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--method", choices=["scipy", "threshold", "macs3"], default="scipy", help="Peak caller.")
    parser.add_argument(
        "--peak-min-score",
        type=float,
        default=0.70,
        help="Strict score cutoff for candidate IZ peaks. Default: 0.70.",
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=0.05,
        help="Minimum local prominence for scipy peaks. Default: 0.05.",
    )
    parser.add_argument(
        "--peak-min-distance",
        type=int,
        default=30000,
        help="Minimum distance between scipy summits in bp. Default: 30000.",
    )
    parser.add_argument("--peak-min-width", type=int, default=5000, help="Minimum peak width in bp. Default: 5000.")
    parser.add_argument("--peak-max-gap", type=int, default=30000, help="Merge/extend across gaps up to bp.")
    parser.add_argument("--smooth-bins", type=int, default=5, help="Rolling mean bins before peak calling.")
    parser.add_argument("--output-prefix", default="izscan", help="Output file prefix.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="izscan",
        description="Whole-genome IZ probability scanner and decoupled peak caller.",
    )
    parser.add_argument("--version", action="version", version=f"izscan {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Scan a genome FASTA with a checkpoint.")
    add_predict_args(predict_parser)

    peaks_parser = subparsers.add_parser(
        "call-peaks", help=f"Call candidate IZ peaks from any score bedGraph. {STRICT_PEAK_NOTE}"
    )
    add_peak_args(peaks_parser)

    run_parser = subparsers.add_parser("run", help="Run prediction and then peak calling.")
    add_predict_args(run_parser)
    run_parser.add_argument("--method", choices=["scipy", "threshold", "macs3"], default="scipy", help="Peak caller.")
    run_parser.add_argument("--peak-min-score", type=float, default=0.70, help="Strict peak score cutoff.")
    run_parser.add_argument("--peak-prominence", type=float, default=0.05, help="Minimum local prominence.")
    run_parser.add_argument("--peak-min-distance", type=int, default=30000, help="Minimum summit distance in bp.")
    run_parser.add_argument("--peak-min-width", type=int, default=5000, help="Minimum peak width in bp.")
    run_parser.add_argument("--peak-max-gap", type=int, default=30000, help="Merge/extend across gaps up to bp.")
    run_parser.add_argument("--smooth-bins", type=int, default=5, help="Rolling mean bins before peak calling.")
    return parser


def _checkpoint_from_args(args: Any):
    return resolve_checkpoint_spec(args.checkpoint_registry, args.checkpoint_label, args.checkpoint, args.model_config)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        summary = scan_genome(args, _checkpoint_from_args(args))
    elif args.command == "call-peaks":
        summary = call_peaks(args)
    elif args.command == "run":
        scan_summary = scan_genome(args, _checkpoint_from_args(args))
        peak_args = argparse.Namespace(**vars(args))
        peak_args.score_track = Path(scan_summary["outputs"]["bedgraph"])
        summary = {"prediction": scan_summary, "peak_calling": call_peaks(peak_args)}
    else:
        parser.error(f"Unsupported command: {args.command}")
        return
    print(json.dumps(summary, ensure_ascii=False, indent=2))

