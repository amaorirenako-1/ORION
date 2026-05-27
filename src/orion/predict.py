from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import __tool_name__, __version__
from .config import CheckpointSpec, write_resolved_run_config
from .fasta import GenomeFasta
from .model import load_predictor
from .utils import ensure_dir, open_text, write_json
from .windows import WindowRecord, iter_windows, resolve_regions


def _score_value(prediction: dict[str, float], score: str) -> float:
    if score not in prediction:
        raise KeyError(f"Prediction has no score column: {score}")
    return float(prediction[score])


class BedGraphWriter:
    def __init__(self, path: str | Path, track_bin_size: int):
        self.path = Path(path)
        self.handle = open_text(self.path, "wt")
        self.track_bin_size = int(track_bin_size)
        self.last_end: dict[str, int] = {}
        self.rows_written = 0

    def write(self, record: WindowRecord, chrom_length: int, score: float) -> None:
        center = (record.start + record.end) // 2
        half = max(1, self.track_bin_size // 2)
        start = max(0, center - half)
        end = min(chrom_length, start + self.track_bin_size)
        previous_end = self.last_end.get(record.chrom, 0)
        if start < previous_end:
            start = previous_end
        if end <= start:
            return
        self.handle.write(f"{record.chrom}\t{start}\t{end}\t{score:.8g}\n")
        self.last_end[record.chrom] = end
        self.rows_written += 1

    def close(self) -> None:
        self.handle.close()


def _flush_batch(
    predictor,
    batch: list[WindowRecord],
    score_name: str,
    windows_writer: csv.DictWriter,
    bedgraph_writer: BedGraphWriter,
    genome: GenomeFasta,
    write_sequences: bool,
) -> int:
    if not batch:
        return 0
    predictions = predictor.predict_batch([record.sequence for record in batch])
    for record, prediction in zip(batch, predictions, strict=False):
        score = _score_value(prediction, score_name)
        row = {
            "chrom": record.chrom,
            "start": record.start,
            "end": record.end,
            "coord_key": record.coord_key,
            "prob": f"{prediction['prob']:.8g}",
            "logit": f"{prediction['logit']:.8g}",
            "score_name": score_name,
            "score": f"{score:.8g}",
            "n_fraction": f"{record.n_fraction:.6g}",
            "window_length": record.end - record.start,
        }
        if write_sequences:
            row["sequence"] = record.sequence
        windows_writer.writerow(row)
        bedgraph_writer.write(record, genome.lengths[record.chrom], score)
    return len(batch)


def scan_genome(args: Any, checkpoint_spec: CheckpointSpec) -> dict[str, Any]:
    out_dir = ensure_dir(args.output_dir)
    genome = GenomeFasta(args.fasta)
    regions = resolve_regions(genome, args.sequence_names, args.regions, args.regions_file)
    chrom_order = {chrom: idx for idx, chrom in enumerate(genome.names)}
    regions = sorted(regions, key=lambda item: (chrom_order.get(item.chrom, 10**9), item.start, item.end))
    predictor = load_predictor(
        checkpoint_spec.model_backend,
        checkpoint_spec.checkpoint,
        checkpoint_spec.config,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    prefix = args.output_prefix
    windows_path = out_dir / f"{prefix}.window_predictions.tsv"
    bedgraph_path = out_dir / f"{prefix}.{args.score}.bedGraph"
    summary_path = out_dir / f"{prefix}.scan_summary.json"
    run_config_path = out_dir / f"{prefix}.resolved_run_config.json"

    fieldnames = [
        "chrom",
        "start",
        "end",
        "coord_key",
        "prob",
        "logit",
        "score_name",
        "score",
        "n_fraction",
        "window_length",
    ]
    if args.write_sequences:
        fieldnames.append("sequence")

    total_predicted = 0
    batch: list[WindowRecord] = []
    with open_text(windows_path, "wt") as windows_handle:
        writer = csv.DictWriter(windows_handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        bedgraph_writer = BedGraphWriter(bedgraph_path, args.track_bin_size or args.stride)
        try:
            iterator = iter_windows(
                genome,
                regions,
                window_size=args.window_size,
                stride=args.stride,
                max_n_frac=args.max_n_frac,
                include_terminal=args.include_terminal_window,
            )
            for record in tqdm(iterator, desc="Scanning windows", unit="window"):
                batch.append(record)
                if len(batch) >= args.batch_size:
                    total_predicted += _flush_batch(
                        predictor, batch, args.score, writer, bedgraph_writer, genome, args.write_sequences
                    )
                    batch = []
            total_predicted += _flush_batch(
                predictor, batch, args.score, writer, bedgraph_writer, genome, args.write_sequences
            )
        finally:
            bedgraph_writer.close()
            genome.close()

    summary = {
        "tool_name": __tool_name__,
        "tool_version": __version__,
        "fasta": str(Path(args.fasta)),
        "checkpoint_label": checkpoint_spec.label,
        "checkpoint": str(checkpoint_spec.checkpoint),
        "model_config": str(checkpoint_spec.config),
        "model_backend": checkpoint_spec.model_backend,
        "score": args.score,
        "window_size": args.window_size,
        "stride": args.stride,
        "track_bin_size": args.track_bin_size or args.stride,
        "max_n_frac": args.max_n_frac,
        "include_terminal_window": args.include_terminal_window,
        "regions": [region.name for region in regions],
        "predicted_windows": total_predicted,
        "bedgraph_intervals": bedgraph_writer.rows_written,
        "outputs": {
            "window_predictions": str(windows_path),
            "bedgraph": str(bedgraph_path),
            "summary": str(summary_path),
            "resolved_run_config": str(run_config_path),
        },
    }
    write_json(summary_path, summary)
    write_resolved_run_config(run_config_path, summary)
    return summary
