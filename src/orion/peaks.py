from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .utils import ensure_dir, open_text, write_json


@dataclass
class ScoreBin:
    chrom: str
    start: int
    end: int
    score: float


@dataclass
class Peak:
    chrom: str
    start: int
    end: int
    summit_start: int
    summit_end: int
    peak_score: float
    mean_score: float
    n_bins: int
    method: str
    cutoff: float
    prominence: float | None = None

    @property
    def width(self) -> int:
        return self.end - self.start


def read_bedgraph(path: str | Path) -> dict[str, list[ScoreBin]]:
    grouped: dict[str, list[ScoreBin]] = {}
    with open_text(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split()
            if len(fields) < 4:
                continue
            item = ScoreBin(fields[0], int(fields[1]), int(fields[2]), float(fields[3]))
            grouped.setdefault(item.chrom, []).append(item)
    for chrom in grouped:
        grouped[chrom].sort(key=lambda item: (item.start, item.end))
    return grouped


def _smooth(values: np.ndarray, smooth_bins: int) -> np.ndarray:
    if smooth_bins <= 1 or values.size == 0:
        return values
    width = min(int(smooth_bins), int(values.size))
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def _peak_from_span(
    bins: list[ScoreBin],
    start_idx: int,
    end_idx_exclusive: int,
    scores: np.ndarray,
    method: str,
    cutoff: float,
    prominence: float | None = None,
) -> Peak:
    span_scores = scores[start_idx:end_idx_exclusive]
    local_summit = int(np.argmax(span_scores)) + start_idx
    selected = bins[start_idx:end_idx_exclusive]
    return Peak(
        chrom=selected[0].chrom,
        start=selected[0].start,
        end=selected[-1].end,
        summit_start=bins[local_summit].start,
        summit_end=bins[local_summit].end,
        peak_score=float(scores[local_summit]),
        mean_score=float(np.mean(span_scores)),
        n_bins=len(selected),
        method=method,
        cutoff=cutoff,
        prominence=prominence,
    )


def call_threshold_peaks(
    grouped: dict[str, list[ScoreBin]],
    min_score: float,
    min_width: int,
    max_gap: int,
    smooth_bins: int,
    method: str = "threshold",
) -> list[Peak]:
    peaks: list[Peak] = []
    for bins in grouped.values():
        if not bins:
            continue
        raw_scores = np.array([item.score for item in bins], dtype=float)
        scores = _smooth(raw_scores, smooth_bins)
        active_start: int | None = None
        previous_idx: int | None = None
        for idx, item in enumerate(bins):
            passes = scores[idx] >= min_score
            separated = previous_idx is not None and item.start - bins[previous_idx].end > max_gap
            if passes and active_start is None:
                active_start = idx
            elif passes and separated:
                span_start = active_start if active_start is not None else idx
                peak = _peak_from_span(bins, span_start, idx, scores, method, min_score)
                if peak.width >= min_width:
                    peaks.append(peak)
                active_start = idx
            elif not passes and active_start is not None:
                peak = _peak_from_span(bins, active_start, idx, scores, method, min_score)
                if peak.width >= min_width:
                    peaks.append(peak)
                active_start = None
            if passes:
                previous_idx = idx
        if active_start is not None:
            peak = _peak_from_span(bins, active_start, len(bins), scores, method, min_score)
            if peak.width >= min_width:
                peaks.append(peak)
    return peaks


def call_scipy_peaks(
    grouped: dict[str, list[ScoreBin]],
    min_score: float,
    prominence: float,
    min_distance: int,
    min_width: int,
    max_gap: int,
    smooth_bins: int,
) -> list[Peak]:
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise RuntimeError("scipy is required for --method scipy. Use --method threshold or install scipy.") from exc

    peaks: list[Peak] = []
    for bins in grouped.values():
        if not bins:
            continue
        raw_scores = np.array([item.score for item in bins], dtype=float)
        scores = _smooth(raw_scores, smooth_bins)
        median_bin = max(1, int(np.median([max(1, item.end - item.start) for item in bins])))
        distance_bins = max(1, int(round(min_distance / median_bin)))
        peak_indices, properties = find_peaks(scores, height=min_score, prominence=prominence, distance=distance_bins)
        for peak_idx, prom in zip(peak_indices, properties.get("prominences", []), strict=False):
            left = int(peak_idx)
            right = int(peak_idx) + 1
            while left > 0 and scores[left - 1] >= min_score and bins[left].start - bins[left - 1].end <= max_gap:
                left -= 1
            while (
                right < len(bins)
                and scores[right] >= min_score
                and bins[right].start - bins[right - 1].end <= max_gap
            ):
                right += 1
            peak = _peak_from_span(bins, left, right, scores, "scipy", min_score, float(prom))
            if peak.width >= min_width:
                peaks.append(peak)
    return merge_nearby_peaks(peaks, max_gap=max_gap)


def merge_nearby_peaks(peaks: Iterable[Peak], max_gap: int) -> list[Peak]:
    sorted_peaks = sorted(peaks, key=lambda item: (item.chrom, item.start, item.end, -item.peak_score))
    merged: list[Peak] = []
    for peak in sorted_peaks:
        if not merged or peak.chrom != merged[-1].chrom or peak.start - merged[-1].end > max_gap:
            merged.append(peak)
            continue
        current = merged[-1]
        if peak.peak_score > current.peak_score:
            summit_start, summit_end, peak_score = peak.summit_start, peak.summit_end, peak.peak_score
        else:
            summit_start, summit_end, peak_score = current.summit_start, current.summit_end, current.peak_score
        total_bins = current.n_bins + peak.n_bins
        mean_score = (current.mean_score * current.n_bins + peak.mean_score * peak.n_bins) / max(1, total_bins)
        merged[-1] = Peak(
            chrom=current.chrom,
            start=min(current.start, peak.start),
            end=max(current.end, peak.end),
            summit_start=summit_start,
            summit_end=summit_end,
            peak_score=peak_score,
            mean_score=mean_score,
            n_bins=total_bins,
            method=current.method,
            cutoff=current.cutoff,
            prominence=max(current.prominence or 0.0, peak.prominence or 0.0),
        )
    return merged


def write_peaks(peaks: list[Peak], out_dir: Path, prefix: str) -> dict[str, str]:
    bed_path = out_dir / f"{prefix}.peaks.bed"
    summary_path = out_dir / f"{prefix}.peaks_summary.tsv"
    with open_text(bed_path, "wt") as handle:
        for idx, peak in enumerate(peaks, start=1):
            peak_id = f"{prefix}_peak_{idx:06d}"
            handle.write(
                "\t".join(
                    [
                        peak.chrom,
                        str(peak.start),
                        str(peak.end),
                        peak_id,
                        f"{peak.peak_score:.8g}",
                        ".",
                        str(peak.summit_start),
                        str(peak.summit_end),
                        f"{peak.mean_score:.8g}",
                        str(peak.n_bins),
                        peak.method,
                        f"{peak.cutoff:.8g}",
                        "" if peak.prominence is None else f"{peak.prominence:.8g}",
                    ]
                )
                + "\n"
            )
    fieldnames = [
        "peak_id",
        "chrom",
        "start",
        "end",
        "width",
        "summit_start",
        "summit_end",
        "peak_score",
        "mean_score",
        "n_bins",
        "method",
        "cutoff",
        "prominence",
    ]
    with open_text(summary_path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for idx, peak in enumerate(peaks, start=1):
            writer.writerow(
                {
                    "peak_id": f"{prefix}_peak_{idx:06d}",
                    "chrom": peak.chrom,
                    "start": peak.start,
                    "end": peak.end,
                    "width": peak.width,
                    "summit_start": peak.summit_start,
                    "summit_end": peak.summit_end,
                    "peak_score": f"{peak.peak_score:.8g}",
                    "mean_score": f"{peak.mean_score:.8g}",
                    "n_bins": peak.n_bins,
                    "method": peak.method,
                    "cutoff": f"{peak.cutoff:.8g}",
                    "prominence": "" if peak.prominence is None else f"{peak.prominence:.8g}",
                }
            )
    return {"peaks_bed": str(bed_path), "peaks_summary": str(summary_path)}


def run_macs3(args: Any) -> dict[str, Any]:
    out_dir = ensure_dir(args.output_dir)
    output_path = out_dir / f"{args.output_prefix}.macs3_bdgpeakcall.bed"
    command = [
        "macs3",
        "bdgpeakcall",
        "-i",
        str(args.score_track),
        "-o",
        str(output_path),
        "-c",
        str(args.peak_min_score),
        "-l",
        str(args.peak_min_width),
        "-g",
        str(args.peak_max_gap),
    ]
    subprocess.run(command, check=True)
    summary = {
        "method": "macs3",
        "score_track": str(args.score_track),
        "command": command,
        "outputs": {"macs3_peaks": str(output_path)},
    }
    summary_path = out_dir / f"{args.output_prefix}.peak_calling_summary.json"
    summary["outputs"]["peak_calling_summary"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def call_peaks(args: Any) -> dict[str, Any]:
    if args.method == "macs3":
        return run_macs3(args)
    out_dir = ensure_dir(args.output_dir)
    grouped = read_bedgraph(args.score_track)
    if args.method == "scipy":
        peaks = call_scipy_peaks(
            grouped,
            min_score=args.peak_min_score,
            prominence=args.peak_prominence,
            min_distance=args.peak_min_distance,
            min_width=args.peak_min_width,
            max_gap=args.peak_max_gap,
            smooth_bins=args.smooth_bins,
        )
    elif args.method == "threshold":
        peaks = call_threshold_peaks(
            grouped,
            min_score=args.peak_min_score,
            min_width=args.peak_min_width,
            max_gap=args.peak_max_gap,
            smooth_bins=args.smooth_bins,
        )
    else:
        raise ValueError(f"Unsupported peak calling method: {args.method}")
    outputs = write_peaks(peaks, out_dir, args.output_prefix)
    summary = {
        "method": args.method,
        "score_track": str(args.score_track),
        "peak_min_score": args.peak_min_score,
        "peak_prominence": args.peak_prominence if args.method == "scipy" else None,
        "peak_min_distance": args.peak_min_distance if args.method == "scipy" else None,
        "peak_min_width": args.peak_min_width,
        "peak_max_gap": args.peak_max_gap,
        "smooth_bins": args.smooth_bins,
        "n_chromosomes": len(grouped),
        "n_score_bins": sum(len(items) for items in grouped.values()),
        "n_peaks": len(peaks),
        "outputs": outputs,
    }
    summary_path = out_dir / f"{args.output_prefix}.peak_calling_summary.json"
    summary["outputs"]["peak_calling_summary"] = str(summary_path)
    write_json(summary_path, summary)
    return summary
