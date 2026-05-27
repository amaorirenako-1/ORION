from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .fasta import GenomeFasta
from .utils import parse_csv_arg, read_lines, unique_in_order


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int

    @property
    def name(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


@dataclass(frozen=True)
class WindowRecord:
    chrom: str
    start: int
    end: int
    sequence: str
    n_fraction: float

    @property
    def coord_key(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


def parse_region_token(token: str, genome: GenomeFasta) -> Region:
    token = token.strip()
    if not token:
        raise ValueError("Empty region token.")
    if ":" not in token:
        if token not in genome.lengths:
            raise KeyError(f"Sequence not found in FASTA: {token}")
        return Region(token, 0, int(genome.lengths[token]))
    chrom, span = token.split(":", 1)
    if chrom not in genome.lengths:
        raise KeyError(f"Sequence not found in FASTA: {chrom}")
    if "-" not in span:
        raise ValueError(f"Region must look like chrom:start-end, got: {token}")
    raw_start, raw_end = span.replace(",", "").split("-", 1)
    start = max(0, int(raw_start))
    end = min(int(raw_end), int(genome.lengths[chrom]))
    if end <= start:
        raise ValueError(f"Region has no positive length after clipping: {token}")
    return Region(chrom, start, end)


def resolve_regions(
    genome: GenomeFasta,
    sequence_names: str | None = None,
    regions: str | None = None,
    regions_file: str | Path | None = None,
) -> list[Region]:
    tokens: list[str] = []
    tokens.extend(parse_csv_arg(sequence_names))
    tokens.extend(parse_csv_arg(regions))
    if regions_file:
        tokens.extend(read_lines(regions_file))
    tokens = unique_in_order(tokens)
    if not tokens:
        return [Region(name, 0, int(genome.lengths[name])) for name in genome.names]
    return [parse_region_token(token, genome) for token in tokens]


def _window_starts(region: Region, window_size: int, stride: int, include_terminal: bool) -> Iterable[int]:
    length = region.end - region.start
    if length < window_size:
        return []
    starts = list(range(region.start, region.end - window_size + 1, stride))
    terminal = region.end - window_size
    if include_terminal and starts and starts[-1] != terminal:
        starts.append(terminal)
    return starts


def iter_windows(
    genome: GenomeFasta,
    regions: Iterable[Region],
    window_size: int,
    stride: int,
    max_n_frac: float,
    include_terminal: bool = True,
) -> Iterator[WindowRecord]:
    if window_size <= 0:
        raise ValueError("--window-size must be positive.")
    if stride <= 0:
        raise ValueError("--stride must be positive.")
    for region in regions:
        for start in _window_starts(region, window_size, stride, include_terminal):
            end = start + window_size
            sequence = genome.fetch(region.chrom, start, end)
            if len(sequence) != window_size:
                continue
            n_fraction = sequence.count("N") / float(window_size)
            if n_fraction > max_n_frac:
                continue
            yield WindowRecord(region.chrom, start, end, sequence, n_fraction)

