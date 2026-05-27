from __future__ import annotations

from pathlib import Path


class GenomeFasta:
    """Small wrapper around pyfaidx with 0-based half-open coordinates."""

    def __init__(self, path: str | Path):
        try:
            from pyfaidx import Fasta
        except ImportError as exc:
            raise RuntimeError("pyfaidx is required for FASTA scanning: pip install pyfaidx") from exc

        self.path = Path(path)
        self._fasta = Fasta(str(self.path), as_raw=True, sequence_always_upper=True)
        self.names = list(self._fasta.keys())
        self.lengths = {name: len(self._fasta[name]) for name in self.names}

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.lengths:
            raise KeyError(f"Sequence not found in FASTA: {chrom}")
        start = max(0, int(start))
        end = min(int(end), self.lengths[chrom])
        if end <= start:
            return ""
        return str(self._fasta[chrom][start:end]).upper()

    def close(self) -> None:
        close = getattr(self._fasta, "close", None)
        if close is not None:
            close()

