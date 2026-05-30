"""Shared ingest types."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PullResult:
    """Outcome of a single source fetch, carrying data + provenance."""

    symbol: str
    source: str
    kind: str                       # 'price' | 'series'
    df: pd.DataFrame | None = None
    status: str = "ok"              # 'ok' | 'empty' | 'error'
    message: str | None = None
    freq: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.symbol}"

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.df is not None and not self.df.empty

    def coverage(self) -> tuple[str | None, str | None, int]:
        if self.df is None or self.df.empty:
            return None, None, 0
        idx = self.df.index
        return (
            str(idx.min().date()),
            str(idx.max().date()),
            int(len(self.df)),
        )
