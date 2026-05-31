"""Ingestion pipeline.

Orchestrates: adapter fetch -> cross-check -> parquet cache -> provenance record.
The pipeline is the single entry point the scheduler and CLI call. It never lets an
unreconciled series pass silently — every pull writes provenance and any QA flags.
"""
from __future__ import annotations

import logging

import pandas as pd

from macro_advisor.config import Config, load_config
from macro_advisor.crosscheck import check_series, reconcile_levels
from macro_advisor.ingest import fred, gdelt, treasury, yahoo
from macro_advisor.ingest.base import PullResult
from macro_advisor.storage import ParquetCache, ProvenanceDB

log = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, config: Config | None = None):
        self.cfg = config or load_config()
        self.cache = ParquetCache(
            self.cfg.path("parquet_dir"), self.cfg.path("series_dir")
        )
        self.db = ProvenanceDB(self.cfg.path("db_path"))
        self._cc = self.cfg.crosscheck

    # -- internal: persist + record -------------------------------------
    def _commit(self, res: PullResult, flags) -> None:
        start, end, n = res.coverage()
        if res.ok:
            self.cache.write(res.symbol, res.df, kind=res.kind)
        self.db.record_pull(
            key=res.key, symbol=res.symbol, source=res.source, kind=res.kind,
            status=res.status, start_date=start, end_date=end, n_rows=n,
            freq=res.freq, message=res.message,
        )
        self.db.clear_flags(res.key)
        for f in flags:
            self.db.raise_flag(key=res.key, code=f.code, severity=f.severity,
                               detail=f.detail)

    def _series_flags(self, df) -> list:
        return check_series(
            df,
            staleness_days=self._cc["staleness_days"],
            min_history_days=self._cc["min_history_days"],
        )

    # -- public pulls ----------------------------------------------------
    def pull_equity(self, symbol: str, start: str | None = None,
                    end: str | None = None, mirror: bool = False) -> PullResult:
        """Pull a Yahoo price series and persist with single-series QA checks.

        (``mirror`` is retained for an optional keyed external mirror; the default
        keyless integrity check is OHLC self-consistency inside ``check_series``.)
        """
        res = yahoo.fetch(symbol, start=start, end=end)
        flags = self._series_flags(res.df) if res.ok else []
        self._commit(res, flags)
        return res

    def pull_treasury(self, start: str | None = None,
                      end: str | None = None) -> dict[str, PullResult]:
        """Pull the Treasury par-yield curve, split per tenor, reconcile vs Yahoo."""
        start = start or self.cfg.universe["meta"]["history_start"]
        wide = treasury.fetch_history(start=start, end=end)
        results: dict[str, PullResult] = {}

        # build the Yahoo yield mirror once (tenor id -> yield series in percent)
        mirror_map: dict[str, pd.Series] = {}
        for m in self.cfg.yield_mirror():
            yr = yahoo.fetch(m["symbol"], start=start, end=end)
            if yr.ok:
                col = "close" if "close" in yr.df.columns else yr.df.columns[0]
                mirror_map[m["maps_to"]] = yr.df[col]

        if wide.empty:
            res = PullResult("UST_CURVE", treasury.SOURCE, "series",
                             status="error", message="curve fetch failed")
            self._commit(res, [])
            return {res.key: res}

        for tenor in wide.columns:
            ser = wide[tenor].dropna().to_frame("value")
            res = PullResult(tenor, treasury.SOURCE, "series", df=ser,
                             status="ok" if not ser.empty else "empty", freq="D")
            flags = self._series_flags(ser) if res.ok else []
            if res.ok and tenor in mirror_map:
                flags += reconcile_levels(
                    ser["value"], mirror_map[tenor],
                    abs_tol=self._cc["yield_abs_tol_bps"] / 100.0,  # bps -> percent
                    code="YIELD_DIVERGENCE",
                )
            self._commit(res, flags)
            results[res.key] = res
        return results

    def pull_fred_optional(self, start: str | None = None) -> dict[str, PullResult]:
        """Best-effort FRED extras (credit OAS, real yields). Degrades silently."""
        results: dict[str, PullResult] = {}
        for item in self.cfg.fred_optional():
            res = fred.fetch(item["series"], start=start)
            flags = self._series_flags(res.df) if res.ok else []
            self._commit(res, flags)
            results[res.key] = res
        return results

    def pull_sentiment(self, start: str | None = None) -> dict[str, PullResult]:
        """Best-effort news/sentiment: FRED hard-sentiment series + GDELT news tone.

        Single-source (no cross-check mirror); degrades silently when a source is
        unreachable. FRED series use the keyless CSV endpoint; GDELT the keyless DOC API.
        """
        results: dict[str, PullResult] = {}
        for item in self.cfg.fred_sentiment():
            res = fred.fetch(item["series"], start=start)
            flags = self._series_flags(res.df) if res.ok else []
            self._commit(res, flags)
            results[res.key] = res
        gcfg = (self.cfg.sentiment.get("gdelt") or {})
        months = int(gcfg.get("timespan_months", 24))
        for item in self.cfg.news_sources():
            res = gdelt.fetch(item["label"], item["query"], timespan_months=months)
            flags = self._series_flags(res.df) if res.ok else []
            self._commit(res, flags)
            results[res.key] = res
        return results

    # -- batch runs ------------------------------------------------------
    def run(self, tiers: tuple[str, ...], start: str | None = None,
            fred_extras: bool = False, sentiment: bool = False) -> dict[str, PullResult]:
        """Pull every Yahoo price symbol in the given tiers; add the Treasury curve
        (and optional FRED extras) when the rates tier is included; add news/sentiment
        series when ``sentiment`` is set."""
        start = start or self.cfg.universe["meta"]["history_start"]
        results: dict[str, PullResult] = {}

        for sym in self.cfg.yahoo_symbols(*tiers):
            res = self.pull_equity(sym, start=start)
            results[res.key] = res
            log.info("yahoo %-10s %-6s rows=%s", sym, res.status, res.coverage()[2])

        if "backtest_rates" in tiers:
            for key, res in self.pull_treasury(start=start).items():
                results[key] = res
                log.info("ust   %-10s %-6s rows=%s",
                         res.symbol, res.status, res.coverage()[2])
            if fred_extras:
                for key, res in self.pull_fred_optional(start=start).items():
                    results[key] = res

        if sentiment:
            for key, res in self.pull_sentiment(start=start).items():
                results[key] = res
                log.info("sent  %-10s %-6s rows=%s",
                         res.symbol, res.status, res.coverage()[2])
        return results

    def run_core(self, **kw) -> dict[str, PullResult]:
        """Light/intraday scope: core stress universe only."""
        kw.pop("fred_extras", None)
        kw.pop("sentiment", None)
        return self.run(("core",), **kw)

    def run_full(self, **kw) -> dict[str, PullResult]:
        """Full scope: core + backtest equity + backtest rates (Treasury curve)."""
        kw.setdefault("sentiment", True)
        return self.run(("core", "backtest_equity", "backtest_rates"), **kw)

    def close(self) -> None:
        self.db.close()
