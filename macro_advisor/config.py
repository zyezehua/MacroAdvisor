"""Configuration loader.

Loads ``config/settings.yaml`` and ``config/universe.yaml``, applies an optional
local override (``config/settings.local.yaml``, gitignored), and exposes a typed-ish
accessor object. All paths are resolved relative to the repo root so the package
works regardless of the current working directory.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Config:
    """Merged, read-only view of settings + universe."""

    settings: dict[str, Any]
    universe: dict[str, Any]

    # -- convenience accessors -------------------------------------------
    @property
    def risk_budget(self) -> dict[str, Any]:
        return self.settings["risk_budget"]

    @property
    def horizons(self) -> dict[str, Any]:
        return self.settings["horizons"]

    @property
    def crosscheck(self) -> dict[str, Any]:
        return self.settings["crosscheck"]

    @property
    def storage(self) -> dict[str, Any]:
        return self.settings["storage"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.settings["backtest"]

    @property
    def signals(self) -> dict[str, Any]:
        return self.settings.get("signals", {})

    @property
    def stress(self) -> dict[str, Any]:
        return self.settings.get("stress", {})

    @property
    def remote(self) -> dict[str, Any]:
        return self.settings.get("remote", {})

    @property
    def predict(self) -> dict[str, Any]:
        return self.settings.get("predict", {})

    @property
    def recommend(self) -> dict[str, Any]:
        return self.settings.get("recommend", {})

    @property
    def sentiment(self) -> dict[str, Any]:
        return self.settings.get("sentiment", {})

    def path(self, key: str) -> Path:
        """Resolve a storage path key (e.g. 'parquet_dir', 'db_path') to an absolute Path."""
        return REPO_ROOT / self.storage[key]

    # -- runtime overrides -----------------------------------------------
    def with_overrides(self, patch: dict[str, Any]) -> "Config":
        """Return a new Config with ``patch`` deep-merged into ``settings``.

        Used by the dashboard to apply live risk-budget / ranking overrides without
        mutating the shared config (this Config is frozen; the universe is shared as-is).
        """
        if not patch:
            return self
        return Config(settings=_deep_merge(self.settings, patch), universe=self.universe)

    # -- universe helpers ------------------------------------------------
    #: sub-blocks whose 'symbol' entries are pulled by dedicated paths, not the
    #: generic price loop (yield mirror tickers, curve/fred docs).
    _NON_PRICE_BLOCKS = ("treasury_curve", "yield_mirror", "fred_optional", "fred")

    def yahoo_symbols(self, *tiers: str) -> list[str]:
        """Collect unique Yahoo price symbols across the given universe tiers.

        Excludes blocks handled by dedicated adapters (Treasury curve, yield mirror,
        FRED). Tiers default to the data-bearing groups if none are supplied.
        """
        tiers = tiers or ("core", "backtest_equity", "backtest_rates")
        symbols: list[str] = []
        for tier in tiers:
            block = self.universe.get(tier, {})
            symbols.extend(
                _collect_field(block, "symbol", exclude=self._NON_PRICE_BLOCKS)
            )
        return list(dict.fromkeys(symbols))

    def yield_mirror(self) -> list[dict[str, str]]:
        """Yahoo yield tickers used to cross-check the Treasury curve."""
        return list(self.universe.get("backtest_rates", {}).get("yield_mirror", []))

    def fred_optional(self) -> list[dict[str, str]]:
        """Best-effort FRED series (credit OAS, real yields); skipped if unreachable."""
        return list(self.universe.get("backtest_rates", {}).get("fred_optional", []))

    def fred_sentiment(self) -> list[dict[str, Any]]:
        """FRED hard-sentiment series (consumer sentiment, financial conditions/stress)."""
        return list(self.universe.get("sentiment", {}).get("fred_sentiment", []))

    def news_sources(self) -> list[dict[str, Any]]:
        """News-tone sources (GDELT queries today; extensible to a keyed news API later)."""
        return list(self.universe.get("sentiment", {}).get("news", []))


def _collect_field(node: Any, field: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """Walk a nested dict/list universe block and pull every ``field`` value,
    skipping any sub-dict reached via a key in ``exclude``."""
    found: list[str] = []
    if isinstance(node, dict):
        if field in node:
            found.append(node[field])
        else:
            for key, val in node.items():
                if key in exclude:
                    continue
                found.extend(_collect_field(val, field, exclude))
    elif isinstance(node, list):
        for item in node:
            found.extend(_collect_field(item, field, exclude))
    return found


@lru_cache(maxsize=1)
def load_config() -> Config:
    settings = _load_yaml(CONFIG_DIR / "settings.yaml")
    local = _load_yaml(CONFIG_DIR / "settings.local.yaml")
    if local:
        settings = _deep_merge(settings, local)
    universe = _load_yaml(CONFIG_DIR / "universe.yaml")
    return Config(settings=settings, universe=universe)
