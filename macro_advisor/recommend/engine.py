"""Recommendation orchestrator: forecasts -> ranked ideas + risk-budgeted portfolio per horizon."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from macro_advisor.config import Config
from macro_advisor.data import MarketStore
from macro_advisor.recommend import portfolio, score


@dataclass
class HorizonRec:
    horizon: str
    ensemble: pd.DataFrame                       # ranked ideas (ensemble of models)
    per_model: dict[str, pd.DataFrame]           # ranked ideas per model
    allocation: pd.DataFrame                     # risk-budgeted portfolio
    summary: dict                                # gross/net/by-class/notional


@dataclass
class RecommendationResult:
    asof: str
    models: list[str]
    horizons: dict[str, HorizonRec] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.horizons


def recommend(store: MarketStore, forecast: pd.DataFrame, cfg: Config) -> RecommendationResult:
    """Build ranked ideas + portfolios for every horizon present in ``forecast``."""
    if forecast is None or forecast.empty:
        return RecommendationResult(asof="", models=[])
    models = sorted(forecast["model"].unique())
    asof = str(pd.Timestamp(forecast["date"].max()).date())
    recs: dict[str, HorizonRec] = {}
    for h in forecast["horizon"].unique():
        ens = score.score_and_rank(score.ensemble_frame(forecast, h), store, cfg)
        per = {m: score.score_and_rank(score.single_frame(forecast, h, m), store, cfg)
               for m in models}
        alloc, summary = portfolio.build_portfolio(ens, cfg)
        recs[h] = HorizonRec(horizon=h, ensemble=ens, per_model=per,
                             allocation=alloc, summary=summary)
    return RecommendationResult(asof=asof, models=models, horizons=recs)
