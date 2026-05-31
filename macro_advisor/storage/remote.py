"""Remote cache sync via Hugging Face Hub.

The deployed dashboard has no local ``data/`` cache (it's gitignored), so the cache lives in
a **public HF dataset repo**: a GitHub Actions cron refreshes the data and ``upload_cache``s
it; the app ``ensure_cache``s it on boot (anonymous read). This mirrors the protocol the user
runs for the Financial Market Stress Indicator project.

Only raw data is synced (parquet price/series frames + the SQLite provenance DB); the app
recomputes signals/stress on the fly, so there's nothing else to ship.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from macro_advisor.config import Config, load_config

log = logging.getLogger(__name__)

DEFAULT_REPO = "zyezehua/macroadvisor-cache"
#: files synced, relative to the data root — price/series parquet, provenance DB,
#: and the Phase-2 OOS prediction/backtest artifacts
_ALLOW = ["prices/*", "series/*", "*.sqlite", "oos/*"]
#: marker dropped in the data root when the cache was sourced from HF (vs a local pull),
#: so the app knows it's safe to periodically re-pull without clobbering local dev data.
_MARKER = ".hf_synced"


def resolve_repo(cfg: Config | None = None) -> str:
    """Repo id precedence: env MACROADVISOR_HF_REPO > config remote.hf_repo > default."""
    env = os.getenv("MACROADVISOR_HF_REPO")
    if env:
        return env
    cfg = cfg or load_config()
    return (cfg.remote or {}).get("hf_repo") or DEFAULT_REPO


def _data_root(cfg: Config) -> str:
    return str(cfg.path("root"))


def cache_present(cfg: Config | None = None) -> bool:
    """True if a usable local cache exists, i.e. there are cached price parquet files.

    Deliberately keyed on parquet — not the SQLite file — because constructing a
    ``MarketStore``/``ProvenanceDB`` creates an *empty* ``.sqlite`` as a side effect. On a
    persistent host (e.g. Streamlit Cloud across redeploys) that empty DB would otherwise be
    mistaken for a populated cache and suppress the HF download.
    """
    cfg = cfg or load_config()
    pdir = cfg.path("parquet_dir")
    return pdir.exists() and any(pdir.glob("*.parquet"))


def upload_cache(cfg: Config | None = None, *, token: str | None = None,
                 message: str = "Refresh MacroAdvisor cache") -> str:
    """Upload the local data cache to the HF dataset repo. Returns the repo id.

    Requires a write token (arg or ``HF_TOKEN`` env). Creates the repo if missing.
    """
    cfg = cfg or load_config()
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("no HF token: pass token= or set HF_TOKEN to upload")
    from huggingface_hub import HfApi

    repo = resolve_repo(cfg)
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=_data_root(cfg),
        path_in_repo=".",
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=_ALLOW,
        commit_message=message,
    )
    log.info("uploaded cache to https://huggingface.co/datasets/%s", repo)
    return repo


def download_cache(cfg: Config | None = None, *, token: str | None = None) -> str:
    """Download the HF dataset cache into the local data root. Returns the local path.

    Token is optional (the repo is public); a read token is only needed if it's private.
    """
    cfg = cfg or load_config()
    token = token or os.getenv("HF_TOKEN")
    from huggingface_hub import snapshot_download

    repo = resolve_repo(cfg)
    local = _data_root(cfg)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=local,
        allow_patterns=_ALLOW,
        token=token,
    )
    # mark the cache as HF-managed so the app may safely re-pull it on a schedule
    (Path(local) / _MARKER).write_text("")
    log.info("downloaded cache from %s -> %s", repo, local)
    return local


def ensure_cache(cfg: Config | None = None, *, token: str | None = None) -> bool:
    """Ensure a local cache exists, downloading from HF if absent. Returns True if downloaded."""
    cfg = cfg or load_config()
    if cache_present(cfg):
        return False
    download_cache(cfg, token=token)
    return True


def sync_for_app(cfg: Config | None = None, *, token: str | None = None) -> str:
    """Cache strategy for the deployed dashboard.

    - **Local dev** (a locally-pulled cache with no HF marker): leave it alone — never
      overwrite freshly pulled data, and don't require HF at all.
    - **Cloud** (no cache yet, or a previously HF-downloaded cache): (re)download the latest
      snapshot. ``snapshot_download`` only transfers changed files, so calling this on the
      app's cache TTL keeps the deployed app fresh with no reboot and no extra secrets.

    Returns ``"local"`` or ``"remote"``.
    """
    cfg = cfg or load_config()
    marker = cfg.path("root") / _MARKER
    if cache_present(cfg) and not marker.exists():
        return "local"
    download_cache(cfg, token=token)
    return "remote"
