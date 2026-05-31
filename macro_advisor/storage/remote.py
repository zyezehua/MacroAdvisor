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

from macro_advisor.config import Config, load_config

log = logging.getLogger(__name__)

DEFAULT_REPO = "zyezehua/macroadvisor-cache"
#: files synced, relative to the data root — price/series parquet + provenance DB
_ALLOW = ["prices/*", "series/*", "*.sqlite"]


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
    """True if a local cache already exists (DB file or any cached price parquet)."""
    cfg = cfg or load_config()
    if cfg.path("db_path").exists():
        return True
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
    log.info("downloaded cache from %s -> %s", repo, local)
    return local


def ensure_cache(cfg: Config | None = None, *, token: str | None = None) -> bool:
    """Ensure a local cache exists, downloading from HF if absent. Returns True if downloaded."""
    cfg = cfg or load_config()
    if cache_present(cfg):
        return False
    download_cache(cfg, token=token)
    return True
