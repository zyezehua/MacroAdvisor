"""Offline tests for the HF-Hub cache sync (huggingface_hub is mocked — no network)."""
from __future__ import annotations

import sys
import types

import pytest

from macro_advisor.config import Config
from macro_advisor.storage import remote


def _cfg(tmp_path, repo=None):
    storage = {
        "root": str(tmp_path),
        "parquet_dir": str(tmp_path / "prices"),
        "series_dir": str(tmp_path / "series"),
        "db_path": str(tmp_path / "ma.sqlite"),
    }
    settings = {"storage": storage}
    if repo is not None:
        settings["remote"] = {"hf_repo": repo}
    return Config(settings=settings, universe={})


@pytest.fixture
def fake_hf(monkeypatch):
    """Install a fake huggingface_hub module that records calls instead of hitting the network."""
    calls = {"create_repo": [], "upload_folder": [], "snapshot_download": []}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, repo_id, **kw):
            calls["create_repo"].append((repo_id, kw))

        def upload_folder(self, **kw):
            calls["upload_folder"].append(kw)

    def snapshot_download(**kw):
        calls["snapshot_download"].append(kw)
        return kw.get("local_dir", "")

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = FakeApi
    mod.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


# -- repo resolution ---------------------------------------------------------
def test_resolve_repo_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("MACROADVISOR_HF_REPO", raising=False)
    assert remote.resolve_repo(_cfg(tmp_path)) == remote.DEFAULT_REPO
    assert remote.resolve_repo(_cfg(tmp_path, repo="me/cfg-repo")) == "me/cfg-repo"
    monkeypatch.setenv("MACROADVISOR_HF_REPO", "me/env-repo")
    assert remote.resolve_repo(_cfg(tmp_path, repo="me/cfg-repo")) == "me/env-repo"


# -- upload ------------------------------------------------------------------
def test_upload_cache_calls_upload_folder(tmp_path, fake_hf):
    cfg = _cfg(tmp_path, repo="me/cache")
    repo = remote.upload_cache(cfg, token="tok")
    assert repo == "me/cache"
    assert fake_hf["create_repo"][0][0] == "me/cache"
    kw = fake_hf["upload_folder"][0]
    assert kw["repo_id"] == "me/cache" and kw["repo_type"] == "dataset"
    assert kw["folder_path"] == str(tmp_path)
    assert kw["allow_patterns"] == remote._ALLOW


def test_upload_requires_token(tmp_path, fake_hf, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF token"):
        remote.upload_cache(_cfg(tmp_path), token=None)


# -- download / ensure -------------------------------------------------------
def test_download_cache_calls_snapshot(tmp_path, fake_hf):
    cfg = _cfg(tmp_path, repo="me/cache")
    remote.download_cache(cfg)
    kw = fake_hf["snapshot_download"][0]
    assert kw["repo_id"] == "me/cache" and kw["repo_type"] == "dataset"
    assert kw["local_dir"] == str(tmp_path)


def test_ensure_cache_skips_when_present(tmp_path, fake_hf):
    cfg = _cfg(tmp_path)
    (tmp_path / "prices").mkdir(parents=True)
    (tmp_path / "prices" / "SPY.parquet").write_bytes(b"x")   # cache present
    assert remote.ensure_cache(cfg) is False
    assert fake_hf["snapshot_download"] == []                 # no download attempted


def test_ensure_cache_downloads_when_absent(tmp_path, fake_hf):
    cfg = _cfg(tmp_path)
    assert remote.ensure_cache(cfg) is True
    assert len(fake_hf["snapshot_download"]) == 1


def test_bare_sqlite_is_not_cache_present(tmp_path, fake_hf):
    """An empty .sqlite (created as a side effect of MarketStore) must NOT count as a cache,
    or the cloud app would skip the HF download (regression: live 'Cache sync: local' bug)."""
    cfg = _cfg(tmp_path)
    (tmp_path / "ma.sqlite").write_bytes(b"")          # bare DB, no parquet
    assert remote.cache_present(cfg) is False
    assert remote.sync_for_app(cfg) == "remote"        # downloads instead of skipping
    assert len(fake_hf["snapshot_download"]) == 1


def test_download_cache_writes_marker(tmp_path, fake_hf):
    cfg = _cfg(tmp_path)
    remote.download_cache(cfg)
    assert (tmp_path / remote._MARKER).exists()


# -- sync_for_app (app cache strategy) ---------------------------------------
def test_sync_for_app_leaves_local_cache(tmp_path, fake_hf):
    """A locally-pulled cache (no HF marker) must not be touched."""
    cfg = _cfg(tmp_path)
    (tmp_path / "prices").mkdir(parents=True)
    (tmp_path / "prices" / "SPY.parquet").write_bytes(b"x")
    assert remote.sync_for_app(cfg) == "local"
    assert fake_hf["snapshot_download"] == []


def test_sync_for_app_repulls_when_marked(tmp_path, fake_hf):
    """An HF-managed cache (marker present) is re-pulled even though it exists."""
    cfg = _cfg(tmp_path)
    (tmp_path / "prices").mkdir(parents=True)
    (tmp_path / "prices" / "SPY.parquet").write_bytes(b"x")
    (tmp_path / remote._MARKER).write_text("")
    assert remote.sync_for_app(cfg) == "remote"
    assert len(fake_hf["snapshot_download"]) == 1


def test_sync_for_app_downloads_when_absent(tmp_path, fake_hf):
    cfg = _cfg(tmp_path)
    assert remote.sync_for_app(cfg) == "remote"
    assert len(fake_hf["snapshot_download"]) == 1
