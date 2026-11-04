"""deploy/hf-space/app.py: the Hugging Face Space entry point (no network, no server)."""

import importlib.util
import io
import re
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[3] / "deploy" / "hf-space"


@pytest.fixture(scope="module")
def space_app():
    spec = importlib.util.spec_from_file_location("hf_space_app", SPACE / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requirements_pin_one_commit(space_app):
    sha = space_app.pinned_sha()
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    text = (SPACE / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==" in text and "+cpu" in text
    assert "#subdirectory=packages/rag_core" in text
    assert "#subdirectory=apps/support_copilot" in text


def test_pinned_sha_rejects_two_commits(space_app, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(f"a @ git+https://github.com/{space_app.REPO}@{'a' * 40}\n"
                   f"b @ git+https://github.com/{space_app.REPO}@{'b' * 40}\n")
    with pytest.raises(RuntimeError):
        space_app.pinned_sha(req)


def test_demo_files_exist_in_repo(space_app):
    demo = Path(__file__).resolve().parents[1] / "demo"
    for name in space_app.DEMO_FILES:
        assert (demo / name).is_file(), name
    assert {p.name for p in (demo / "kb").glob("*.txt")} == {
        Path(n).name for n in space_app.DEMO_FILES if n.startswith("kb/")
    }


def test_fetch_demo_data_downloads_once(space_app, tmp_path):
    urls = []

    def opener(url, timeout):
        urls.append(url)
        return io.BytesIO(url.encode())

    space_app.fetch_demo_data("c" * 40, tmp_path, opener=opener)
    space_app.fetch_demo_data("c" * 40, tmp_path, opener=opener)  # restart: cached

    assert len(urls) == len(space_app.DEMO_FILES)
    assert urls[0] == (f"https://raw.githubusercontent.com/{space_app.REPO}/{'c' * 40}/"
                       f"apps/support_copilot/demo/kb/delivery.txt")
    assert (tmp_path / "shop_tickets.json").is_file()


def test_configure_keeps_space_settings(space_app, tmp_path, monkeypatch):
    # A private environment: configure() must not leak demo settings into other tests.
    env = {"RATE_LIMIT_PER_HOUR": "5"}
    monkeypatch.setattr(space_app.os, "environ", env)
    space_app.configure(tmp_path / "demo", tmp_path / "data")

    assert env["RATE_LIMIT_PER_HOUR"] == "5"
    assert env["LLM_PROVIDER"] == "openai"
    assert env["DEMO_KB_DIR"] == str(tmp_path / "demo" / "kb")
