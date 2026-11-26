import re
from pathlib import Path

import huggingface_hub

from src import cloud_setup
from src.cloud_setup import ARTIFACT_FILES, ensure_artifacts, ensure_data, missing_artifacts
from src.download import SHA256

ROOT = Path(__file__).resolve().parents[1]


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_missing_artifacts_lists_only_absent_files(tmp_path):
    touch(tmp_path / ARTIFACT_FILES[0])
    assert missing_artifacts(tmp_path) == ARTIFACT_FILES[1:]


def test_nothing_is_downloaded_when_everything_exists(tmp_path, monkeypatch):
    for name in ARTIFACT_FILES:
        touch(tmp_path / name)
    for name in SHA256:
        touch(tmp_path / "data" / name)

    def fail(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    monkeypatch.setattr(cloud_setup, "download", fail)
    assert ensure_artifacts(tmp_path) is False
    assert ensure_data(tmp_path / "data") is False


def test_missing_artifacts_are_fetched_from_the_pinned_revision(tmp_path, monkeypatch):
    touch(tmp_path / "hybrid_config.json")
    calls = []

    def fake_download(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        touch(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert ensure_artifacts(tmp_path) is True
    assert {f for _, f, _ in calls} == set(ARTIFACT_FILES) - {"hybrid_config.json"}
    assert {(r, rev) for r, _, rev in calls} == {(cloud_setup.ARTIFACT_REPO, cloud_setup.ARTIFACT_REVISION)}
    assert cloud_setup.ARTIFACT_REVISION and len(cloud_setup.ARTIFACT_REVISION) == 40
    assert missing_artifacts(tmp_path) == []


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_streamlit_cloud_requirements_match_the_main_pins():
    main = _pins(ROOT / "requirements.txt")
    cloud = _pins(ROOT / "deploy" / "streamlit" / "requirements.txt")
    shared = set(main) & set(cloud)
    assert shared, "no shared packages found"
    assert {p: cloud[p] for p in shared} == {p: main[p] for p in shared}
    torch_lines = [l for l in (ROOT / "deploy" / "streamlit" / "requirements.txt").read_text().splitlines()
                   if l.startswith("torch @")]
    assert all(f"torch-{main['torch']}%2Bcpu" in l for l in torch_lines) and len(torch_lines) == 3
