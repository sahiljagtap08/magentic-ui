"""Regression tests: client-controlled ``user_id`` must never escape
``<app_dir>/files/user`` when it is turned into an on-disk run directory.

Before the fix, ``POST /api/sessions`` with
``{"user_id": "../../../../tmp/evil"}`` produced a run whose directory
(and therefore upload target and sandbox workspace mount) resolved to
``/tmp/evil/<sid>/<rid>``.
"""

from __future__ import annotations

# httpx/TestClient responses are untyped under strict mode.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownLambdaType=false, reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magentic_ui.backend.datamodel import Run, RunStatus, Session
from magentic_ui.backend.teammanager._upload_validation import validate_uploaded_files
from magentic_ui.backend.teammanager.teammanager import TeamManager
from magentic_ui.backend.utils.utils import (
    RunPathError,
    build_run_dir,
    validate_user_id,
)
from magentic_ui.backend.web.deps import get_db
from magentic_ui.backend.web.routes import sessions as sessions_routes


_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# build_run_dir / validate_user_id
# ---------------------------------------------------------------------------


class TestBuildRunDir:
    def test_normal_run(self, tmp_path: Path) -> None:
        run = Run(
            id=3,
            session_id=2,
            created_at=_NOW,
            status=RunStatus.CREATED,
            user_id="u@x.com",
        )
        assert build_run_dir(tmp_path, run) == tmp_path / "files/user/u@x.com/2/3"

    def test_missing_fields_use_placeholders(self, tmp_path: Path) -> None:
        run = Run(created_at=_NOW, status=RunStatus.CREATED)
        assert (
            build_run_dir(tmp_path, run)
            == tmp_path / "files/user/unknown_user/unknown_session/unknown_run"
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "../../../../tmp/evil",
            "..",
            ".",
            "a/b",
            "a\\b",
            "/abs",
            "nul\x00byte",
            " ",
        ],
    )
    def test_rejects_traversal_user_id(self, tmp_path: Path, bad: str) -> None:
        run = Run(
            id=1, session_id=1, created_at=_NOW, status=RunStatus.CREATED, user_id=bad
        )
        with pytest.raises(RunPathError):
            build_run_dir(tmp_path, run)

    def test_result_always_under_files_user(self, tmp_path: Path) -> None:
        run = Run(
            id=1, session_id=1, created_at=_NOW, status=RunStatus.CREATED, user_id="..x"
        )
        out = build_run_dir(tmp_path, run)
        assert out.resolve().is_relative_to((tmp_path / "files" / "user").resolve())


class TestValidateUserId:
    def test_accepts_email_like(self) -> None:
        assert validate_user_id("guestuser@gmail.com") == "guestuser@gmail.com"

    @pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "..", ".", "", "x\x00"])
    def test_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_user_id(bad)


# ---------------------------------------------------------------------------
# TeamManager.prepare_host_run_dir
# ---------------------------------------------------------------------------


def test_prepare_host_run_dir_refuses_escape(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    outside = tmp_path / "evil"
    tm = TeamManager(app_dir=app_dir)
    run = Run(
        id=1,
        session_id=1,
        created_at=_NOW,
        status=RunStatus.CREATED,
        user_id="../../evil",
    )
    with pytest.raises(RunPathError):
        tm.prepare_host_run_dir(run=run)
    assert not outside.exists()


def test_prepare_host_run_dir_creates_inside(tmp_path: Path) -> None:
    tm = TeamManager(app_dir=tmp_path)
    run = Run(
        id=7, session_id=5, created_at=_NOW, status=RunStatus.CREATED, user_id="u"
    )
    out = tm.prepare_host_run_dir(run=run)
    assert out == tmp_path / "files/user/u/5/7"
    assert out.is_dir()


# ---------------------------------------------------------------------------
# validate_uploaded_files uses the same containment rule
# ---------------------------------------------------------------------------


def test_validate_uploaded_files_rejects_all_when_run_escapes(tmp_path: Path) -> None:
    run = Run(
        id=1,
        session_id=1,
        created_at=_NOW,
        status=RunStatus.CREATED,
        user_id="../../evil",
    )
    files: list[dict[str, Any]] = [{"name": "a.txt", "path": "files/user/x/1/1/a.txt"}]
    assert validate_uploaded_files(tmp_path, files, run=run) == []


# ---------------------------------------------------------------------------
# /api/sessions routes
# ---------------------------------------------------------------------------


def _make_client(db: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(sessions_routes.router, prefix="/sessions")
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _ok(data: Any) -> MagicMock:
    resp = MagicMock()
    resp.status = True
    resp.data = data
    resp.message = "ok"
    return resp


def test_create_session_rejects_traversal_user_id() -> None:
    db = MagicMock()
    client = _make_client(db)
    r = client.post("/sessions/", json={"name": "x", "user_id": "../../../../tmp/evil"})
    assert r.status_code == 400
    db.upsert.assert_not_called()


def test_create_session_ignores_client_supplied_id() -> None:
    db = MagicMock()
    db.upsert.side_effect = lambda model, **kw: _ok(
        {"id": 99, "user_id": model.user_id} if isinstance(model, Session) else model
    )
    client = _make_client(db)
    r = client.post("/sessions/", json={"id": 7, "name": "x", "user_id": "u@x.com"})
    assert r.status_code == 200
    created = db.upsert.call_args_list[0].args[0]
    assert isinstance(created, Session)
    assert created.id is None  # must not overwrite session 7


def test_update_session_pins_id_and_user_to_path() -> None:
    db = MagicMock()
    db.get.return_value = _ok([{"id": 1, "user_id": "me"}])
    db.upsert.side_effect = lambda model, **kw: _ok(model)
    client = _make_client(db)
    r = client.put(
        "/sessions/1?user_id=me",
        json={"id": 42, "name": "renamed", "user_id": "someone-else"},
    )
    assert r.status_code == 200
    updated = db.upsert.call_args.args[0]
    assert updated.id == 1
    assert updated.user_id == "me"
