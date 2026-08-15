from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

from contentsys.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests away from the real database and the real .env.

    Without this, a test run would happily write into the owner's live
    knowledge base, which is exactly the kind of thing you only notice after
    it has happened.
    """
    monkeypatch.setenv("CONTENTSYS_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("CONTENTSYS_EXPORT_DIR", str(tmp_path / "exports"))
    os.environ.pop("CONTENTSYS_PROVIDER", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def session() -> Iterator[Session]:
    """An in-memory database with every table created."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
