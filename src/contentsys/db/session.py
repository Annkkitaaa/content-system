"""Database engine and sessions."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from contentsys.config import get_settings

# Imported for the side effect of registering every table on SQLModel.metadata
# before create_all runs. Without it, a fresh database comes up empty.
from contentsys.db import models as _models  # noqa: F401


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url
    if url.startswith("sqlite:///"):
        # Create the parent directory rather than failing on first run with an
        # unable-to-open-database error, which reads like a permissions problem
        # and is not one.
        path = Path(url.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, echo=False)


def create_all() -> None:
    """Create any missing tables.

    Fine for local use. Alembic owns schema changes once there is data worth
    keeping, which is why migrations exist from the first phase rather than
    being retrofitted after the first painful reset.
    """
    SQLModel.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional session that rolls back on failure."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine. For tests that repoint the database."""
    get_engine.cache_clear()
