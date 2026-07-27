"""Database engine, session factory, and transaction scope.

One engine per process, created lazily and cached: the engine owns a
connection pool, and constructing more than one wastes connections and
defeats pooling.

Sessions are handed out through a context manager that commits on success
and rolls back on failure. Any write path that raises halfway through must
leave nothing behind — a partial audit trail is worse than none, because it
looks complete.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rti_engine.config.settings import get_settings

PSYCOPG_PREFIX = "postgresql+psycopg://"
PLAIN_PREFIX = "postgresql://"


def normalise_dsn(dsn: str) -> str:
    """Make the DSN name psycopg 3 explicitly.

    SQLAlchemy treats a bare ``postgresql://`` URL as psycopg 2, which this
    project does not install. Rewriting here keeps the driver choice out of
    configuration, where it would be an implementation detail leaking into
    an environment variable.
    """
    if dsn.startswith(PSYCOPG_PREFIX):
        return dsn
    if dsn.startswith(PLAIN_PREFIX):
        return PSYCOPG_PREFIX + dsn[len(PLAIN_PREFIX) :]
    return dsn


def resolve_dsn() -> str:
    """Return the configured database URL, or fail loudly."""
    dsn = get_settings().postgres_dsn
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is not set; check your .env file")
    return normalise_dsn(dsn)


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use.

    ``pool_pre_ping`` tests a pooled connection before handing it out, which
    avoids failures from connections the database has closed while idle.
    """
    return create_engine(resolve_dsn(), pool_pre_ping=True, future=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory.

    ``expire_on_commit=False`` lets objects stay readable after the
    transaction closes, which matters when a result is returned from inside
    a session scope.
    """
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session, committing or rolling back on exit."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
