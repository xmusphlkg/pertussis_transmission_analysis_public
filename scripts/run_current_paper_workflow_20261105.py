"""Workspace lock shared by the numerical analysis modules."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def workspace_lock():
    with (ROOT / '.paper-workflow.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another workspace job is running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
