from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterable
from typing import Callable, TypeVar

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


T = TypeVar("T")
R = TypeVar("R")


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_worker_thread_limits() -> None:
    """Prevent each worker process from spawning a full BLAS thread pool."""
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_n_jobs(requested: int | None = None) -> int:
    env_value = os.environ.get("PERTUSSIS_N_JOBS")
    if env_value:
        requested = int(env_value)
    if requested is None:
        requested = -1

    cpus = available_cpus()
    if requested == 0:
        return 1
    if requested < 0:
        return max(1, cpus + 1 + requested)
    return max(1, min(int(requested), cpus))


@contextmanager
def tqdm_joblib(progress_bar):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        progress_bar.close()


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    desc: str,
    n_jobs: int | None = None,
) -> list[R]:
    item_list = list(items)
    workers = min(resolve_n_jobs(n_jobs), max(1, len(item_list)))
    if workers == 1:
        return [func(item) for item in tqdm(item_list, desc=desc)]

    configure_worker_thread_limits()
    with tqdm_joblib(tqdm(total=len(item_list), desc=f"{desc} ({workers} workers)")):
        return Parallel(n_jobs=workers, backend="loky", inner_max_num_threads=1)(
            delayed(func)(item) for item in item_list
        )
