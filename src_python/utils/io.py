from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(base)
    if not updates:
        return out
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def set_by_dotted_path(config: dict[str, Any], path: str, value: Any) -> None:
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def ensure_output_dirs() -> None:
    for relative in [
        "outputs/simulations",
        "outputs/summaries",
        "outputs/calibrations",
        "outputs/figures",
        "outputs/tables",
        "outputs/metadata",
        "manuscript_notes",
    ]:
        project_path(relative).mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to parquet (and optionally CSV).

    By default, large simulation outputs (outputs/simulations/) are written
    as parquet only to avoid multi-GB CSV files.  All other paths (summaries,
    tables, manuscript_notes, etc.) still receive both formats so that
    downstream R scripts and spreadsheet tools can read them directly.

    Set the environment variable PERTUSSIS_WRITE_CSV=1 to force CSV output
    for every file (e.g. for debugging or one-off exports).
    """
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine whether to write a companion CSV.
    force_csv = os.environ.get("PERTUSSIS_WRITE_CSV", "0").strip() == "1"
    # Simulation timeseries/full-run files are large; skip CSV unless forced.
    is_large_simulation = "outputs/simulations" in path.as_posix()
    write_csv = force_csv or not is_large_simulation
    csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        if write_csv:
            df.to_csv(csv_path, index=False)
        elif csv_path.exists():
            csv_path.unlink()
        return
    if path.suffix == ".csv":
        if write_csv:
            df.to_csv(path, index=False)
        elif path.exists():
            path.unlink()
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
        except Exception:
            pass
        return
    raise ValueError(f"Unsupported output suffix for {path}")


def write_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
    tmp_path.replace(path)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            if not path.exists() or parquet_path.stat().st_mtime >= path.stat().st_mtime:
                return pd.read_parquet(parquet_path)
    if path.exists():
        return pd.read_csv(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)
