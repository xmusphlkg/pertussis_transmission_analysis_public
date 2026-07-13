"""Release checks for current negative-control and stress-test artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import file_sha256, validate_run_metadata
from src_python.utils.io import project_path


BLOCK_STEM = "panel_pomp_block_stress"
BLOCK_GATE_PATH = project_path("outputs", "tables", f"{BLOCK_STEM}_gate.csv")


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise RuntimeError(f"{name} is not a complete Boolean: {value!r}")


def require_current_block_stress_result() -> dict[str, Any]:
    """Require the current, fully executed block stress test and its known failure."""

    metadata = validate_run_metadata(BLOCK_STEM)
    if not BLOCK_GATE_PATH.exists():
        raise FileNotFoundError(f"Block stress gate is missing: {BLOCK_GATE_PATH}")
    gate = pd.read_csv(BLOCK_GATE_PATH)
    if len(gate) != 1:
        raise RuntimeError(f"Block stress gate must have one row, found {len(gate)}")
    row = gate.iloc[0]
    required = {
        "execution_complete",
        "data_integrity_gate_pass",
        "primary_predictive_scope",
        "predictive_gate_pass",
        "publication_gate_pass",
        "full_compartment_pomp_claimed",
        "posterior_parameter_uncertainty_claimed",
    }
    missing = sorted(required.difference(gate.columns))
    if missing:
        raise RuntimeError(f"Block stress gate is missing columns: {missing}")
    if not _boolean(row["execution_complete"], name="execution_complete"):
        raise RuntimeError("Block stress execution is incomplete")
    if not _boolean(row["data_integrity_gate_pass"], name="data_integrity_gate_pass"):
        raise RuntimeError("Block stress data-integrity gate failed")
    if str(row["primary_predictive_scope"]) != "one_year_block":
        raise RuntimeError("Block stress artifact has the wrong predictive scope")
    if _boolean(row["predictive_gate_pass"], name="predictive_gate_pass"):
        raise RuntimeError(
            "The one-year block stress test now passes; manuscript failure claims must be refreshed"
        )
    if _boolean(row["publication_gate_pass"], name="publication_gate_pass"):
        raise RuntimeError("Block stress artifact unexpectedly claims publication readiness")
    if _boolean(row["full_compartment_pomp_claimed"], name="full_compartment_pomp_claimed"):
        raise RuntimeError("Block stress artifact is mislabelled as a full compartment POMP")
    if _boolean(
        row["posterior_parameter_uncertainty_claimed"],
        name="posterior_parameter_uncertainty_claimed",
    ):
        raise RuntimeError("Block stress artifact is mislabelled as posterior uncertainty")
    if str(metadata.get("forecast_mode")) != "block":
        raise RuntimeError("Block stress metadata has the wrong forecast mode")
    output_hashes = metadata.get("output_artifact_sha256")
    expected_hash = (
        output_hashes.get("panel_pomp_block_stress_gate")
        if isinstance(output_hashes, dict)
        else None
    )
    if expected_hash != file_sha256(BLOCK_GATE_PATH):
        raise RuntimeError("Block stress gate hash is stale")
    return metadata
