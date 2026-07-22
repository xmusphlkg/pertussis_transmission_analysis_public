from __future__ import annotations

import pytest

from src_python.simulation.audit_joint_mode_bank import _parse_origin_spec


def test_parse_origin_spec_preserves_ordinary_stem() -> None:
    assert _parse_origin_spec("completed_run") == ("completed_run", None)


def test_parse_origin_spec_reads_positive_chain_selector() -> None:
    assert _parse_origin_spec("completed_audit::chain=4") == (
        "completed_audit",
        4,
    )


@pytest.mark.parametrize(
    "origin_spec",
    ["::chain=1", "completed::chain=", "completed::chain=x", "completed::chain=0"],
)
def test_parse_origin_spec_rejects_invalid_chain_selector(
    origin_spec: str,
) -> None:
    with pytest.raises(ValueError):
        _parse_origin_spec(origin_spec)
