from __future__ import annotations


def treated_infectiousness_relative(treatment: dict, strain: str) -> float:
    key = "sensitive" if strain == "S" else "resistant"
    return max(0.0, 1.0 - float(treatment[key]["infectiousness_reduction"]))


def treated_recovery_rate(base_recovery_rate: float, treatment: dict, strain: str) -> float:
    key = "sensitive" if strain == "S" else "resistant"
    duration_factor = max(0.05, 1.0 - float(treatment[key]["infectious_duration_reduction"]))
    return base_recovery_rate / duration_factor
