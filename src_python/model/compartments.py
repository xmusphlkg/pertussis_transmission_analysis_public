from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VACCINE_ORIGINS: tuple[str, ...] = (
    "unvaccinated",
    "maternal",
    "dose1_recent",
    "dose1_waned",
    "dose2_recent",
    "dose2_waned",
    "recent",
    "waned",
)
STRAINS: tuple[str, ...] = ("S", "R")

SUSCEPTIBLE_COMPARTMENTS_BY_ORIGIN: dict[str, str] = {
    "unvaccinated": "S",
    "maternal": "M_protected",
    "dose1_recent": "V_dose1_recent",
    "dose1_waned": "V_dose1_waned",
    "dose2_recent": "V_dose2_recent",
    "dose2_waned": "V_dose2_waned",
    # Legacy names are retained as the 3+ dose states for output compatibility.
    "recent": "V_recent",
    "waned": "V_waned",
}

SUSCEPTIBLE_COMPARTMENTS: tuple[str, ...] = tuple(SUSCEPTIBLE_COMPARTMENTS_BY_ORIGIN.values())


def susceptible_name(origin: str) -> str:
    return SUSCEPTIBLE_COMPARTMENTS_BY_ORIGIN[origin]


def exposed_name(strain: str, origin: str) -> str:
    return f"E_{strain}_{origin}"


def infectious_name(strain: str, symptom: str, origin: str) -> str:
    return f"I_{strain}_{symptom}_{origin}"


def treated_name(strain: str, origin: str) -> str:
    return f"T_{strain}_{origin}"


COMPARTMENTS: tuple[str, ...] = (
    *SUSCEPTIBLE_COMPARTMENTS,
    *(exposed_name(strain, origin) for strain in STRAINS for origin in VACCINE_ORIGINS),
    *(
        infectious_name(strain, symptom, origin)
        for strain in STRAINS
        for symptom in ("sym", "asym")
        for origin in VACCINE_ORIGINS
    ),
    *(treated_name(strain, origin) for strain in STRAINS for origin in VACCINE_ORIGINS),
    "R_natural",
    "W_natural",
)

COMPARTMENT_ALIASES = {
    "V": "M_protected",
    "V_dose3plus_recent": "V_recent",
    "V_dose3plus_waned": "V_waned",
    "R": "R_natural",
    "W": "W_natural",
    "E_S": "E_S_unvaccinated",
    "E_R": "E_R_unvaccinated",
    "I_S_sym": "I_S_sym_unvaccinated",
    "I_S_asym": "I_S_asym_unvaccinated",
    "I_R_sym": "I_R_sym_unvaccinated",
    "I_R_asym": "I_R_asym_unvaccinated",
    "T_S": "T_S_unvaccinated",
    "T_R": "T_R_unvaccinated",
}


def compartment_name(name: str) -> str:
    return COMPARTMENT_ALIASES.get(name, name)


def compartment_index(name: str) -> int:
    return COMPARTMENTS.index(compartment_name(name))


@dataclass(frozen=True)
class StateIndex:
    age_groups: tuple[str, ...]

    @property
    def n_age(self) -> int:
        return len(self.age_groups)

    @property
    def n_compartments(self) -> int:
        return len(COMPARTMENTS)

    @property
    def size(self) -> int:
        return self.n_age * self.n_compartments

    def index(self, age: int | str, compartment: str) -> int:
        age_idx = self.age_groups.index(age) if isinstance(age, str) else age
        comp_idx = compartment_index(compartment)
        return age_idx * self.n_compartments + comp_idx

    def reshape(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float).reshape((self.n_age, self.n_compartments))

    def flatten(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).reshape(self.size)

    def compartment(self, state: np.ndarray, name: str) -> np.ndarray:
        return self.reshape(state)[:, compartment_index(name)]
