from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src_python.simulation.common import current_run_metadata, load_configs, write_run_metadata
from src_python.utils.io import project_path, write_dataframe


AGE_GROUPS = (
    "infant_0_2m",
    "infant_3_11m",
    "child_1_4y",
    "child_5_9y",
    "adolescent_10_17y",
    "young_adult_18_39y",
    "middle_adult_40_64y",
    "elderly_65plus",
)
INFANT_GROUPS = {"infant_0_2m", "infant_3_11m"}
SETTINGS = ("home", "school", "work", "other")
SCENARIOS = (
    "homogeneous_all_contacts",
    "setting_clustered",
    "setting_clustered_high_household",
)
DEFAULT_COUNTRIES = (
    "Australia",
    "China",
    "Japan",
    "South_Africa",
    "United_States",
)
STRUCTURAL_CAVEAT = (
    "Structural sensitivity illustration only: stochastic household/contact-setting clustering "
    "is not calibrated to surveillance and does not replace the deterministic main model."
)


@dataclass(frozen=True)
class CountryInputs:
    country: str
    iso3: str
    population: np.ndarray
    contact_matrix: np.ndarray
    setting_matrices: dict[str, np.ndarray]
    setting_matrix_available: bool
    setting_decomposition: str
    contact_source: str


@dataclass
class SyntheticPopulation:
    age_index: np.ndarray
    household_id: np.ndarray
    household_members: list[np.ndarray]


def _age_kind(label: str) -> str:
    if label in INFANT_GROUPS:
        return "infant"
    if label == "child_1_4y":
        return "preschool"
    if label in {"child_5_9y", "adolescent_10_17y"}:
        return "school"
    if label in {"young_adult_18_39y", "middle_adult_40_64y"}:
        return "adult"
    return "elderly"


def _synthetic_setting_shares(source_age: str, target_age: str) -> dict[str, float]:
    """Fallback split used when location-specific contact matrices are unavailable."""
    source_kind = _age_kind(source_age)
    target_kind = _age_kind(target_age)

    home = 0.16
    school = 0.0
    work = 0.0
    if "infant" in {source_kind, target_kind}:
        home = 0.72
    elif {source_kind, target_kind} & {"preschool"} and {source_kind, target_kind} & {"adult", "school"}:
        home = 0.44
    elif source_kind == target_kind == "preschool":
        home = 0.34
    elif "elderly" in {source_kind, target_kind} and "adult" in {source_kind, target_kind}:
        home = 0.32
    elif source_kind == target_kind == "adult":
        home = 0.22

    if source_kind == target_kind == "school":
        school = 0.58
    elif {source_kind, target_kind} <= {"preschool", "school"}:
        school = 0.30

    if source_kind == target_kind == "adult":
        work = 0.34
    elif {source_kind, target_kind} == {"adult", "elderly"}:
        work = 0.08

    total_structured = home + school + work
    if total_structured > 0.92:
        scale = 0.92 / total_structured
        home *= scale
        school *= scale
        work *= scale
    other = max(0.0, 1.0 - home - school - work)
    shares = {"home": home, "school": school, "work": work, "other": other}
    total = sum(shares.values())
    return {key: float(value / total) for key, value in shares.items()}


def _setting_share_array() -> np.ndarray:
    shares = np.zeros((len(AGE_GROUPS), len(AGE_GROUPS), len(SETTINGS)), dtype=float)
    for i, source_age in enumerate(AGE_GROUPS):
        for j, target_age in enumerate(AGE_GROUPS):
            split = _synthetic_setting_shares(source_age, target_age)
            shares[i, j, :] = [split[setting] for setting in SETTINGS]
    return shares


def _relative_susceptibility() -> np.ndarray:
    return np.array([1.45, 1.25, 1.05, 0.95, 0.90, 0.72, 0.68, 0.75], dtype=float)


def _scenario_setting_weights(scenario: str) -> np.ndarray:
    if scenario == "setting_clustered_high_household":
        return np.array([1.85, 0.70, 0.55, 0.38], dtype=float)
    if scenario == "setting_clustered":
        return np.array([1.35, 0.85, 0.70, 0.50], dtype=float)
    return np.ones(len(SETTINGS), dtype=float)


def _fallback_setting_matrices(contact_matrix: np.ndarray) -> dict[str, np.ndarray]:
    setting_shares = _setting_share_array()
    return {
        setting: contact_matrix * setting_shares[:, :, setting_idx]
        for setting_idx, setting in enumerate(SETTINGS)
    }


def _matrix_from_long(df: pd.DataFrame, *, country: str, location: str | None = None) -> np.ndarray | None:
    rows = df.loc[df["country"].eq(country)].copy()
    if location is not None and "location" in rows.columns:
        rows = rows.loc[rows["location"].eq(location)].copy()
    if rows.empty:
        return None
    matrix = (
        rows.pivot_table(
            index="source_age_group",
            columns="target_age_group",
            values="contacts_per_day",
            aggfunc="mean",
        )
        .reindex(index=AGE_GROUPS, columns=AGE_GROUPS)
    )
    if matrix.isna().any().any() or matrix.shape != (len(AGE_GROUPS), len(AGE_GROUPS)):
        return None
    return matrix.to_numpy(dtype=float)


def _load_setting_matrices(country: str, contact_matrix: np.ndarray) -> tuple[dict[str, np.ndarray], bool, str]:
    setting_path = project_path("data", "processed", "country_contact_matrices_by_location_8groups.csv")
    if setting_path.exists():
        setting_df = pd.read_csv(setting_path)
        matrices: dict[str, np.ndarray] = {}
        for setting in SETTINGS:
            matrix = _matrix_from_long(setting_df, country=country, location=setting)
            if matrix is not None:
                matrices[setting] = matrix
        if set(matrices) == set(SETTINGS):
            return matrices, True, "contactdata_location_specific_8group_matrices"
    return _fallback_setting_matrices(contact_matrix), False, "synthetic_from_all_setting_contact_matrix"


def _load_country_inputs(selected_countries: tuple[str, ...]) -> list[CountryInputs]:
    configs = load_configs()
    profiles = configs["countries"]
    countries: list[CountryInputs] = []
    for country in selected_countries:
        if country not in profiles:
            raise KeyError(f"Country profile not found: {country}")
        profile = profiles[country]
        population = np.array([float(profile["population"][age]) for age in AGE_GROUPS], dtype=float)
        matrix = np.asarray(profile["contact_matrix"], dtype=float)
        if matrix.shape != (len(AGE_GROUPS), len(AGE_GROUPS)):
            raise ValueError(f"{country} contact matrix has shape {matrix.shape}, expected 8x8.")
        setting_matrices, setting_available, setting_decomposition = _load_setting_matrices(country, matrix)
        countries.append(
            CountryInputs(
                country=country,
                iso3=str(profile.get("iso3", "")),
                population=population,
                contact_matrix=matrix,
                setting_matrices=setting_matrices,
                setting_matrix_available=setting_available,
                setting_decomposition=setting_decomposition,
                contact_source=str(profile.get("contact_source", "Prem/contactdata all-setting matrix")),
            )
        )
    return countries


def _integer_age_counts(population: np.ndarray, population_size: int) -> np.ndarray:
    weights = population / population.sum()
    raw = weights * int(population_size)
    counts = np.floor(raw).astype(int)
    remainder = int(population_size - counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    counts[:2] = np.maximum(counts[:2], 2)
    counts[-1] = max(counts[-1], 1)
    delta = int(counts.sum() - population_size)
    if delta > 0:
        for idx in np.argsort(counts)[::-1]:
            removable = max(0, counts[idx] - (2 if idx < 2 else 1))
            take = min(delta, removable)
            counts[idx] -= take
            delta -= take
            if delta == 0:
                break
    return counts


def _take_one(available: list[list[int]], choices: tuple[int, ...], rng: np.random.Generator) -> int | None:
    nonempty = [idx for idx in choices if available[idx]]
    if not nonempty:
        return None
    group = int(rng.choice(nonempty))
    return available[group].pop()


def _build_population(country: CountryInputs, population_size: int, rng: np.random.Generator) -> SyntheticPopulation:
    counts = _integer_age_counts(country.population, population_size)
    age_index = np.concatenate([np.repeat(i, count) for i, count in enumerate(counts)]).astype(int)
    rng.shuffle(age_index)
    available = [[] for _ in AGE_GROUPS]
    for person_id, age in enumerate(age_index):
        available[int(age)].append(int(person_id))
    for ids in available:
        rng.shuffle(ids)

    households: list[list[int]] = []
    infant_ids = []
    for infant_group in (0, 1):
        while available[infant_group]:
            infant_ids.append(available[infant_group].pop())
    rng.shuffle(infant_ids)

    for infant_id in infant_ids:
        members = [infant_id]
        adult = _take_one(available, (5, 6), rng)
        if adult is not None:
            members.append(adult)
        if rng.random() < 0.55:
            second_adult = _take_one(available, (5, 6), rng)
            if second_adult is not None:
                members.append(second_adult)
        if rng.random() < 0.45:
            sibling = _take_one(available, (2, 3, 4), rng)
            if sibling is not None:
                members.append(sibling)
        households.append(members)

    remaining = [person_id for ids in available for person_id in ids]
    rng.shuffle(remaining)
    cursor = 0
    while cursor < len(remaining):
        size = int(rng.choice([1, 2, 3, 4, 5], p=[0.24, 0.33, 0.20, 0.15, 0.08]))
        households.append(remaining[cursor : cursor + size])
        cursor += size

    household_id = np.empty(len(age_index), dtype=int)
    household_members: list[np.ndarray] = []
    for hh_id, members in enumerate(households):
        member_array = np.asarray(members, dtype=int)
        household_members.append(member_array)
        household_id[member_array] = hh_id
    return SyntheticPopulation(age_index=age_index, household_id=household_id, household_members=household_members)


def _negative_binomial(mean: float, dispersion_k: float, rng: np.random.Generator) -> int:
    if mean <= 0.0:
        return 0
    if not np.isfinite(dispersion_k) or dispersion_k <= 0.0:
        return int(rng.poisson(mean))
    p = dispersion_k / (dispersion_k + mean)
    return int(rng.negative_binomial(dispersion_k, p))


def _global_target(
    source_age: int,
    force: np.ndarray,
    infected: np.ndarray,
    age_index: np.ndarray,
    rng: np.random.Generator,
) -> int | None:
    available_by_age = np.bincount(age_index[~infected], minlength=len(AGE_GROUPS)).astype(float)
    weights = force[source_age, :] * available_by_age
    total = weights.sum()
    if total <= 0.0:
        return None
    target_age = int(rng.choice(len(AGE_GROUPS), p=weights / total))
    candidates = np.flatnonzero((age_index == target_age) & (~infected))
    if candidates.size == 0:
        return None
    return int(rng.choice(candidates))


def _home_target(
    source_id: int,
    source_age: int,
    force: np.ndarray,
    population: SyntheticPopulation,
    infected: np.ndarray,
    rng: np.random.Generator,
) -> int | None:
    members = population.household_members[int(population.household_id[source_id])]
    candidates = members[(members != source_id) & (~infected[members])]
    if candidates.size == 0:
        return None
    weights = force[source_age, population.age_index[candidates]]
    total = weights.sum()
    if total <= 0.0:
        return None
    return int(rng.choice(candidates, p=weights / total))


def _source_scale(
    country: CountryInputs,
    scenario: str,
    target_reproduction: float,
) -> float:
    susceptibility = _relative_susceptibility()
    source_weights = country.population / country.population.sum()
    if scenario == "homogeneous_all_contacts":
        expected = (country.contact_matrix * susceptibility[None, :]).sum(axis=1)
    else:
        setting_weights = _scenario_setting_weights(scenario)
        setting_stack = np.stack([country.setting_matrices[setting] for setting in SETTINGS], axis=2)
        force = setting_stack * setting_weights[None, None, :]
        expected = (force.sum(axis=2) * susceptibility[None, :]).sum(axis=1)
    average = float(np.dot(source_weights, expected))
    return float(target_reproduction / max(average, 1e-12))


def _choose_index_case(
    population: SyntheticPopulation,
    rng: np.random.Generator,
    *,
    infant_household_index_probability: float,
) -> int:
    if rng.random() < infant_household_index_probability:
        candidates = []
        for members in population.household_members:
            ages = population.age_index[members]
            if np.isin(ages, [0, 1]).any():
                candidates.extend(members[~np.isin(ages, [0, 1])].tolist())
        if candidates:
            return int(rng.choice(candidates))

    index_weights = np.array([0.0, 0.0, 0.12, 0.22, 0.24, 0.25, 0.13, 0.04], dtype=float)
    person_weights = index_weights[population.age_index]
    person_weights = person_weights / person_weights.sum()
    return int(rng.choice(len(population.age_index), p=person_weights))


def _simulate_replicate(
    country: CountryInputs,
    scenario: str,
    replicate: int,
    population_size: int,
    target_reproduction: float,
    max_generations: int,
    seed: int,
    infant_household_index_probability: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    synthetic_population = _build_population(country, population_size, rng)
    infected = np.zeros(population_size, dtype=bool)
    generation = np.full(population_size, -1, dtype=int)
    infection_setting = np.full(population_size, "none", dtype=object)

    index_id = _choose_index_case(
        synthetic_population,
        rng,
        infant_household_index_probability=infant_household_index_probability,
    )
    infected[index_id] = True
    generation[index_id] = 0
    infection_setting[index_id] = "index"
    active = [index_id]
    scale = _source_scale(country, scenario, target_reproduction)
    susceptibility = _relative_susceptibility()

    if scenario == "homogeneous_all_contacts":
        force_by_setting = None
        homogeneous_force = scale * country.contact_matrix * susceptibility[None, :]
        dispersion = 0.70
    else:
        setting_weights = _scenario_setting_weights(scenario)
        setting_stack = np.stack([country.setting_matrices[setting] for setting in SETTINGS], axis=2)
        force_by_setting = (
            scale
            * setting_stack
            * setting_weights[None, None, :]
            * susceptibility[None, :, None]
        )
        homogeneous_force = None
        dispersion = 0.45

    for gen in range(1, max_generations + 1):
        next_active: list[int] = []
        for source_id in active:
            source_age = int(synthetic_population.age_index[source_id])
            if scenario == "homogeneous_all_contacts":
                assert homogeneous_force is not None
                mean = float(homogeneous_force[source_age, :].sum())
                for _ in range(_negative_binomial(mean, dispersion, rng)):
                    target_id = _global_target(
                        source_age,
                        homogeneous_force,
                        infected,
                        synthetic_population.age_index,
                        rng,
                    )
                    if target_id is None or infected[target_id]:
                        continue
                    infected[target_id] = True
                    generation[target_id] = gen
                    infection_setting[target_id] = "all_contacts"
                    next_active.append(target_id)
            else:
                assert force_by_setting is not None
                for setting_idx, setting in enumerate(SETTINGS):
                    force = force_by_setting[:, :, setting_idx]
                    mean = float(force[source_age, :].sum())
                    for _ in range(_negative_binomial(mean, dispersion, rng)):
                        if setting == "home":
                            target_id = _home_target(source_id, source_age, force, synthetic_population, infected, rng)
                        else:
                            target_id = _global_target(
                                source_age,
                                force,
                                infected,
                                synthetic_population.age_index,
                                rng,
                            )
                        if target_id is None or infected[target_id]:
                            continue
                        infected[target_id] = True
                        generation[target_id] = gen
                        infection_setting[target_id] = setting
                        next_active.append(target_id)
        active = next_active
        if not active or infected.sum() >= int(population_size * 0.35):
            break

    infected_ages = synthetic_population.age_index[infected]
    infant_mask = np.isin(infected_ages, [0, 1])
    infant_settings = infection_setting[infected][infant_mask]
    total_infections = int(infected.sum())
    infant_infections = int(infant_mask.sum())
    first_infant_generation = generation[infected][infant_mask]
    household_clusters = np.unique(synthetic_population.household_id[infected]).size
    return {
        "analysis": "individual_stochastic_structural_sensitivity",
        "country": country.country,
        "iso3": country.iso3,
        "scenario": scenario,
        "replicate": int(replicate),
        "seed": int(seed),
        "population_size": int(population_size),
        "target_reproduction_number": float(target_reproduction),
        "infant_household_index_probability": float(infant_household_index_probability),
        "setting_matrix_available": bool(country.setting_matrix_available),
        "setting_decomposition": country.setting_decomposition,
        "index_age_group": AGE_GROUPS[int(synthetic_population.age_index[index_id])],
        "total_infections": total_infections,
        "outbreak_20plus": bool(total_infections >= 20),
        "extinction_3_or_fewer": bool(total_infections <= 3),
        "infant_infections": infant_infections,
        "any_infant_infection": bool(infant_infections > 0),
        "infant_home_infections": int(np.isin(infant_settings, ["home", "index"]).sum()),
        "infant_nonhome_infections": int(np.isin(infant_settings, ["school", "work", "other", "all_contacts"]).sum()),
        "generation_first_infant": int(first_infant_generation.min()) if first_infant_generation.size else -1,
        "max_generation_reached": int(generation[infected].max()),
        "household_clusters_touched": int(household_clusters),
        "structural_sensitivity_caveat": STRUCTURAL_CAVEAT,
    }


def _summarize_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, iso3, scenario), group in replicates.groupby(["country", "iso3", "scenario"], sort=False):
        total = group["total_infections"].astype(float)
        infant = group["infant_infections"].astype(float)
        rows.append(
            {
                "analysis": "individual_stochastic_structural_sensitivity",
                "country": country,
                "iso3": iso3,
                "scenario": scenario,
                "n_replicates": int(len(group)),
                "population_size": int(group["population_size"].iloc[0]),
                "target_reproduction_number": float(group["target_reproduction_number"].iloc[0]),
                "infant_household_index_probability": float(group["infant_household_index_probability"].iloc[0]),
                "setting_matrix_available": bool(group["setting_matrix_available"].iloc[0]),
                "setting_decomposition": str(group["setting_decomposition"].iloc[0]),
                "extinction_probability_3_or_fewer": float(group["extinction_3_or_fewer"].mean()),
                "outbreak_probability_20plus": float(group["outbreak_20plus"].mean()),
                "median_total_infections": float(total.median()),
                "q025_total_infections": float(total.quantile(0.025)),
                "q975_total_infections": float(total.quantile(0.975)),
                "mean_infant_infections": float(infant.mean()),
                "probability_any_infant_infection": float(group["any_infant_infection"].mean()),
                "q95_infant_infections": float(infant.quantile(0.95)),
                "mean_household_clusters_touched": float(group["household_clusters_touched"].mean()),
                "interpretation": (
                    "Distribution across stochastic introductions in a small synthetic population; "
                    "compare scenarios by extinction, outbreak tail, and infant-infection probabilities."
                ),
                "structural_sensitivity_caveat": STRUCTURAL_CAVEAT,
            }
        )
    return pd.DataFrame(rows)


def _contact_audit(country_inputs: list[CountryInputs]) -> pd.DataFrame:
    raw_path = project_path("data/raw/external/contactdata_prem_contact_matrices_16.csv")
    raw_setting_path = project_path("data/raw/external/contactdata_prem_contact_matrices_by_location_16.csv")
    processed_path = project_path("data/processed/country_contact_matrices_8groups.csv")
    processed_setting_path = project_path("data/processed/country_contact_matrices_by_location_8groups.csv")
    raw_columns = pd.read_csv(raw_path, nrows=0).columns.tolist() if raw_path.exists() else []
    raw_setting_columns = pd.read_csv(raw_setting_path, nrows=0).columns.tolist() if raw_setting_path.exists() else []
    processed_columns = pd.read_csv(processed_path, nrows=0).columns.tolist() if processed_path.exists() else []
    processed_setting_columns = (
        pd.read_csv(processed_setting_path, nrows=0).columns.tolist() if processed_setting_path.exists() else []
    )

    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    raw_setting = pd.read_csv(raw_setting_path) if raw_setting_path.exists() else pd.DataFrame()
    processed = pd.read_csv(processed_path) if processed_path.exists() else pd.DataFrame()
    processed_setting = pd.read_csv(processed_setting_path) if processed_setting_path.exists() else pd.DataFrame()
    rows = []
    for country in country_inputs:
        raw_country = raw_setting.loc[raw_setting["country"].eq(country.country)] if not raw_setting.empty else pd.DataFrame()
        processed_country = (
            processed.loc[processed["country"].eq(country.country)] if not processed.empty else pd.DataFrame()
        )
        processed_setting_country = (
            processed_setting.loc[processed_setting["country"].eq(country.country)]
            if not processed_setting.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "country": country.country,
                "iso3": country.iso3,
                "raw_contact_file": raw_setting_path.as_posix(),
                "raw_rows_for_country": int(len(raw_country)),
                "raw_contact_columns": ";".join(raw_setting_columns),
                "raw_setting_matrix_available": "location" in raw_setting_columns,
                "raw_setting_columns": "location" if "location" in raw_setting_columns else "none",
                "processed_contact_file": processed_path.as_posix(),
                "processed_rows_for_country": int(len(processed_country)),
                "processed_contact_columns": ";".join(processed_columns),
                "processed_setting_contact_file": processed_setting_path.as_posix(),
                "processed_setting_rows_for_country": int(len(processed_setting_country)),
                "processed_setting_matrix_available": bool(country.setting_matrix_available),
                "processed_setting_columns": ";".join(processed_setting_columns) if processed_setting_columns else "none",
                "country_profile_contact_groups": int(country.contact_matrix.shape[0]),
                "contact_source": country.contact_source,
                "toy_model_setting_use": (
                    "contactdata home/school/work/other matrices aggregated to 8 model age groups"
                    if country.setting_matrix_available
                    else "fallback split of all-setting contacts into home/school/work/other"
                ),
                "structural_sensitivity_caveat": STRUCTURAL_CAVEAT,
            }
        )
    return pd.DataFrame(rows)


def _setting_decomposition_table(country_inputs: list[CountryInputs]) -> pd.DataFrame:
    rows = []
    for country in country_inputs:
        for i, source_age in enumerate(AGE_GROUPS):
            for j, target_age in enumerate(AGE_GROUPS):
                all_contacts = float(country.contact_matrix[i, j])
                row = {
                    "country": country.country,
                    "iso3": country.iso3,
                    "source_age_group": source_age,
                    "target_age_group": target_age,
                    "all_setting_contacts_per_day": all_contacts,
                    "setting_matrix_available": bool(country.setting_matrix_available),
                    "setting_decomposition": country.setting_decomposition,
                    "structural_sensitivity_caveat": STRUCTURAL_CAVEAT,
                }
                for setting in SETTINGS:
                    setting_contacts = float(country.setting_matrices[setting][i, j])
                    row[f"{setting}_contacts_per_day"] = setting_contacts
                    row[f"{setting}_share"] = setting_contacts / all_contacts if all_contacts > 0 else 0.0
                rows.append(row)
    return pd.DataFrame(rows)


def build_tables(
    *,
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    replicates: int = 100,
    population_size: int = 1_500,
    target_reproduction: float = 1.08,
    max_generations: int = 8,
    infant_household_index_probability: float = 0.30,
    seed: int = 20260521,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    country_inputs = _load_country_inputs(countries)
    rows = []
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(country_inputs) * len(SCENARIOS) * replicates)
    seed_iter = iter(child_seeds)
    for country in country_inputs:
        for scenario in SCENARIOS:
            for replicate in range(1, replicates + 1):
                child_seed = int(next(seed_iter).generate_state(1)[0])
                rows.append(
                    _simulate_replicate(
                        country,
                        scenario,
                        replicate,
                        population_size,
                        target_reproduction,
                        max_generations,
                        child_seed,
                        infant_household_index_probability,
                    )
                )
    replicate_table = pd.DataFrame(rows)
    return (
        replicate_table,
        _summarize_replicates(replicate_table),
        _contact_audit(country_inputs),
        _setting_decomposition_table(country_inputs),
    )


def main(
    *,
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    replicates: int = 100,
    population_size: int = 1_500,
    target_reproduction: float = 1.08,
    max_generations: int = 8,
    infant_household_index_probability: float = 0.30,
    seed: int = 20260521,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replicate_table, summary, contact_audit, setting_decomposition = build_tables(
        countries=countries,
        replicates=replicates,
        population_size=population_size,
        target_reproduction=target_reproduction,
        max_generations=max_generations,
        infant_household_index_probability=infant_household_index_probability,
        seed=seed,
    )
    write_dataframe(
        replicate_table,
        project_path("outputs", "tables", "individual_stochastic_toy_replicates.csv"),
    )
    write_dataframe(
        summary,
        project_path("outputs", "tables", "individual_stochastic_toy_summary.csv"),
    )
    write_dataframe(
        contact_audit,
        project_path("outputs", "tables", "individual_stochastic_toy_contact_audit.csv"),
    )
    write_dataframe(
        setting_decomposition,
        project_path("outputs", "tables", "individual_stochastic_toy_setting_decomposition.csv"),
    )
    write_run_metadata(
        "individual_stochastic_toy",
        current_run_metadata(
            "individual_stochastic_toy",
            row_counts={
                "replicates": int(len(replicate_table)),
                "summary": int(len(summary)),
                "contact_audit": int(len(contact_audit)),
                "setting_decomposition": int(len(setting_decomposition)),
            },
        ),
    )
    return replicate_table, summary, contact_audit, setting_decomposition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a small individual stochastic structural-sensitivity toy model "
            "for contact-setting clustering and infant outcomes."
        )
    )
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=1_500)
    parser.add_argument("--target-reproduction", type=float, default=1.08)
    parser.add_argument("--max-generations", type=int, default=8)
    parser.add_argument("--infant-household-index-probability", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()
    main(
        countries=tuple(args.countries),
        replicates=args.replicates,
        population_size=args.population_size,
        target_reproduction=args.target_reproduction,
        max_generations=args.max_generations,
        infant_household_index_probability=args.infant_household_index_probability,
        seed=args.seed,
    )
