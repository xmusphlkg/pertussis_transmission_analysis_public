from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_python.simulation.common import current_run_metadata, write_run_metadata

TABLE_DIR = ROOT / "outputs" / "tables"


ROWS = [
    {
        "candidate_or_platform": "BPZE1 live-attenuated intranasal vaccine",
        "route_or_platform": "Intranasal live attenuated Bordetella pertussis",
        "development_status_as_of_2026_05_21": (
            "Phase 2b adult immunogenicity/safety and controlled human infection evidence; "
            "school-age phase 2b study registered."
        ),
        "mechanistic_relevance": (
            "Designed to induce nasal mucosal immunity and reduce colonization, making it the "
            "closest clinical candidate to the model's high-transmission-blocking product target."
        ),
        "model_representation": (
            "Represented by the existing hypothetical next_generation upper-bound profile and by "
            "VE_inf sensitivity/PSA ranges; not assigned product-specific efficacy."
        ),
        "reason_not_modeled_as_available_policy": (
            "No licensed product or population effectiveness estimate was available at analysis lock; "
            "challenge and immunogenicity endpoints do not directly identify country-level VE_sus, VE_inf, or VE_dur."
        ),
        "evidence_source": (
            "Keech et al [42]; "
            "Gbesemete et al [43]; "
            "ClinicalTrials.gov NCT03942406, NCT05461131, NCT05116241."
        ),
    },
    {
        "candidate_or_platform": "Outer-membrane-vesicle or OMV-adjuvanted pertussis platforms",
        "route_or_platform": "OMV-based or OMV-adjuvanted formulations; mostly preclinical",
        "development_status_as_of_2026_05_21": (
            "Recent preclinical and translational studies; no late-stage pertussis efficacy trial identified."
        ),
        "mechanistic_relevance": (
            "May broaden antigenic coverage and enhance Th1/Th17 or tissue-resident responses, "
            "potentially affecting susceptibility, infectiousness, or colonization duration."
        ),
        "model_representation": (
            "Covered by infection_blocking, transmission_blocking, next_generation, and VE_inf/VE_dur sensitivity ranges."
        ),
        "reason_not_modeled_as_available_policy": (
            "Mechanistic animal or immunology evidence cannot be translated into a named product policy scenario."
        ),
        "evidence_source": (
            "Locati et al [44]; "
            "related OMV literature cited therein."
        ),
    },
    {
        "candidate_or_platform": "Genetically detoxified recombinant pertussis-toxin acellular vaccines",
        "route_or_platform": "Injectable recombinant acellular booster formulations",
        "development_status_as_of_2026_05_21": (
            "BioNet reports licensed recombinant pertussis booster products in Asia; Pertagen2x phase II/III study registered."
        ),
        "mechanistic_relevance": (
            "Potentially stronger or more durable antibody responses than chemically detoxified aP boosters, "
            "but not primarily a mucosal-transmission-blocking platform."
        ),
        "model_representation": (
            "Most consistent with adolescent_booster, symptom_protective, infection_blocking, or waning-duration sensitivity scenarios."
        ),
        "reason_not_modeled_as_available_policy": (
            "Product availability and schedules vary by jurisdiction, and transmission-blocking parameters were not identifiable."
        ),
        "evidence_source": "BioNet pertussis product information; ClinicalTrials.gov NCT05193734.",
    },
    {
        "candidate_or_platform": "New multi-component acellular pertussis combination vaccines",
        "route_or_platform": "Injectable DTaP/Tdap-like acellular combinations",
        "development_status_as_of_2026_05_21": (
            "CanSino DTcP phase 3 active-not-recruiting trial identified; other multi-component formulations remain product-specific."
        ),
        "mechanistic_relevance": (
            "Relevant to clinical protection and possibly infection blocking, but less direct evidence for mucosal carriage reduction."
        ),
        "model_representation": (
            "Covered by symptom_protective and infection_blocking vaccine-mechanism profiles rather than a separate named scenario."
        ),
        "reason_not_modeled_as_available_policy": (
            "No published population-level transmission-blocking estimate was available for parameterizing VE_inf or VE_dur."
        ),
        "evidence_source": "ClinicalTrials.gov NCT05951725.",
    },
]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(ROWS)
    out = TABLE_DIR / "vaccine_pipeline_mechanism_mapping.csv"
    table.to_csv(out, index=False)
    try:
        table.to_parquet(out.with_suffix(".parquet"), index=False)
    except Exception:
        pass
    metadata = current_run_metadata(
        "vaccine_pipeline_mapping",
        row_counts={out.relative_to(ROOT).as_posix(): int(len(table))},
    )
    metadata.update(
        {
            "analysis_lock_date": "2026-05-21",
            "source_note": "Mechanism mapping table assembled from cited vaccine pipeline evidence.",
        }
    )
    write_run_metadata("vaccine_pipeline_mapping", metadata)
    print(f"Wrote {out.relative_to(ROOT)} with {len(table)} rows.")


if __name__ == "__main__":
    main()
