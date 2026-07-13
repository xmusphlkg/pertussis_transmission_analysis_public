"""Audit that submission text reflects either a valid release or an explicit block."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src_python.simulation.common import (
    current_run_metadata,
    load_configs,
    publication_country_names,
    write_run_metadata,
)
from src_python.utils.io import project_path, write_dataframe
from src_python.validation.publication_gate import require_figure2_publication_gate


STEM = "submission_text_release_audit"
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FORBIDDEN_ACTIVE_PATTERNS = {
    "old_2025_2050_horizon": re.compile(
        r"(?:2025\s*(?:[–-]|to|through)\s*(?:50|2050)|"
        r"January\s+1,?\s*2025\s+(?:to|through)\s+December\s+31,?\s*2050)",
        re.I,
    ),
    "unsupported_2052_horizon": re.compile(r"\b2052\b", re.I),
    "old_ten_profile_scope": re.compile(
        r"(?:\bten[-\s]+(?:programme[-\s]+|country[-\s]+)?profiles\b|"
        r"\b10\s+(?:country\s+)?profiles\b|\b10\s+countries\b)",
        re.I,
    ),
    "affirmative_joint_cri": re.compile(r"\b(?:95%\s+)?joint[- ]?cri\b", re.I),
    "affirmative_full_joint_interval": re.compile(
        r"(?<!not )(?<!not a )\bfull[- ]joint(?:\s+Bayesian)?\s+credible[- ]interval",
        re.I,
    ),
}


def active_markdown(text: str) -> str:
    return HTML_COMMENT.sub("", text)


def _release_ready() -> bool:
    try:
        require_figure2_publication_gate()
    except (FileNotFoundError, RuntimeError, ValueError):
        return False
    return True


def audit_submission_texts(
    documents: dict[str, str],
    *,
    release_ready: bool,
    figure2_exists: bool,
    expected_start_date: str | None = None,
    expected_end_date: str | None = None,
    expected_publication_country_count: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(document: str, check: str, passed: bool, details: str) -> None:
        rows.append(
            {
                "document": document,
                "check": check,
                "status": "pass" if passed else "fail",
                "details": details,
            }
        )

    active = {name: active_markdown(text) for name, text in documents.items()}
    for name, text in active.items():
        for check, pattern in FORBIDDEN_ACTIVE_PATTERNS.items():
            matches = pattern.findall(text)
            add(name, check, not matches, f"active_matches={matches}")

    configs = load_configs()
    calendar = configs["baseline"]["calendar"]
    start = pd.Timestamp(
        expected_start_date or calendar["analysis_start_date"]
    )
    end = pd.Timestamp(expected_end_date or calendar["analysis_end_date"])
    country_count = int(
        expected_publication_country_count
        if expected_publication_country_count is not None
        else len(publication_country_names(configs))
    )
    number_words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve",
    }
    count_label = number_words.get(country_count, str(country_count))
    short_horizon = f"{start.year}–{end.year % 100:02d}"
    start_label = f"{start.strftime('%b')} {start.day}, {start.year}"
    end_label = f"{end.strftime('%b')} {end.day}, {end.year}"

    main = active.get("main_manuscript", "")
    add(
        "main_manuscript",
        "current_horizon_declared",
        short_horizon in main and start_label in main and end_label in main,
        f"requires {short_horizon} and endpoints {start_label}/{end_label}",
    )
    add(
        "main_manuscript",
        "nine_country_publication_scope_declared",
        f"{count_label} calibrated publication profiles" in main,
        f"requires exact {count_label}-profile publication wording",
    )
    release_documents = {
        "main_manuscript", "cover_letter", "conference_abstract", "supplement"
    }
    not_ready_markers = {
        name: "NOT SUBMISSION READY" in text or "results withheld" in text.lower()
        for name, text in active.items()
        if name in release_documents
    }
    if release_ready:
        for name, marked in not_ready_markers.items():
            add(name, "release_status_text", not marked, "release-ready text must remove draft block")
        add(
            "main_manuscript",
            "figure2_release_artifact_present",
            figure2_exists and "![Figure 2]" in main,
            f"figure2_exists={figure2_exists}",
        )
    else:
        for name, marked in not_ready_markers.items():
            add(name, "release_status_text", marked, "blocked release must be explicit")
        add(
            "main_manuscript",
            "failed_release_suppresses_figure2",
            not figure2_exists and "![Figure 2]" not in main,
            f"figure2_exists={figure2_exists}",
        )
    return pd.DataFrame(rows)


def main(*, require_release_ready: bool = False) -> pd.DataFrame:
    paths = {
        "main_manuscript": project_path("manuscript", "submission_ready", "main_manuscript.md"),
        "cover_letter": project_path("manuscript", "submission_ready", "cover_letter.md"),
        "conference_abstract": project_path(
            "manuscript", "submission_ready", "pediatrics_conference_abstract_english.md"
        ),
        "supplement": project_path(
            "manuscript", "submission_ready", "supplementary_material.md"
        ),
        "title_page": project_path(
            "manuscript", "submission_ready", "title_page_author_details.md"
        ),
        "supplementary_methods_template": project_path(
            "manuscript", "appendix_templates", "supplementary_methods.md"
        ),
        "supplementary_tables_template": project_path(
            "manuscript", "appendix_templates", "supplementary_tables.md"
        ),
        "supplementary_figures_template": project_path(
            "manuscript", "appendix_templates", "supplementary_figures.md"
        ),
    }
    missing = [str(path) for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Submission documents are missing: {missing}")
    release_ready = _release_ready()
    figure2_path = project_path(
        "outputs", "figures", "figure_2_country_strategy_prioritization.png"
    )
    audit = audit_submission_texts(
        {name: Path(path).read_text(encoding="utf-8") for name, path in paths.items()},
        release_ready=release_ready,
        figure2_exists=Path(figure2_path).exists(),
    )
    write_dataframe(
        audit,
        project_path("outputs", "tables", f"{STEM}.csv"),
    )
    passed = bool(audit["status"].eq("pass").all())
    write_run_metadata(
        STEM,
        current_run_metadata(
            STEM,
            row_counts={"checks": int(len(audit)), "failures": int((~audit["status"].eq("pass")).sum())},
        )
        | {"release_ready": release_ready, "passed": passed},
    )
    if not passed:
        failures = audit.loc[audit["status"].eq("fail"), ["document", "check"]]
        raise RuntimeError(
            "Submission text release audit failed: " + str(failures.to_dict("records"))
        )
    if require_release_ready and not release_ready:
        raise RuntimeError("Submission is internally consistent but release gates have not passed")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-release-ready", action="store_true")
    args = parser.parse_args()
    main(require_release_ready=bool(args.require_release_ready))
