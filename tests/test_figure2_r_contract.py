from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable")
def test_r_figure2_country_contract_fails_when_a_publication_country_is_missing() -> None:
    expression = """
    source('scripts_R/lib/bootstrap.R')
    source('scripts_R/figures/figure_2/data.R')
    old_python <- Sys.getenv('PERTUSSIS_PYTHON', unset = NA_character_)
    Sys.setenv(PERTUSSIS_PYTHON = '/bin/false')
    python_gate_failed <- tryCatch({
      require_current_figure2_python_gate()
      FALSE
    }, error = function(e) TRUE)
    stopifnot(python_gate_failed)
    if (is.na(old_python)) Sys.unsetenv('PERTUSSIS_PYTHON') else Sys.setenv(PERTUSSIS_PYTHON = old_python)
    accepted <- validate_figure_2_country_contract(
      c('A', 'B', 'C'), c('A', 'B', 'C', 'D'), 'D', c('A', 'B', 'C')
    )
    stopifnot(setequal(accepted, c('A', 'B', 'C')))
    failed <- tryCatch({
      validate_figure_2_country_contract(
        c('A', 'B'), c('A', 'B', 'C', 'D'), 'D', c('A', 'B', 'C')
      )
      FALSE
    }, error = function(e) TRUE)
    stopifnot(failed)
    stopifnot(identical(
      require_current_figure2_python_gate,
      require_current_figure_2_python_gate
    ))
    """

    subprocess.run(
        ["Rscript", "-e", expression],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
