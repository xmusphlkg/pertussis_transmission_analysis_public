# Pertussis Transmission Model - Public Reproducibility Pipeline
#
# This public Makefile covers data processing, calibration, simulations,
# retained source-data generation, figure rendering, and source-data validation.

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RSCRIPT ?= Rscript
N_JOBS ?= 40

STAMP_DIR := .pipeline_stamps
$(shell mkdir -p $(STAMP_DIR))

PYTHON_INIT := src_python/__init__.py
DATA_PY := $(shell find src_python/data src_python/utils -type f -name '*.py')
DATA_MODEL_PY := src_python/model/contact_matrix.py
CALIBRATION_PY := $(shell find src_python/calibration src_python/model src_python/utils -type f -name '*.py')
SIMULATION_PY := $(shell find src_python/simulation src_python/model src_python/utils -type f -name '*.py')
SIMULATION_COMMON_PY := src_python/simulation/common.py
RUNTIME_DATA_INPUTS := data/raw/country_resistance_timeline.csv \
                       data/raw/pertussis_diagnostic_standards_timeline.csv \
                       data/raw/covid_npi_contact_reduction_timeline.csv
PUBLICATION_NOTE_PY := publication_inputs/generate_health_utility_analysis.py \
                       publication_inputs/generate_vaccine_pipeline_mapping.py \
                       publication_inputs/generate_high_risk_review_tables.py \
                       publication_inputs/generate_lancet_child_adolescent_tables.py
ACTIVE_R_FIGURE_SOURCES := scripts_R/_helpers.R \
                           scripts_R/00_setup.R \
                           scripts_R/10_shared.R \
                           scripts_R/10_figure_set.R \
                           scripts_R/10_extended_data.R \
                           scripts_R/11_figure_1_baseline_heterogeneity.R \
                           scripts_R/12_figure_2_country_strategy_prioritization.R \
                           scripts_R/13_figure_3_endpoint_dependence.R \
                           scripts_R/14_figure_4_resistance_conditioned_control.R \
                           scripts_R/15_figure_5_future_vaccine_mechanisms.R \
                           scripts_R/15_extended_data_1_country_inputs.R \
                           scripts_R/16_extended_data_2_diagnostics_sensitivity.R \
                           scripts_R/20_extended_data_6_baseline_dynamics.R \
                           scripts_R/21_extended_data_7_vaccine_deep_dive.R \
                           scripts_R/22_extended_data_8_resistance_dynamics.R \
                           scripts_R/23_extended_data_9_full_grid.R \
                           scripts_R/24_extended_data_10_intervention_extended.R \
                           scripts_R/25_extended_data_11_model_structure.R \
                           scripts_R/26_extended_data_13_resistance_hindcast.R \
                           scripts_R/27_extended_data_9_scenario_robustness.R \
                           scripts_R/28_extended_data_10_resistance_management_policy.R \
                           scripts_R/29_extended_data_11_implementation_structural_robustness.R

.PHONY: all data calibrate simulate bayesian hindcast publication-data figures \
        validate-public test clean-stamps help FORCE

FORCE:

help:
	@echo "Pertussis Transmission Model Public Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  all               Generate figures and run public validation"
	@echo "  data              Process raw data into model inputs"
	@echo "  calibrate         Run country-level calibration"
	@echo "  simulate          Run core scenario simulations"
	@echo "  bayesian          Run Bayesian beta-grid uncertainty analysis"
	@echo "  hindcast          Run resistance hindcast validation"
	@echo "  publication-data  Generate retained source-data tables"
	@echo "  figures           Generate R figures"
	@echo "  validate-public   Validate model/source-data outputs without private submission files"
	@echo "  test              Run pytest suite"
	@echo "  clean-stamps      Remove pipeline stamps"
	@echo ""
	@echo "Options:"
	@echo "  N_JOBS=40         Number of parallel workers"
	@echo "  PYTHON=python3    Python interpreter"
	@echo "  RSCRIPT=Rscript   R interpreter"

all: figures validate-public

$(STAMP_DIR)/data: config/model_settings.yaml \
                   data/raw/external/*.xlsx \
                   data/raw/external/*.csv \
                   $(PYTHON_INIT) $(DATA_PY) $(DATA_MODEL_PY)
	$(PYTHON) -m src_python.data.build_who_inputs
	$(PYTHON) -m src_python.data.build_country_inputs
	@mkdir -p $(STAMP_DIR)
	@touch $@

data: $(STAMP_DIR)/data

$(STAMP_DIR)/calibrate: $(STAMP_DIR)/data config/model_settings.yaml $(RUNTIME_DATA_INPUTS) \
                         $(PYTHON_INIT) $(CALIBRATION_PY) $(SIMULATION_COMMON_PY)
	$(PYTHON) -m src_python.calibration.run_all --n-jobs $(N_JOBS)
	@mkdir -p $(STAMP_DIR)
	@touch $@

calibrate: $(STAMP_DIR)/calibrate

$(STAMP_DIR)/simulate: $(STAMP_DIR)/calibrate config/model_settings.yaml $(RUNTIME_DATA_INPUTS) \
                       $(PYTHON_INIT) $(SIMULATION_PY)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_baseline
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_vaccine_scenarios
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_resistance_scenarios
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_reporting_scenarios
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_country_scenarios
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_heatmap_grid
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_fitness_grid --posterior-draws 0
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_intervention_scenarios
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_routine_timeliness_sensitivity
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_sensitivity
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_immunity_sensitivity
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_resistance_fitness_sensitivity
	@mkdir -p $(STAMP_DIR)
	@touch $@

simulate: $(STAMP_DIR)/simulate

$(STAMP_DIR)/bayesian: $(STAMP_DIR)/calibrate config/model_settings.yaml $(RUNTIME_DATA_INPUTS) \
                       $(PYTHON_INIT) $(SIMULATION_PY)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_bayesian_uncertainty --n-jobs $(N_JOBS) --solver-mode calibration --sampler beta_grid --proposal-scale 1.0 --warmup 0 --draws 250 --fix-durations --fix-parameters reporting_multiplier,VE_sus,VE_inf,relative_infectiousness_asymptomatic,fitness_R --grid-points 81 --grid-log-beta-half-width 0.08 --grid-max-points 641 --grid-max-refinements 10 --grid-smoothing auto --grid-eval-jobs $(N_JOBS)
	@mkdir -p $(STAMP_DIR)
	@touch $@

bayesian: $(STAMP_DIR)/bayesian

$(STAMP_DIR)/hindcast: $(STAMP_DIR)/calibrate $(RUNTIME_DATA_INPUTS) $(PYTHON_INIT) $(SIMULATION_PY)
	$(PYTHON) -m src_python.simulation.run_resistance_hindcast
	@mkdir -p $(STAMP_DIR)
	@touch $@

hindcast: $(STAMP_DIR)/hindcast

$(STAMP_DIR)/publication-data: $(STAMP_DIR)/simulate $(STAMP_DIR)/bayesian $(STAMP_DIR)/hindcast \
                               config/model_settings.yaml \
                               outputs/tables/strategy_domain_interpretation.csv \
                               $(PYTHON_INIT) $(SIMULATION_PY) $(PUBLICATION_NOTE_PY)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_calibration_diagnostics
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_age_pattern_sensitivity
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_resistance_mechanism_decomposition --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_program_portfolio_factorial --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_infant_contact_sensitivity --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_maternal_duration_sensitivity --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_shock_recovery_sensitivity --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_temporal_assumption_sensitivity --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_treatment_implementation_sensitivity --n-jobs $(N_JOBS)
	$(PYTHON) -m src_python.simulation.run_individual_stochastic_toy
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_joint_psa_rank_acceptability --samples 128 --seed 20260521 --sample-batch-size 8 --n-jobs $(N_JOBS)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_fitness_grid --skip-deterministic-grid --posterior-draws 100 --posterior-batch-size 32 --psa-benefit-samples -1 --psa-batch-size 4 --n-jobs $(N_JOBS)
	$(PYTHON) -c "from src_python.simulation.common import write_manuscript_tables; write_manuscript_tables()"
	$(PYTHON) publication_inputs/generate_health_utility_analysis.py
	$(PYTHON) publication_inputs/generate_vaccine_pipeline_mapping.py
	$(PYTHON) publication_inputs/generate_high_risk_review_tables.py
	$(PYTHON) publication_inputs/generate_lancet_child_adolescent_tables.py
	@mkdir -p $(STAMP_DIR)
	@touch $@

publication-data: $(STAMP_DIR)/publication-data

$(STAMP_DIR)/figures: $(STAMP_DIR)/publication-data $(ACTIVE_R_FIGURE_SOURCES)
	$(RSCRIPT) scripts_R/00_setup.R
	$(RSCRIPT) scripts_R/10_figure_set.R
	@mkdir -p $(STAMP_DIR)
	@touch $@

figures: $(STAMP_DIR)/figures

validate-public:
	$(PYTHON) -c "from src_python.utils.validation import validate_population_conservation, validate_baseline_outputs, validate_main_output_windows, validate_publication_outputs; validate_population_conservation(); validate_baseline_outputs(); validate_main_output_windows(); validate_publication_outputs(); print('Public validation checks passed.')"

test:
	$(PYTHON) -m pytest tests/ -v

clean-stamps:
	rm -rf $(STAMP_DIR)
