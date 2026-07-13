# Public reproducibility pipeline for the pertussis transmission analysis.

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RSCRIPT ?= Rscript
N_JOBS ?= 64
STAMP_DIR := .pipeline_stamps
$(shell mkdir -p $(STAMP_DIR))

PYTHON_INIT := src_python/__init__.py
DATA_PY := $(shell find src_python/data src_python/utils -type f -name '*.py')
DATA_MODEL_PY := src_python/model/contact_matrix.py
CALIBRATION_PY := $(shell find src_python/calibration src_python/model src_python/utils -type f -name '*.py')
SIMULATION_PY := $(shell find src_python/simulation src_python/model src_python/utils -type f -name '*.py')
VALIDATION_PY := $(shell find src_python/validation -type f -name '*.py')
RUNTIME_DATA_INPUTS := data/raw/country_resistance_timeline.csv \
                       data/raw/pertussis_diagnostic_standards_timeline.csv \
                       data/raw/covid_npi_contact_reduction_timeline.csv

.PHONY: full data calibrate simulate calibration-hindcast \
        panel-pomp-block-stress hindcast publication-diagnostics joint-psa \
        fitness-grid-psa publication-tables publication-data figures \
        validate-public test clean-stamps help FORCE

FORCE:

help:
	@echo "Pertussis public reproducibility pipeline"
	@echo "  full                     Rebuild the active non-MCMC publication route"
	@echo "  data                     Process source data"
	@echo "  calibrate                Reconstruct country starting states"
	@echo "  simulate                 Run deterministic scenario simulations"
	@echo "  calibration-hindcast     Run leakage-safe prequential POMP/ensemble gate"
	@echo "  panel-pomp-block-stress  Run unassimilated annual block stress test"
	@echo "  publication-data         Generate retained diagnostics and source tables"
	@echo "  publication-diagnostics Rebuild non-PSA publication diagnostics"
	@echo "  joint-psa                Run/resume the 128-point paired joint PSA"
	@echo "  fitness-grid-psa         Project PSA samples onto the fitness grid"
	@echo "  publication-tables      Rebuild retained source tables"
	@echo "  figures                  Render main and extended figures"
	@echo "  validate-public          Validate retained output windows, hashes, and gates"
	@echo "  test                     Run the complete public pytest suite"
	@echo "  N_JOBS=64                Bounded process workers; use up to 100 if available"

full: figures
	$(MAKE) validate-public

$(STAMP_DIR)/data: config/model_settings.yaml data/raw/external/*.xlsx \
                   data/raw/external/*.csv $(PYTHON_INIT) $(DATA_PY) $(DATA_MODEL_PY)
	$(PYTHON) -m src_python.data.build_who_inputs
	$(PYTHON) -m src_python.data.build_country_inputs
	@touch $@

data: $(STAMP_DIR)/data

$(STAMP_DIR)/calibrate: $(STAMP_DIR)/data config/model_settings.yaml \
                         $(RUNTIME_DATA_INPUTS) $(PYTHON_INIT) $(CALIBRATION_PY)
	$(PYTHON) -m src_python.calibration.run_all --n-jobs $(N_JOBS)
	@touch $@

calibrate: $(STAMP_DIR)/calibrate

$(STAMP_DIR)/simulate: $(STAMP_DIR)/calibrate config/model_settings.yaml \
                       config/parameter_distributions.yaml $(RUNTIME_DATA_INPUTS) \
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
	@touch $@

simulate: $(STAMP_DIR)/simulate

$(STAMP_DIR)/calibration-hindcast: $(STAMP_DIR)/calibrate $(RUNTIME_DATA_INPUTS) \
                                 $(PYTHON_INIT) $(CALIBRATION_PY) $(SIMULATION_PY) $(VALIDATION_PY)
	@rm -f $@
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.validation.run_panel_pomp_hindcast --forecast-mode prequential --test-years 2023-2026 --particles 512 --predictive-draws 4096 --mc-replicates 3 --n-jobs $(N_JOBS) --require-predictive-gate
	@touch $@

calibration-hindcast: $(STAMP_DIR)/calibration-hindcast

$(STAMP_DIR)/panel-pomp-block-stress: $(STAMP_DIR)/calibration-hindcast \
                                    $(RUNTIME_DATA_INPUTS) $(PYTHON_INIT) \
                                    $(CALIBRATION_PY) $(SIMULATION_PY) $(VALIDATION_PY)
	@rm -f $@
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.validation.run_panel_pomp_hindcast --forecast-mode block --test-years 2023-2026 --particles 512 --predictive-draws 4096 --mc-replicates 3 --n-jobs $(N_JOBS)
	@touch $@

panel-pomp-block-stress: $(STAMP_DIR)/panel-pomp-block-stress

$(STAMP_DIR)/hindcast: $(STAMP_DIR)/calibrate $(RUNTIME_DATA_INPUTS) \
                       $(PYTHON_INIT) $(SIMULATION_PY)
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_resistance_hindcast --n-jobs $(N_JOBS)
	@touch $@

hindcast: $(STAMP_DIR)/hindcast

$(STAMP_DIR)/publication-diagnostics: $(STAMP_DIR)/simulate $(STAMP_DIR)/hindcast \
                                      config/model_settings.yaml config/parameter_distributions.yaml
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
	@touch $@

publication-diagnostics: $(STAMP_DIR)/publication-diagnostics

$(STAMP_DIR)/joint-psa: $(STAMP_DIR)/simulate config/model_settings.yaml \
                        config/parameter_distributions.yaml manuscript_notes/run_joint_psa_resumable.py
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) manuscript_notes/run_joint_psa_resumable.py --samples 128 --seed 20260521 --sample-batch-size 4 --n-jobs $(N_JOBS)
	@touch $@

joint-psa: $(STAMP_DIR)/joint-psa

$(STAMP_DIR)/fitness-grid-psa: $(STAMP_DIR)/joint-psa config/model_settings.yaml \
                              config/parameter_distributions.yaml
	PERTUSSIS_N_JOBS=$(N_JOBS) $(PYTHON) -m src_python.simulation.run_fitness_grid --skip-deterministic-grid --posterior-draws 0 --psa-benefit-samples -1 --psa-batch-size 4 --n-jobs $(N_JOBS)
	@touch $@

fitness-grid-psa: $(STAMP_DIR)/fitness-grid-psa

$(STAMP_DIR)/publication-tables: $(STAMP_DIR)/publication-diagnostics \
                                 $(STAMP_DIR)/fitness-grid-psa \
                                 $(STAMP_DIR)/calibration-hindcast \
                                 $(STAMP_DIR)/panel-pomp-block-stress
	$(PYTHON) -c "from src_python.simulation.common import write_manuscript_tables; write_manuscript_tables()"
	$(PYTHON) manuscript_notes/generate_health_utility_analysis.py
	$(PYTHON) manuscript_notes/generate_vaccine_pipeline_mapping.py
	$(PYTHON) manuscript_notes/generate_high_risk_review_tables.py
	$(PYTHON) manuscript_notes/generate_lancet_child_adolescent_tables.py
	@touch $@

publication-tables: $(STAMP_DIR)/publication-tables

$(STAMP_DIR)/publication-data: $(STAMP_DIR)/publication-diagnostics \
                               $(STAMP_DIR)/joint-psa \
                               $(STAMP_DIR)/fitness-grid-psa \
                               $(STAMP_DIR)/publication-tables
	@touch $@

publication-data: $(STAMP_DIR)/publication-data

$(STAMP_DIR)/figures: $(STAMP_DIR)/publication-data $(shell find scripts_R -type f -name '*.R')
	$(RSCRIPT) scripts_R/00_setup.R
	$(RSCRIPT) scripts_R/10_figure_set.R
	@touch $@

figures: $(STAMP_DIR)/figures

validate-public:
	$(PYTHON) manuscript_notes/validate_publication_outputs.py

test:
	$(PYTHON) -m pytest tests/ -q

clean-stamps:
	rm -rf $(STAMP_DIR)
