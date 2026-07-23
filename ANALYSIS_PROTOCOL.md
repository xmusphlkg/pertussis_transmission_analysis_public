# Versioned Analysis Protocol

## Status and scope

This document records the analysis protocol for the current pertussis
transmission manuscript release. It was finalised retrospectively during the
2026 reproducibility audit and was not prospectively registered. It therefore
documents the implemented estimands, validation rules, claim boundaries, and
deviations from the superseded workflow; it must not be described as a
preregistered protocol.

- Protocol version: 1.0
- Protocol lock: July 14, 2026
- General evidence lock: July 4, 2026
- Resistance-input lock: July 7, 2026
- Prospective scenario horizon: January 1, 2027, to December 31, 2050
- Canonical configuration: `config/model_settings.yaml`
- Canonical profile configuration: `config/country_profiles.yaml`
- Canonical parameter ranges: `config/parameter_distributions.yaml`

## Questions and estimands

The study addresses four separate questions.

1. Can a semi-mechanistic observation model predict the next reported
   surveillance interval better than prespecified simple baselines at panel
   level?
2. Under a common deterministic reference structure, which programme-only
   scenario produces the lowest conditional annualised symptomatic-case index
   among people younger than 18 years within each accepted profile?
3. How do conclusions change when the decision endpoint is infant severe
   outcomes rather than pooled morbidity among people younger than 18 years?
4. How do resistance-management and hypothetical vaccine-mechanism scenarios
   change conditional residual burden under their stated assumptions?

The primary predictive estimand is one-reporting-interval-ahead aggregate
notification prediction. The primary policy estimand is the within-profile
relative change in the deterministic annualised symptomatic-case index among
people younger than 18 years versus current practice. These estimands are
distinct: predictive validation of aggregate notifications does not validate
the magnitude or ranking of age-specific policy effects.

## Profile eligibility

Profiles require usable aggregate surveillance intervals, vaccination
schedule and coverage inputs, demographic trajectories, contact matrices, and
at least one policy-relevant programme or resistance contrast. Formal
publication calibration and predictive assessment additionally require at
least five recent non-overlapping surveillance intervals and an accepted
state-reconstruction fit.

The publication set contains Australia, Brazil, China, Japan, New Zealand,
Sweden, Thailand, the United Kingdom, and the United States. South Africa has
only three recent non-overlapping intervals and is retained only as an
exploratory deterministic profile. It is excluded from pooled publication
rankings, intervals, and predictive claims.

## Deterministic transmission model

The model is stratified by eight age groups, sensitive or macrolide-resistant
strain status, and vaccine-origin history. Compartments represent susceptible,
exposed, symptomatic infectious, asymptomatic infectious, treated infectious,
recovered, and waned-natural-immunity states. Transmission uses age-specific
mixing, seasonal and supported multi-year forcing, configured pandemic contact
reduction, vaccination histories, treatment, post-exposure prophylaxis, and
resistance assumptions.

State reconstruction targets aggregate reported surveillance counts. It
estimates sensitive-strain transmission, a reporting multiplier, and dated
latent log-transmission deviations using a regularised negative-binomial
target. Acceptance requires numerical convergence, absolute-fit tolerance,
temporal-shape checks, and full-rank, adequately conditioned Gauss–Newton
geometry for the regularised objective; this geometry does not establish data
identifiability. For scenario years after the final fitted year, the final
transmission deviation is propagated through 2050 by the fixed-parameter AR(1)
conditional mean, without future process innovations. The reconstructed state
is a deterministic scenario starting point, not a posterior distribution.

## Scenario groups and endpoints

Scenarios are analysed in separate decision domains:

- programme-only scenarios;
- programme plus resistance-management scenarios; and
- hypothetical future vaccine-mechanism scenarios.

Current practice is the comparator. The main programme-only set comprises
routine schedule timeliness, adolescent booster scale-up, pregnancy Tdap
scale-up, close-contact adult adjuncts, the high-intensity infant-exposure
package, and targeted high-risk post-exposure prophylaxis. Coverage-floor
contrasts, future mechanisms, and combined upper-bound scenarios are excluded
from the current programme-only ranking. Figure 2a uses the production-runtime
coverage-floor-only contrast as a delivery-lever diagnostic alongside routine
timeliness; both use the same within-profile current-practice denominator.

The primary policy endpoint is the conditional annualised symptomatic-case
index per 100 000 people younger than 18 years. Infant hospitalisations and
deaths are a separate severe-outcome axis. Secondary outcomes include
age-specific symptomatic cases, reported cases, infections, and resistant
infections. Rankings use unrounded conditional point estimates. Within each
profile, the leader is the programme-only strategy with the lowest primary
burden and the runner-up is the strategy with the second-lowest burden. The
paired decision margin is defined as runner-up burden minus leader burden, so
positive values favour the selected leader.

## Predictive validation

Deterministic mechanistic expectations are offsets in a semi-mechanistic
discrepancy partially observed Markov process. The latent discrepancy follows
irregular-time candidate processes and observations follow exposure-scaled
NB2 laws. Candidate process and dispersion settings are selected inside each
outer fold from three preceding validation years, with country scores shrunk
towards the panel score.

POMP predictions are stacked with three prespecified baselines: seasonal
naive, exponentially weighted recent rate, and damped log trend. Ensemble
weights use only past validation observations. The formal prequential analysis
uses 512 particles, 4096 predictive draws, and three independent Monte Carlo
replicates for 2023–26 outer folds. Each observation is scored before it is
assimilated.

Diagnostics comprise country-balanced and interval-level log score, absolute
log1p error, 95% coverage, log1p interval width, baseline comparisons,
country-specific coverage, and Monte Carlo variability. A multiplicity-adjusted
Wilson audit tests whether a universal country-calibration claim is tenable.
An unassimilated one-year block forecast is a prespecified stress test, not a
substitute for the next-interval task.

The publication gate supports only panel-level next-interval notification
prediction. It does not authorise claims of absolute national burden,
calibrated prediction in every country, validated annual forecasting, or
validated age-specific policy effects.

## Sensitivity and uncertainty

Figure 2a reports the deterministic production-runtime contrast between a
nominal coverage-floor-only lever and routine timeliness, using relative
reductions against the same current-practice denominator. It is a configured
model-lever comparison, not an empirical causal decomposition.

Figure 2b reports consequence-aware fragility across a fixed-seed design of 128
prespecified paired Latin-hypercube selected-input settings. Its six and only
six sampled inputs are the infant-contact multiplier, baseline vaccine effect
on infectiousness, relative infectiousness of asymptomatic infection,
asymptomatic infectious duration, resistant-strain relative fitness, and the
programme PEP-coverage multiplier. Every setting compares the same six
programme-only strategies (routine timeliness, adolescent booster, pregnancy
Tdap scale-up, close-contact adult adjunct, infant-exposure reduction, and
targeted high-risk PEP) on the primary symptomatic-case endpoint among people
younger than 18 years. Figure 2b combines the exact count of settings in which
the locked reference choice remains lowest-burden with the empirical 95th
percentile of fixed-reference regret, defined as 100 times reference-choice
burden minus setting-specific minimum burden, divided by current-practice
burden. These quantities are deterministic design summaries, not probabilities,
confidence intervals, or expected regret. Resistance-management uptake and
guided-pathway PEP reach are not Figure 2b inputs.

The resistance-management sensitivity is a separate five-year, fixed-seed
design of 128 paired Latin-hypercube settings with exactly two prospective
implementation inputs: resistance-management uptake and guided-pathway PEP
reach. Each setting compares resistance-guided management with routine
timeliness under two prespecified structural strata, restored and not-restored
PEP effectiveness against resistant infection. Both strata are reported; they
are not assigned probabilities or used as mixture weights. Results from this
design are independent resistance-management design summaries and are not part
of Figure 2b or its six-dimensional programme-only sample.

Figure 2c reports the locked relative reduction for all 54 profile-strategy
combinations. Cell brackets are paired full-refit parametric-bootstrap 95%
estimation confidence intervals and are neither Bayesian credible intervals nor
future-observation prediction intervals. Programme-only scenarios are ranked on
the primary symptomatic-case endpoint among people younger than 18 years;
infant ranks remain a distinct secondary endpoint.

Additional diagnostics address age-pattern weighting, schedule coverage versus
timeliness, analysis window, pandemic shock recovery, infant contacts,
resistance fitness, treatment and PEP implementation, vaccine mechanisms, and
structural assumptions. Negative or failed diagnostics are retained and
reported.

## Missing data and exclusions

The workflow does not impute individual participant data. Missing country-level
programme or resistance inputs use explicit, versioned assumptions or
conservative anchors documented in the supplementary tables. Where a profile
lacks sufficient surveillance intervals, it is excluded from formal
publication calibration and prediction rather than borrowing an unreported
time series. Third-party raw data are redistributed only when permitted;
otherwise the source location and processing code are retained.

## Deviations from the superseded workflow

The following changes were made during the reproducibility audit and are
treated as protocol deviations rather than hidden analytic flexibility.

1. The earlier high-dimensional MCMC route was removed from the publication
   dependency graph because it was computationally inefficient and did not
   resolve the observation-identifiability problem.
2. The main predictive route was replaced by leakage-safe prequential
   discrepancy-POMP validation with prespecified baselines and a separate
   failed annual block stress test.
3. Policy uncertainty was relabelled as deterministic Latin-hypercube
   selected-input sensitivity rather than a posterior interval.
4. The prospective scenario start was moved to January 1, 2027, after the
   2026 evidence and validation lock.
5. South Africa was removed from the formal publication set after applying the
   minimum-interval rule; the publication set therefore changed from ten to
   nine profiles.
6. All policy quantities were relabelled as conditional indices or contrasts,
   and claims of absolute national burden or validated forecasts to 2050 were
   removed.

## Reproducibility and release rules

The release is rebuilt with bounded process parallelism and can use up to 100
workers on the current server. Input, configuration, source, and output hashes
are retained in run metadata. A release is current only when the predictive
gate, annual block stress record, output-window validation, submission-text
audit, and complete test suite all pass against the same source/configuration
state.

The public repository is the citable reproducibility snapshot:
https://github.com/xmusphlkg/pertussis_transmission_analysis_public
