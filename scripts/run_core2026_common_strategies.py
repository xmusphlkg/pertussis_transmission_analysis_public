"""Common-strategy projections (00, M0, 0A, MA, current) at a fitted point.

Writes annual_age_onsets.csv.gz, receipts.csv.gz and complete.json into
<output-dir>/<country>/coverage_<NN>/ for each requested country and coverage.
All strategies of one country and coverage start from the same saved history.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src_python.pertussis.scenarios.common_strategies import STRATEGY_IDS, run_common_strategies
from src_python.pertussis.scenarios.fitted import (build_fitted_inputs, fit_parameters, history_to,
                                                    load_fit)
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.config import REPO_ROOT
from src_python.pertussis.states import MODEL_VERSION
from scripts.guard_projection_observation_variant_20261102 import validate_projection_variant


def source_hashes(fit_dir):
    files = list((REPO_ROOT / 'src_python/pertussis/model').glob('*.py'))
    files += list((REPO_ROOT / 'src_python/pertussis/inputs').glob('*.py'))
    files += [REPO_ROOT / p for p in (
        'src_python/pertussis/grid.py', 'src_python/pertussis/states.py',
        'src_python/pertussis/scenarios/common_strategies.py',
        'src_python/pertussis/scenarios/programme_scenarios.py',
        'src_python/pertussis/scenarios/overlapping_booster.py',
        'src_python/pertussis/scenarios/fitted.py', 'scripts/run_core2026_common_strategies.py',
        'scripts/guard_projection_observation_variant_20261102.py')]
    files += [Path(fit_dir) / n for n in ('run_spec.json', 'fit_summary.csv', 'complete.json')]
    return {str(p.resolve().relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}

_HO = 'outputs/core2026/health_outcomes_20261026'
_CH = 'outputs/core2026/country_health_calibration_20261027'
_MR = 'outputs/core2026/model_resolution_20261029'
FIT_DIRS = {
    'Australia': f'{_HO}/polish_cycle2025_australia_main',
    'England': f'{_HO}/polish_cycle2025_england_main',
    'Sweden': f'{_CH}/Sweden/fit',
    'Japan': f'{_MR}/Japan/preconditioned',
    'New_Zealand': f'{_MR}/New_Zealand/preconditioned',
    'China': f'{_CH}/China/fit',
    'Brazil': 'outputs/core2026/brazil_extended_fit_20261029',
    'Thailand': f'{_CH}/Thailand/fit',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--countries', nargs='+', default=list(FIT_DIRS))
    p.add_argument('--coverages', type=float, nargs='+', default=[0.7])
    p.add_argument('--start', type=int, default=2027)
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--adolescent-age', type=float, default=None)
    p.add_argument('--tdap-years', type=float, default=None,
                   help='Conditional sensitivity: change waning from 2027, retaining fitted history.')
    p.add_argument('--booster-structure', choices=['overlap'], default='overlap')
    p.add_argument('--fit-map', type=Path, help='JSON mapping countries to validated current-model fit directories')
    args = p.parse_args()
    end = args.start + args.horizon
    fit_map = json.loads(args.fit_map.read_text()) if args.fit_map else FIT_DIRS
    for country in args.countries:
        started = time.monotonic()
        fit_dir = Path(fit_map[country])
        spec, row = load_fit(fit_dir)
        validate_projection_variant(spec, row)
        inputs = build_fitted_inputs(spec, end)
        params = fit_parameters(spec, row.to_dict())
        profile = spec.get('country_profile') or {}
        age = args.adolescent_age or profile.get('adolescent_age', 12.)
        share = profile.get('maternal_effective_share', .9)
        hist = history_to(inputs, params, args.start)
        simulator = Simulator
        history_check = dict(historical_model_version=MODEL_VERSION)
        prospective = params if args.tdap_years is None else replace(params, tdap_years=args.tdap_years)
        hashes = source_hashes(fit_dir)
        request = dict(model_version=MODEL_VERSION, start_year=args.start, horizon=args.horizon, booster_structure=args.booster_structure,
                       tdap_years=float(prospective.tdap_years), fit_directory=str(fit_dir), source_hashes=hashes)
        for cov in args.coverages:
            out = args.output_dir / country / f'coverage_{int(round(cov * 100)):02d}'
            if (out / 'complete.json').exists():
                saved = json.loads((out / 'complete.json').read_text())
                if any(saved.get(k) != v for k, v in request.items()):
                    raise ValueError(f'{out}: completed run differs from requested settings or sources')
                continue
            out.mkdir(parents=True, exist_ok=True)
            onsets, receipts = run_common_strategies(
                inputs, prospective, country, history_state=hist.y, history_acc=hist.acc,
                start_year=args.start, end_year=end, coverage=cov, adolescent_age=age,
                maternal_effective_share=share, simulator_class=simulator)
            meta = dict(draw_id=int(row.start_index), structure_id=spec['structure'], start_year=args.start)
            onsets.assign(**meta).to_csv(out / 'annual_age_onsets.csv.gz', index=False)
            receipts.assign(**meta).to_csv(out / 'receipts.csv.gz', index=False)
            (out / 'complete.json').write_text(json.dumps(dict(
                profile=country, design='common_strategy_set', strategies=list(STRATEGY_IDS),
                coverage=cov, adolescent_age=age, maternal_effective_share=share,
                draw_id=int(row.start_index), **request, **history_check,
                parameters=prospective.as_dict(), historical_parameters=params.as_dict(),
                duration_scope=('fitted' if args.tdap_years is None else 'all_Tdap_protection_from_2027_common_fitted_history'),
                fit_success=bool(row.success),
                inference_status='fitted_point_without_parameter_uncertainty',
                elapsed_seconds=time.monotonic() - started), indent=2) + '\n')
            print(f'{country} coverage {cov}: done ({time.monotonic() - started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
