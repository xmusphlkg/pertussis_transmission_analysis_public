"""Re-estimate matched COVID-history scenarios with the current core model.

Restricted fits estimate NPI intensity, importation and observation parameters,
holding immunity parameters at the same parent point. Full fits are optional.
Scenario envelopes are structural sensitivities, not measured contact intervals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src_python.pertussis.config import REPO_ROOT as ROOT
from src_python.pertussis.inference.core2026 import Core2026Inference, InferenceSettings, ParallelGradient, FAILED_OBJECTIVE
from src_python.pertussis.inputs.bundle import NPI_TIMELINE
from src_python.pertussis.model.simulate import Simulator
from src_python.pertussis.scenarios.fitted import build_fitted_inputs, load_fit

OUT_ROOT = ROOT / 'outputs/core2026/covid_refit_20261104'
FIT_MAP = ROOT / 'outputs/core2026/current_preserving_fit_map_20261103.json'
SCENARIOS = ('baseline', 'gradual_import', 'recovery_365', 'recovery_90', 'npi_lower', 'npi_upper', 'no_npi')


def scenario_timelines(country, dates, scenario_id):
    """Build daily non-home contact and import multipliers without file mutation."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f'unknown COVID scenario: {scenario_id}')
    country = 'United_Kingdom' if country == 'England' else country
    table = pd.read_csv(NPI_TIMELINE)
    table = table.loc[table.country.eq(country)].sort_values('period_start')
    reduction = np.zeros(len(dates))
    import_suppression = np.zeros(len(dates))
    column = {'npi_lower': 'contact_reduction_lower', 'npi_upper': 'contact_reduction_upper'}.get(scenario_id, 'contact_reduction_mean')
    for row in table.to_dict('records'):
        start = np.datetime64(row['period_start'], 'D')
        end = np.datetime64(row['period_end'], 'D') + np.timedelta64(1, 'D')
        value = float(row[column])
        # Only extend documented recovery ramps; monthly Australia values have
        # no ramp and must not create overlapping year-long restriction tails.
        ramp = int(row['ramp_days'])
        if ramp and scenario_id in ('recovery_365', 'recovery_90'):
            ramp = 365 if scenario_id == 'recovery_365' else 90
        inside = (dates >= start) & (dates < end)
        reduction[inside] = np.maximum(reduction[inside], value)
        if value > 0:
            import_suppression[inside] = .9
        if ramp:
            after = (dates >= end) & (dates < end + np.timedelta64(ramp, 'D'))
            progress = 1 - (dates[after] - end).astype(int) / ramp
            reduction[after] = np.maximum(reduction[after], value * progress)
            if value > 0:
                import_suppression[after] = np.maximum(import_suppression[after], .9 * progress)
    if scenario_id == 'no_npi':
        return np.zeros(len(dates)), np.ones(len(dates))
    imports = 1 - import_suppression if scenario_id == 'gradual_import' else np.where(reduction > 0, .1, 1.)
    return np.clip(reduction, 0, 1), imports


def build_scenario_inputs(spec, end_year, scenario_id=None):
    """Projection entry point: retain the same scenario during the whole history."""
    scenario_id = scenario_id or spec.get('covid_scenario', 'baseline')
    inputs = build_fitted_inputs(spec, end_year)
    inputs.npi_reduction, inputs.import_multiplier = scenario_timelines(spec['country'], inputs.calendar.dates, scenario_id)
    inputs.metadata['covid_scenario'] = scenario_id
    return inputs


def build_problem(spec, scenario_id, *, iterations=150, workers=2):
    settings = replace(InferenceSettings(**spec['settings']), max_iterations=iterations, n_workers=workers, gradient_step=.00005)
    problem = Core2026Inference(spec['country'], settings)
    problem.inputs.npi_reduction, problem.inputs.import_multiplier = scenario_timelines(spec['country'], problem.calendar.dates, scenario_id)
    problem.simulator = Simulator(problem.inputs)
    problem.base_y0, problem.base_acc0 = problem.simulator.initial_state()
    problem._cache.clear()
    return problem


def active_coordinates(problem, scope):
    if scope == 'full':
        return np.arange(len(problem.space.names))
    return np.array([i for i,n in enumerate(problem.space.names)
                     if n in ('npi_scale', 'importation_per_million_per_year') or i in problem.observation_index], dtype=int)


def fit_one(country, scenario_id, *, iterations=150, workers=2, scope='restricted', output_root=OUT_ROOT, parent_override=None):
    parent = Path(parent_override) if parent_override else ROOT / json.loads(FIT_MAP.read_text())[country]
    spec, parent_row = load_fit(parent)
    if spec['country'] != country:
        raise ValueError('parent country mismatch')
    best_start = int(json.loads((parent / 'complete.json').read_text())['best_start'])
    coordinate_path = parent / f'start_{best_start}_coordinates.npy'
    z0 = np.load(coordinate_path, allow_pickle=False)
    problem = build_problem(spec, scenario_id, iterations=iterations, workers=workers)
    active = active_coordinates(problem, scope)
    problem.structural_index = np.intersect1d(problem.structural_index, active)
    problem.observation_index = np.intersect1d(problem.observation_index, active)
    output = Path(output_root) / country / scenario_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial = problem.objective_z(z0)
    if not np.isfinite(initial) or initial >= FAILED_OBJECTIVE:
        raise ValueError('parent start is infeasible under scenario')
    baseline_problem = build_problem(spec, 'baseline', iterations=iterations, workers=workers)
    baseline_objective = baseline_problem.objective_z(z0)
    if parent_override is None and not np.isclose(baseline_objective, float(parent_row.objective), atol=1e-5, rtol=1e-8):
        raise ValueError(f'parent baseline replay mismatch: {baseline_objective} vs {parent_row.objective}')
    replay = float(baseline_objective)
    del baseline_problem
    run = dict(spec)
    run.update(settings=asdict(problem.settings), parent_fit=str(parent), covid_scenario=scenario_id,
               reestimation_scope=scope, estimated_coordinate_names=[problem.space.names[i] for i in active],
               fixed_coordinate_names=[n for i,n in enumerate(problem.space.names) if i not in active],
               parent_baseline_replay_objective=replay, initial_scenario_objective=float(initial),
               selection='training posterior objective; held-out outcomes not used for selection',
               projection_builder='scripts.run_covid_refit_20261104.build_scenario_inputs',
               scenario_interpretation='Structural sensitivity of historical contact/import recovery; unchanged observed outcomes and diagnostic timeline')
    (output/'run_spec.json').write_text(json.dumps(run, indent=2)+'\n')
    np.save(output/'initial_coordinates.npy', z0)
    gradient = ParallelGradient(problem, workers, .00005)
    lower, upper = np.asarray(problem.space.bounds).T
    trace = []
    best = [float(initial), z0.copy()]
    try:
        # Diagonal scaling affects optimisation only; no uncertainty inference.
        points = []
        h = .0005
        for i in active:
            for sign in (1,-1):
                z = z0.copy();z[i] += sign*h;points.append(z)
        values = gradient.evaluate_many(points)
        curvature = np.array([(values[2*k]-2*initial+values[2*k+1])/h**2 for k in range(len(active))])
        scales = np.clip(1/np.sqrt(np.where(np.isfinite(curvature)&(curvature>0),curvature,1)), .001,2.)
        def unpack(u):
            z=z0.copy();z[active]+=scales*u;return z
        def fun(u):
            z=unpack(u);f,g=gradient.value_and_grad(z)
            if np.isfinite(f) and f<best[0]:
                best[:]=[float(f),z.copy()]
                np.save(output/'checkpoint.npy',z)
            return f,g[active]*scales
        def callback(u):
            trace.append(best[0])
            (output/'progress.json').write_text(json.dumps({'iteration':len(trace),'best_objective':best[0],'seconds':time.monotonic()-started})+'\n')
            if len(trace)%10==0:print(country,scenario_id,len(trace),best[0],flush=True)
        result=minimize(fun,np.zeros(len(active)),jac=True,method='L-BFGS-B',
                        bounds=list(zip((lower[active]-z0[active])/scales,(upper[active]-z0[active])/scales)),
                        callback=callback,options={'maxiter':iterations,'maxcor':20,'gtol':1e-5,'ftol':1e-11})
        z=best[1];checks=[]
        for step in (.00005,.00001):
            gradient.step=step;f,g=gradient.value_and_grad(z)
            projected=np.where((np.isclose(z,lower)&(g>0))|(np.isclose(z,upper)&(g<0)),0,g)
            checks.append({'step':step,'objective':float(f),'max_active_projected_gradient':float(np.max(np.abs(projected[active])))})
        objective,total,age,components=problem.evaluate(z)
        converged=all(c['max_active_projected_gradient']<.05 for c in checks)
        np.save(output/'start_0_coordinates.npy',z)
        total.to_csv(output/'start_0_total_comparison.csv',index=False)
        age.to_csv(output/'start_0_age_comparison.csv',index=False)
        pd.DataFrame({'iteration':range(1,len(trace)+1),'objective':trace}).to_csv(output/'start_0_trace.csv',index=False)
        pd.DataFrame([{'start_index':0,'objective':objective,'success':converged,'optimizer_success':bool(result.success),
                       'nit':int(result.nit),'message':str(result.message),'variant':problem.variant.name,
                       'initial_objective':initial,'max_abs_projected_gradient_checked':max(c['max_active_projected_gradient'] for c in checks),
                       **problem.space.natural(z),**components}]).to_csv(output/'fit_summary.csv',index=False)
        complete={'best_start':0,'all_starts_finished':True,'fit_converged':converged,
                  'status':'completed_local_fit' if converged else 'completed_not_stationary','gradient_checks':checks,
                  'gradient_scope':scope,'seconds':time.monotonic()-started,'optimizer_success':bool(result.success),
                  'selection':run['selection']}
        (output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
        print(country,scenario_id,'finished',objective,'stationary',converged,'seconds',complete['seconds'],flush=True)
    finally:
        gradient.close()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--country',required=True)
    parser.add_argument('--scenario',choices=SCENARIOS,required=True)
    parser.add_argument('--iterations',type=int,default=150)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--scope',choices=('restricted','full'),default='restricted')
    parser.add_argument('--output-root',type=Path,default=OUT_ROOT)
    parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    fit_one(args.country,args.scenario,iterations=args.iterations,workers=args.workers,scope=args.scope,
            output_root=args.output_root,parent_override=args.parent)


if __name__=='__main__':main()
