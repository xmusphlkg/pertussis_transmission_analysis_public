#!/usr/bin/env python3
"""Reanalyse public CHAMPION-1 participant AUCs; no transmission modelling.

Read XLSX using stdlib ZIP/XML. NumPy supplies participant bootstrap sampling;
SciPy supplies Student t quantiles for the published geometric-mean interval.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path
import platform
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/core2026/05_evidence/mucosal_vaccines/clinical_sources/41541_2026_1443_MOESM2_ESM.xlsx"
OUT = SOURCE.parent.parent / "trial_reanalysis"
SHEET = "Delta Immunoglobulin responses"
SOURCE_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41541-026-01443-7/MediaObjects/41541_2026_1443_MOESM2_ESM.xlsx"
ARTICLE_URL = "https://www.nature.com/articles/s41541-026-01443-7"
TRIAL_URL = "https://eprints.soton.ac.uk/509183/1/PIIS2666524725001399.pdf"
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
GROUPS = ("BPZE1", "Placebo")


def read_participants(path: Path) -> tuple[list[dict], list[dict]]:
    """Resolve sheet by name and workbook relationship, then require A/B/C schema."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = [s for s in book.findall("x:sheets/x:sheet", NS) if s.get("name") == SHEET]
        if len(sheets) != 1:
            raise ValueError("Expected exactly one named AUC sheet")
        rid = sheets[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = [r.get("Target") for r in relations if r.get("Id") == rid]
        if len(targets) != 1:
            raise ValueError("Missing or duplicate sheet relationship")
        target = targets[0]
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        rows = ET.fromstring(archive.read(member)).findall("x:sheetData/x:row", NS)
        participants, excluded = [], []
        header_seen = False
        for row in rows:
            cells = {}
            for cell in row.findall("x:c", NS):
                column = cell.get("r").rstrip("0123456789")
                if column not in ("A", "B", "C"):
                    continue
                value = cell.find("x:v", NS)
                raw = "" if value is None else value.text
                if cell.get("t") == "s":
                    raw = shared[int(raw)]
                elif cell.get("t") == "inlineStr":
                    raw = "".join(cell.find("x:is", NS).itertext())
                cells[column] = raw
            if cells.get("A") == "Participant number":
                if cells.get("B") != "Assignment" or cells.get("C") != "B. pertussis colonisation density (AUC)":
                    raise ValueError("AUC column schema changed")
                header_seen = True
                continue
            if cells.get("B", "") == "":
                if cells.get("A", "").isdigit():
                    raise ValueError("Participant is missing assignment")
                continue
            if not header_seen or cells["B"] not in GROUPS:
                raise ValueError("Unrecognised participant row")
            participant = {"participant_id": int(cells["A"]), "assignment": cells["B"], "source_excel_row": int(row.get("r"))}
            if cells.get("C") == "Not challenged with Bp":
                excluded.append({**participant, "reason": cells["C"]})
                continue
            auc = float(cells["C"])
            if not math.isfinite(auc) or auc < 0:
                raise ValueError("AUC must be finite and nonnegative")
            participants.append({**participant, "auc_cfu_per_ml_days": auc, "late_positive_auc": int(auc > 0)})
    if len({p["participant_id"] for p in participants + excluded}) != len(participants + excluded):
        raise ValueError("Duplicate participant identifier")
    return participants, excluded


def metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Last axis is participants; zeros remain zeros except in geometric summaries."""
    positive_n = (values > 0).sum(axis=-1)
    total = values.sum(axis=-1)
    conditional = np.divide(total, positive_n, out=np.full_like(total, np.nan, dtype=float), where=positive_n > 0)
    return {
        "late_positive_risk": (values > 0).mean(axis=-1),
        "arithmetic_auc_mean": values.mean(axis=-1),
        "auc_median": np.median(values, axis=-1),
        "positive_subset_arithmetic_auc_mean": conditional,
        "geometric_auc_mean_zeros_as_one": np.exp(np.log(np.where(values == 0, 1, values)).mean(axis=-1)),
    }


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b != 0))


def effects(v: dict, p: dict) -> dict:
    risk = safe_ratio(v["late_positive_risk"], p["late_positive_risk"])
    whole = safe_ratio(v["arithmetic_auc_mean"], p["arithmetic_auc_mean"])
    conditional = safe_ratio(v["positive_subset_arithmetic_auc_mean"], p["positive_subset_arithmetic_auc_mean"])
    geometric = safe_ratio(v["geometric_auc_mean_zeros_as_one"], p["geometric_auc_mean_zeros_as_one"])
    return {
        "late_positive_risk_ratio": risk,
        "one_minus_late_positive_risk_ratio": 1 - risk,
        "late_positive_risk_difference": v["late_positive_risk"] - p["late_positive_risk"],
        "whole_group_arithmetic_auc_ratio": whole,
        "one_minus_whole_group_arithmetic_auc_ratio": 1 - whole,
        "positive_subset_arithmetic_auc_ratio_descriptive": conditional,
        "geometric_auc_ratio_zeros_as_one": geometric,
        "one_minus_geometric_auc_ratio_zeros_as_one": 1 - geometric,
        "decomposition_risk_times_positive_auc_ratio": risk * conditional,
    }


def interval(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    lower, upper = np.quantile(finite, [0.025, 0.975]) if finite.size else (None, None)
    return {"bootstrap_percentile_95_lower": lower, "bootstrap_percentile_95_upper": upper,
            "bootstrap_valid_replicates": int(finite.size), "bootstrap_undefined_replicates": int(np.size(values) - finite.size)}


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if isinstance(v, (float, np.floating)) and not math.isfinite(v) else v) for k, v in row.items()})
    return stream.getvalue().encode()


def self_check() -> dict:
    v, p = np.array([0., 2., 4., 0.]), np.array([0., 1., 2., 3.])
    e = effects(metrics(v), metrics(p))
    assert np.isclose(e["whole_group_arithmetic_auc_ratio"], 1)
    assert np.isclose(e["decomposition_risk_times_positive_auc_ratio"], 1)
    zero = metrics(np.zeros(4))
    assert zero["arithmetic_auc_mean"] == 0 and np.isnan(zero["positive_subset_arithmetic_auc_mean"])
    assert zero["geometric_auc_mean_zeros_as_one"] == 1
    assert np.isnan(effects(metrics(v), zero)["whole_group_arithmetic_auc_ratio"])
    assert effects(zero, metrics(p))["whole_group_arithmetic_auc_ratio"] == 0
    return {"synthetic_decomposition": True, "zero_auc_kept_in_arithmetic": True,
            "empty_positive_subset_and_zero_denominator_reported_undefined": True}


def build(source: Path, draws: int, seed: int) -> dict[str, bytes]:
    if draws < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates")
    checks = self_check()
    participants, excluded = read_participants(source)
    assert len(participants) == 45 and len(excluded) == 1
    arrays = {g: np.array([p["auc_cfu_per_ml_days"] for p in participants if p["assignment"] == g]) for g in GROUPS}
    assert tuple(arrays[g].size for g in GROUPS) == (24, 21)
    assert tuple(int((arrays[g] == 0).sum()) for g in GROUPS) == (14, 7)
    rng = np.random.default_rng(seed)
    points = {g: metrics(arrays[g]) for g in GROUPS}
    boot = {g: metrics(arrays[g][rng.integers(0, arrays[g].size, size=(draws, arrays[g].size))]) for g in GROUPS}
    group_rows, published_rows = [], []
    for g in GROUPS:
        a = arrays[g]
        for metric, estimate in points[g].items():
            group_rows.append({"assignment": g, "n": a.size, "n_zero": int((a == 0).sum()),
                               "n_positive": int((a > 0).sum()), "metric": metric,
                               "estimate": float(estimate), **interval(boot[g][metric])})
        logged = np.log(np.where(a == 0, 1, a))
        margin = t.ppf(0.975, a.size - 1) * logged.std(ddof=1) / math.sqrt(a.size)
        low, high = np.exp([logged.mean() - margin, logged.mean() + margin])
        expected = (18, 4, 87) if g == "BPZE1" else (611, 61, 6105)
        actual = (round(float(np.exp(logged.mean()))), round(float(low)), round(float(high)))
        assert actual == expected, (g, actual, expected)
        published_rows.append({"assignment": g, "n": a.size, "geometric_auc_mean_zeros_as_one": float(np.exp(logged.mean())),
                               "log_student_t_95_lower": low, "log_student_t_95_upper": high,
                               "published_rounded_mean": expected[0], "published_rounded_lower": expected[1], "published_rounded_upper": expected[2]})
    assert np.isclose(points["BPZE1"]["arithmetic_auc_mean"], 3971.9375, rtol=0, atol=1e-9)
    assert np.isclose(points["Placebo"]["arithmetic_auc_mean"], 69698.38095238095, rtol=0, atol=1e-8)
    point_effects, boot_effects = effects(points["BPZE1"], points["Placebo"]), effects(boot["BPZE1"], boot["Placebo"])
    assert np.isclose(point_effects["whole_group_arithmetic_auc_ratio"], point_effects["decomposition_risk_times_positive_auc_ratio"])
    valid = np.isfinite(boot_effects["decomposition_risk_times_positive_auc_ratio"])
    assert np.allclose(boot_effects["whole_group_arithmetic_auc_ratio"][valid], boot_effects["decomposition_risk_times_positive_auc_ratio"][valid], rtol=1e-12, atol=1e-14)
    effect_rows = [{"metric": k, "estimate": float(v), **interval(boot_effects[k])} for k, v in point_effects.items()]
    bootstrap_rows = [{"replicate": i + 1,
                       **{f"{g}_{k}": float(v[i]) for g in GROUPS for k, v in boot[g].items()},
                       **{k: float(v[i]) for k, v in boot_effects.items()}} for i in range(draws)]
    outputs = {"participant_auc.csv": csv_bytes(participants), "excluded_participants.csv": csv_bytes(excluded),
               "group_summary.csv": csv_bytes(group_rows), "published_geometric_summary_reconstruction.csv": csv_bytes(published_rows),
               "effect_decomposition.csv": csv_bytes(effect_rows),
               "bootstrap_joint_effects.csv.gz": gzip.compress(csv_bytes(bootstrap_rows), mtime=0)}
    checks.update({"numeric_auc_count": 45, "group_counts": [24, 21], "group_zero_counts": [14, 7],
                   "published_geometric_means_and_log_t_intervals_reproduced": True,
                   "arithmetic_means_original_zeros_reproduced": True,
                   "point_decomposition": True, "bootstrap_valid_decompositions": int(valid.sum())})
    definitions = {
        "population": "45 challenged participants with numeric AUC in the public sheet; group counts match the 2025 trial mITT analysis. One unchallenged participant was excluded.",
        "endpoint": "B. pertussis colonisation density AUC, CFU/mL times days, for days 9, 11 and 14 after challenge; late_positive_auc is AUC > 0. The workbook supplies cumulative AUC, while the 2025 trial Table 2 defines its observation days.",
        "direction": "All group ratios are BPZE1 divided by Placebo; risk difference is BPZE1 minus Placebo. One-minus-ratio fields retain negative values when present.",
        "arithmetic_auc": "Original numerical AUC values, including exact zeros. Whole-group mean = positive fraction times positive-subset arithmetic mean.",
        "positive_subset": "Descriptive mean among observed AUC-positive participants in each assigned arm; the subsets are selected after vaccination. The conditional mean ratio quantifies the observed composition of whole-group AUC, rather than a randomised causal effect among a common subgroup.",
        "geometric_reconstruction": "Replace only exact zero AUC by 1 before taking logs; exp(mean(log AUC)) and exp(mean +/- t(0.975,n-1)*sample_SD(log AUC)/sqrt(n)) reproduce the published rounded geometric mean and 95% interval. This transformation is used only for these geometric summaries.",
        "bootstrap": "Independent within-arm resampling of participant AUCs with replacement at original sizes 24 and 21; all outcomes and ratios calculated from the same participant resample. Percentile 2.5/97.5 intervals describe small-sample empirical uncertainty conditional on the observed challenged participants. These intervals are separate from the published log-t intervals. Zero positive counts or zero control denominators yield undefined relevant statistics; valid and undefined counts are reported per metric.",
        "decomposition": "whole_group_arithmetic_auc_ratio = late_positive_risk_ratio * positive_subset_arithmetic_auc_ratio_descriptive whenever both subsets are nonempty.",
        "use": "Trial endpoint evidence for late colonisation and cumulative culture density after controlled challenge. Transmission hazard, onward infectiousness and duration of protection require their own mapping and evidence.",
    }
    meta = {"analysis": "bpze1_public_participant_auc_reanalysis", "source_path": source.name if not source.resolve().is_relative_to(ROOT) else str(source.resolve().relative_to(ROOT)),
            "source_urls": {"workbook": SOURCE_URL, "data_article": ARTICLE_URL, "trial_table_2": TRIAL_URL},
            "license": {"name": "Creative Commons Attribution 4.0 International", "url": "https://creativecommons.org/licenses/by/4.0/", "attribution": "Hill et al., npj Vaccines (2026), doi:10.1038/s41541-026-01443-7, Supplementary Data workbook, Delta Immunoglobulin responses sheet, columns A-C"},
            "bootstrap": {"replicates": draws, "seed": seed, "generator": "numpy.default_rng PCG64", "percentile_method": "linear"},
            "trial_reported_relative_geometric_auc_reduction": {"estimate_percent": 97, "ci95_lower_percent": 59, "ci95_upper_percent": 100, "source": "2025 trial Table 2, mITT population", "status": "Published relative-effect interval recorded separately; reconstruction checks concern the two group-specific geometric means and log-t intervals."},
            "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "definitions": definitions, "checks": checks}
    outputs["metadata.json"] = (json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261107)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Recompute and require byte-identical outputs")
    mode.add_argument("--refresh", action="store_true", help="Regenerate outputs (also the default)")
    parser.add_argument("--self-check", action="store_true", help="Run small synthetic algebra checks only")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    outputs = build(args.source, args.bootstrap_replicates, args.seed)
    if args.check:
        for name, data in outputs.items():
            if not (args.output / name).exists() or (args.output / name).read_bytes() != data:
                raise ValueError(f"Output differs from fresh calculation: {name}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            (args.output / name).write_bytes(data)
    print(json.dumps({"mode": "check" if args.check else "write", "outputs": list(outputs),
                      "checks": json.loads(outputs["metadata.json"])["checks"]}, indent=2))


if __name__ == "__main__":
    main()
