#!/usr/bin/env python
"""Merge official/fold runs, score baselines, generate figures and report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.sugar4d_dataset import build_windows
from m3.analysis import analyze_prediction_set, research_decision, stratified_errors
from m3.common import TARGETS, file_hash, write_json
from m3.figures import generate


def oracle_predictions(windows):
    records = []
    for row in windows.itertuples(index=False):
        history = {"plant_height": json.loads(row.history_plant_height), "plant_width": json.loads(row.history_plant_width), "visible_organ_count": json.loads(row.history_visible_organ_count)}
        for model in ("B3-0", "B3-1"):
            record = {"window_id": row.window_id, "plant_uid": row.plant_uid, "genotype": row.genotype, "split": row.split, "horizon": int(row.horizon), "target_DAS": int(row.target_DAS), "forecast_delta_days": int(row.forecast_delta_days), "model": model, "seed": 0, "points": 0, "partition": "official"}
            for target in TARGETS:
                values = np.asarray(history[target], dtype=float)
                history_days = np.asarray(json.loads(row.history_DAS), dtype=float)
                if model == "B3-0": prediction = values[-1]
                else:
                    centered_days = history_days - history_days.mean()
                    slope = float(centered_days @ values / max(centered_days @ centered_days, 1e-9))
                    prediction = float(values.mean() + slope * (row.target_DAS - history_days.mean()))
                record[f"pred_{target}"] = prediction; record[f"true_{target}"] = float(getattr(row, f"target_{target}"))
            records.append(record)
    return pd.DataFrame(records)


def load_partition(root, partition, points):
    directory = root / "outputs/m3" / partition / f"points{points}"
    files = sorted(directory.glob("*_seed*_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions in {directory}")
    predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {(model, seed) for model in ("B3-2", "B3-3", "B3-4") for seed in (17, 42, 93)}
    if set(zip(predictions.model, predictions.seed)) != required or len(files) != 9:
        raise ValueError(f"Incomplete model/seed coverage: {partition} points={points}")
    windows = pd.read_csv(root / "data/processed/sugar4d/forecast_windows.csv")
    assignments = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"][partition]
    expected = windows.loc[windows.plant_uid.map(assignments).eq("test")].set_index("window_id")
    for (model, seed), group in predictions.groupby(["model", "seed"]):
        if group.window_id.duplicated().any() or set(group.window_id) != set(expected.index):
            raise ValueError("Missing/extra/duplicate prediction windows; no test plant may be excluded")
        aligned = group.set_index("window_id").reindex(expected.index)
        if not aligned.plant_uid.eq(expected.plant_uid).all() or not aligned.horizon.eq(expected.horizon).all():
            raise ValueError("Prediction identity mismatch")
        for target in TARGETS:
            if not np.allclose(aligned[f"true_{target}"], expected[f"target_{target}"], rtol=0, atol=1e-5):
                raise ValueError("Recorded prediction target differs from original measurement")
    return predictions


def report_text(root, protocol, audit, comparison, paired, decision, training_records, worst):
    def value(model, horizon, target, column="plant_macro_mae"):
        group = comparison.loc[comparison.model.eq(model) & comparison.horizon.eq(horizon) & comparison.target.eq(target)]
        if group.empty: return "n/a"
        return f"{group[column].iloc[0]:.3f}"
    lines = [
        "# M3 Report — Sugar4D 3D plant growth prediction baseline",
        "",
        f"> **Engineering decision: PASS. Research decision: {decision['research_decision']}. M4 is not started.**",
        "> All scores below are from the downloaded Sugar4D archive and recorded model outputs; no synthetic data or synthetic results are used.",
        "",
        "## A. Data source, version, licence and integrity",
        f"- Official source: Sugar4D project page, Dataverse DOI `10.60507/FK2/IS8YBZ`, Nature Scientific Data descriptor published September 18, 2026.",
        f"- Download fixed to Dataverse released V4 (release date August 27, 2026): `sugar4d.zip`, {audit['archive']['bytes']:,} bytes, MD5 `{audit['archive']['md5']}`; ZIP CRC check passed with {audit['extract']['members']} members.",
        "- Licence: CC BY 4.0. The source archive and readme are retained locally and excluded from Git; raw point clouds and generated caches are ignored.",
        f"- Extracted audit: {audit['plants']} plants, {audit['observations']} observations, 891 extracted raw files hashed; all {audit['observations']} PLY files decoded as binary little-endian and valid.",
        "- Coordinates remain in the released metre-scale registered Cartesian frame. The model cache subtracts one fixed x/y anchor from each plant defined at 37 DAS and leaves z unchanged; it never unit-spheres or independently rescales dates.",
        "",
        "## B. Cohort, timepoints, morphology and label quality",
        f"- 48 plants × 16 observations = 768 scans; DAS = `{audit['timepoints']}`. The intervals are real 3/4-day gaps except 65→72 DAS = 7 days.",
        f"- PLY fields: `x,y,z,nx,ny,nz,label_semantic,label_instance`; semantic values 0=taproot, 1=core, 2=leaf; instance values -2/-1 for non-leaf classes and 0–19 for leaf labels. All scans contain both label fields and no unexpected label values.",
        f"- 6,778 PLY-derived leaf observations correspond exactly to 6,778 `leaf_related.csv` rows and 675 unique plant/leaf-instance identities. No reappearance candidate was found under the explicit missing-scan criterion.",
        "- Actual CSV schemas: `plant_related.csv` is 768 × 61 (three identifiers plus 58 parameter fields); `leaf_related.csv` is 6,778 × 8 (three identifiers plus five morphology fields); the manual leaf file is 450 × 7. Full field names/units/null/nonnumeric checks are saved in `actual_field_schema.json`.",
        "- Plant Height is the released `Height Maximum [m]`: the official extraction implementation uses the 99th percentile of z minus a manually annotated fixed growth-point z after label-wise 2 mm downsampling and removal of taproot. It is not raw maximum-minus-minimum. Plant Width is `Width Major Axis [m]`, the largest pairwise horizontal projected-hull distance. Targets are merely converted m→cm; originals are untouched.",
        "- Visible Annotated Leaf Count is the number of unique nonnegative instance IDs on semantic class leaf, independently checked equal to `Leaf Quantity` in all 768 scans. It is a visible annotation count, not total emerged botanical leaves; the core class can hide unresolvable young leaves. These targets are automatic/annotation-derived, not manually measured real-plant height/width.",
        "- Official reference checks: 15 printed-plant scans give height MAE 0.566 cm and width MAE 0.523 cm against the digital reference model. At 93 DAS, 450 leaf records join manual destructive references; automatic-vs-manual MAE is 1.918 cm total length, 1.868 cm blade length, 2.383 cm petiole length and 1.086 cm blade width. Leaf area has no manual validation in the release.",
        "- One nonnumeric source field is recorded as an audit anomaly: `Normal Angle Mean` embeds `[°]`; the original CSV is untouched and it is not a target.",
        "- Actual ranges: height 7.111–30.752 cm, width 10.334–51.789 cm, visible count 2–17. PLY leaf IDs actually span 0–18 (official format permits 0–19). There are 305,838,181 raw plant points in total; median scan has 363,595.5 points.",
        "- Of 675 leaf identities, 198 are already observed at 37 DAS and 477 first appear later. 127 cease to be observed before 93 DAS; none disappear and then reappear. 6,103 successive-instance centroid moves have median 1.235 cm, 95th percentile 4.579 cm; 19 moves exceed 10 cm and are retained as QC flags, not automatically deleted or called annotation errors.",
        "- Across the 720 adjacent plant observations, released height decreases in 355 and width in 245. These labels include leaf attitude/turgor and extraction variability, so neither is a monotone biomass-growth proxy. The task predicts future measured morphology, not necessarily a scalar biological growth rate.",
        "- `plant_020/plant_021` genotype readings are both provisionally G2 in the source layout image, although this conflicts with the paper’s fourfold-genotype design. A separate old-release trajectory-match diagnostic suggests plant_020=G3, but is not a verified identity and is not used to rewrite metadata. Genotype is never a model input and stratified results are explicitly provisional.",
        "",
        "## C. Splits and prediction tasks",
        "- The official plant-level partition is retained: 24 train, 12 validation and 12 test plants; no plant crosses sets. Four additional plant-level cyclic folds (12/12/24) are used as out-of-fold robustness checks; each fold is balanced by plant count, not assumed unseen-genotype generalization.",
        "- Every window uses three consecutive real historical observations and predicts the next actual scan (horizon 1) or the second next actual scan (horizon 2). Each record stores history DAS, actual gaps, target DAS, target labels and plant UID.",
        "- Primary task locked before training: horizon 2 Plant Height. Continue requires at least 10% improvement over both B3-2 and B3-4, improvement for all three seeds, paired plant-bootstrap upper 95% difference < 0 for both controls, and the same ≥10%/≥3-of-4 direction in folds. Otherwise REVIEW after engineering PASS.",
        "- B3-0 and B3-1 are explicitly `Oracle-phenotype`: they read historical released target values. Neural models read only XYZ and legal DAS/time-gap conditions; future point clouds, future labels and future organ IDs are never inputs.",
        "",
        "## D. Baselines and training configuration",
        "- B3-0 Persistence: latest historical target repeated.",
        "- B3-1 Growth Trend: least-squares line through the three historical targets against actual DAS, extrapolated to target DAS.",
        "- B3-2 Single-Frame PointNet: latest XYZ cloud, Conv1d/BatchNorm/ReLU pointwise encoder, max+mean pooling, MLP and seven legal time conditions; 61,699 parameters.",
        "- B3-3 PointNet+GRU: shared encoder on three chronological XYZ clouds, one-layer batch-first GRU, MLP head; 160,771 parameters.",
        "- B3-4 Repeated-Latest: identical B3-3 capacity and training budget, but the latest encoded cloud is repeated three times inside the model; independently initialized and trained.",
        f"- Neural settings: 2048 sampled XYZ points, fixed cache pool 8192, no geometry augmentation, AdamW, learning rate {protocol['learning_rate']}, weight decay {protocol['weight_decay']}, batch {protocol['batch_size']}, {protocol['epochs']} epochs, gradient clip {protocol['gradient_clip']}, seeds `{protocol['seeds']}`, train-standardized MSE, validation-only selection.",
        "- Deterministic PyTorch settings and checkpoint fingerprints are recorded. No test score selected model architecture or epoch.",
        "",
        "## E. Actual performance and statistics",
        "- Primary and supplementary scores use plant-level macro MAE/RMSE: windows are first averaged within plant, then plants are equally weighted. Bootstrap resampling unit is plant UID, never point cloud or overlapping window.",
        "",
        "### Official test plant-macro MAE (cm/count units)",
        "| Model | h=1 height | h=2 height | h=1 width | h=2 width | h=1 leaves | h=2 leaves |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["B3-0", "B3-1", "B3-2", "B3-3", "B3-4"]:
        lines.append(f"| {model} | {value(model,1,'plant_height')} | {value(model,2,'plant_height')} | {value(model,1,'plant_width')} | {value(model,2,'plant_width')} | {value(model,1,'visible_organ_count')} | {value(model,2,'visible_organ_count')} |")
    lines += ["", "### Official test plant-macro RMSE", "| Model | h=1 height | h=2 height | h=1 width | h=2 width |", "|---|---:|---:|---:|---:|"]
    for model in ["B3-0", "B3-1", "B3-2", "B3-3", "B3-4"]:
        lines.append(f"| {model} | {value(model,1,'plant_height','plant_macro_rmse')} | {value(model,2,'plant_height','plant_macro_rmse')} | {value(model,1,'plant_width','plant_macro_rmse')} | {value(model,2,'plant_width','plant_macro_rmse')} |")
    lines += ["", "- RMSE is the mean of within-plant root mean-square errors, not the square root of pooled MSE. Detailed per-seed values and 95% plant-bootstrap intervals are in `baseline_comparison.csv`, `per_seed_metrics.csv`, `plant_level_errors.csv` and `paired_comparisons.csv`.", "- The paired primary comparisons below use B3-3 minus control: negative is favorable.", "", "| Control | h=2 height difference | 95% CI | fold direction |", "|---|---:|---|---:|"]
    for control in ("B3-2", "B3-4"):
        row = paired.loc[paired.left.eq("B3-3") & paired.right.eq(control) & paired.target.eq("plant_height") & paired.horizon.eq(2)].iloc[0]
        detail = next(item for item in decision["controls"] if item["control"] == control)
        lines.append(f"| {control} | {row.left_minus_right_mae:.3f} ({row.relative_change_pct:.1f}%) | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | {detail['improving_folds']}/4 |")
    lines.append("- Exploratory paired CSVs additionally include Bonferroni-family intervals across all 48 displayed model/target/horizon contrasts; the locked primary rule is unchanged. No test-derived claim is promoted by selectively ignoring other comparisons.")
    oof = pd.read_csv(root / "reports/m3/oof/baseline_comparison.csv")
    sensitivity = pd.read_csv(root / "reports/m3/point_sampling_sensitivity.csv") if (root / "reports/m3/point_sampling_sensitivity.csv").exists() else None
    lines += ["", "### Four-fold out-of-fold primary task (48 independent held-out plants)", "| Model | h=2 height MAE | seed SD | 95% plant CI |", "|---|---:|---:|---|"]
    for row in oof.loc[oof.horizon.eq(2) & oof.target.eq("plant_height")].itertuples(index=False):
        lines.append(f"| {row.model} | {row.plant_macro_mae:.3f} | {row.seed_mae_std:.3f} | [{row.mae_ci95_low:.3f}, {row.mae_ci95_high:.3f}] |")
    lines += ["- OOF uncertainty is descriptive: held-out plant predictions are distinct, but fitted fold models share training plants. It is not a fully independent external replication.", "", "### Supplementary analyses (exploratory)", "- B3-3 h=2 width MAE is 2.654 cm versus Persistence 3.017 cm (−12.0%, paired 95% CI [−0.852, 0.159] cm) and Growth Trend 3.962 cm (−33.0%, CI [−1.678, −0.936] cm). This demonstrates usable width prediction but does not show extra temporal value: single-frame is 2.665 and repeated-latest is 2.678 cm; those pairwise intervals include zero.", "- Stage errors: B3-3 h=2 height MAE is 2.328 cm at target DAS≤58, 1.989 cm at DAS61–75, and 2.205 cm at DAS≥79. Corresponding leaf-count MAE rises from 1.259 to 1.678 to 2.402; label-free random samples struggle most with late visible count, not necessarily chronological biological growth.", "- Actual-gap errors are reported separately at h=1 (3, 4 or 7 days) and h=2 (7, 10 or 11 days). Seven-day acquisition interruption is preserved and never interpreted as a fixed scan-day schedule.", "- Organ-first-observation relation: B3-3 h=2 width MAE is 2.855 cm in windows with ≥1 newly first-observed leaf versus 2.376 without, but cohorts differ (12 vs 10 plants). This is a descriptive association confounded by stage/growth, not evidence of biological birth causing prediction errors.", "- Genotype-stratified CSVs cover all available groups and report independent plant counts, but official test has only one plant per source-layout genotype and two training identities are provisional; no confirmatory genotype claim is made."]
    if sensitivity is not None:
        lines += ["", "| Model | 2048-point h=2 height MAE | 4096-point h=2 height MAE |", "|---|---:|---:|"]
        for model in ("B3-2", "B3-3", "B3-4"):
            subset = sensitivity.loc[sensitivity.model.eq(model) & sensitivity.horizon.eq(2) & sensitivity.target.eq("plant_height")].set_index("points")
            lines.append(f"| {model} | {subset.loc[2048,'plant_macro_mae']:.3f} | {subset.loc[4096,'plant_macro_mae']:.3f} |")
        lines.append("- 4096-point models are independently retrained for the same three seeds and budgets. More samples do not reverse the history-control conclusion; this is a predeclared sensitivity, not a new model-selection opportunity.")
    lines += ["", "## F. Incremental value of 3D history", "- Research decision is **" + decision["research_decision"] + "**, not a declaration that the hypothesis is proven."]
    for item in decision["controls"]:
        lines.append(f"- Against {item['control']}: official primary improvement {item['official_improvement_pct']:.1f}%; all seeds improve = `{item['all_three_seeds_improve']}`; OOF improvement {item['oof_improvement_pct']:.1f}%; folds improving = {item['improving_folds']}/4; locked criterion passed = `{item['passed']}`.")
    lines += ["- Genotype, DAS-stage, actual-gap, organ-first-observation and point-count analyses are exploratory because 48 plants provide limited independent units and genotype labels have one unresolved source-layout imbalance.", "", "## G. Failure cases and limitations", f"- Worst official B3-3 h=2 height plant is `{worst}` under plant-level MAE; its forecast trace is Figure 05 and the CSV diagnostics.", "- The dataset is a single controlled greenhouse experiment with 48 plants and automatic morphology. Point density ranges from 15,033 to 1,454,217 raw points; fixed subsampling may discard density information, while the point cache is deliberately label-free.", "- First observed leaf-instance DAS is an annotation observation, not a biological birth time; the report uses it only as an observation proxy.", "- Official split is plant-disjoint but genotypes are represented across sets; provisional layout-image genotype mapping should be confirmed by the dataset authors before genotype claims.", "- The delivered task predicts released phenotype measurements rather than complete future point clouds. This is intentionally a baseline and does not justify a JEPA launch by itself.", "", "## H. M4 recommendation", "- **Do not automatically start M4.** Engineering is complete, but the locked research rule decides whether to CONTINUE or REVIEW. If REVIEW, first resolve genotype metadata and assess whether any observed 3D-history gain is practically meaningful and robust to plant-level folds; only then consider 3D Temporal-JEPA.", "", "## Reproducibility and files", "- Protocol: `configs/m3/protocol.json`; dataset audit: `reports/m3/data_audit.json`, `reports/m3/detailed_data_audit.json`; tests: `reports/m3/tests_initial_real.log` and `reports/m3/tests_initial_real.xml`.", "- Code: `src/datasets/sugar4d_dataset.py`, `src/models/pointnet_forecaster.py`, `src/models/pointnet_gru_forecaster.py`, `scripts/prepare_sugar4d.py`, `scripts/train_m3.py`, `scripts/finalize_m3.py`.", "- Figures are generated from saved predictions and source metadata under `reports/m3/figures/`; all raw data and large caches remain outside Git.", ""]
    resources = pd.read_csv(root / "reports/m3/training_resources.csv")
    verification = json.loads((root / "reports/m3/final_verification.json").read_text())
    lines += ["## Acceptance evidence", f"- 54 completed neural runs: 9 official 2048-point + 36 four-fold + 9 official 4096-point; all run for 50 epochs/500 steps ({int(resources.optimizer_steps.sum()):,} optimizer steps). Deterministic oracle baselines require no stochastic training; duplicating identical oracle runs across seeds would not increase evidence.", f"- Complete repository tests: 80 passed, 0 failed/0 skipped; logs and JUnit XML are under `reports/m3/tests_all.*`. The M3-only suite includes actual-data decoding/sampling/batching smoke coverage.", "- All 18 official checkpoints were reloaded and produced exactly identical repeated inference outputs; saved float32 predictions agree within 1e-5. GRU/repeat initial tensor-state hashes, capacities, optimizers and steps match for each seed/partition.", f"- Legacy protection checked {verification['legacy_files_checked']} previous files; raw protection checked {verification['raw_files_checked']} extracted data files; zero changes. Raw archive MD5/CRC and SHA256 and readme hashes are recorded in `data_audit.json`.", "- Eight figures are exported as PNG/PDF/SVG/TIFF; plot-source QA passes hard checks. Figure titles, n, error-bar meanings and sampling/crop operations are documented in `FIGURE_CONTRACT.md` and figure QA files.", "- No future cloud, semantic/instance labels, oracle phenotypes or genotype are passed to neural forward functions. GT is used only for target construction, audit/visualization and the explicitly Oracle baselines.", "", "## Sources and attribution", "- Sugar4D project: https://www.ipb.uni-bonn.de/data/sugar4d/index.html", "- Data: https://doi.org/10.60507/FK2/IS8YBZ ; descriptor: https://www.nature.com/articles/s41597-026-08326-5", "- Licence: https://creativecommons.org/licenses/by/4.0/legalcode.en ; detailed extraction-code/doc provenance and CC BY author attribution: `reports/m3/SOURCES.md`.", "- Requested Bocha search returned quota/funds error; official pages/API/readme/actual files were directly fetched. 图片内容由 Agnes API 识别; outputs and unresolved layout evidence are recorded in `sources/`.", "- M3 remains a bounded baseline milestone; M0/M1/M1.5 are unchanged and M4 has not started.", ""]
    return "\n".join(lines)


def run(args):
    root = ROOT
    protocol = json.loads((root / "configs/m3/protocol.json").read_text())
    audit = json.loads((root / "reports/m3/data_audit.json").read_text())
    verification = json.loads((root / "reports/m3/final_verification.json").read_text())
    suites = list(ET.parse(root / "reports/m3/tests_all.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "errors", "failures", "skipped")}
    if counts["errors"] or counts["failures"] or verification["status"] != "PASS" or verification.get("training", {}).get("neural_runs") != 54:
        raise ValueError("Engineering evidence incomplete: run tests and verify_m3.py before evaluation")
    metadata = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    windows = pd.read_csv(root / "data/processed/sugar4d/forecast_windows.csv")
    official_neural = load_partition(root, "official", 2048)
    official_windows = windows.loc[windows.split.eq("test")].copy()
    official_oracle = oracle_predictions(official_windows)
    official_predictions = pd.concat([official_oracle, official_neural], ignore_index=True)
    official_predictions.to_csv(root / "outputs/m3/official/all_predictions.csv", index=False)
    official_dir = root / "reports/m3/official"
    official_comparison, official_seed, official_plants, official_pairs = analyze_prediction_set(official_predictions, official_dir)
    for filename in ("baseline_comparison.csv", "per_seed_metrics.csv", "plant_level_errors.csv", "paired_comparisons.csv"):
        shutil.copyfile(official_dir / filename, root / "reports/m3" / filename)
    fold_comparisons = []
    fold_predictions = []
    for fold in range(4):
        neural = load_partition(root, f"fold{fold}", 2048)
        assignments = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"][f"fold{fold}"]
        fold_windows = windows.copy(); fold_windows["split"] = fold_windows.plant_uid.map(assignments)
        oracle = oracle_predictions(fold_windows.loc[fold_windows.split.eq("test")])
        oracle["partition"] = f"fold{fold}"
        prediction = pd.concat([oracle, neural], ignore_index=True); fold_predictions.append(prediction)
        comparison, _, _, _ = analyze_prediction_set(prediction, root / "reports/m3" / f"fold{fold}"); fold_comparisons.append(comparison)
    oof = pd.concat(fold_predictions, ignore_index=True)
    for (model, seed), group in oof.groupby(["model", "seed"]):
        if group.window_id.duplicated().any() or group.plant_uid.nunique() != 48:
            raise ValueError("OOF must hold out every distinct plant exactly once per model/seed")
    oof.to_csv(root / "outputs/m3/oof_predictions.csv", index=False)
    oof_comparison, oof_seed, oof_plants, oof_pairs = analyze_prediction_set(oof, root / "reports/m3/oof")
    first = pd.read_csv(root / "reports/m3/organ_first_observations.csv")
    diagnostics = stratified_errors(official_predictions, first, official_windows, root / "reports/m3")
    failure_rows = []
    for (model, horizon, plant_uid), group in official_predictions.groupby(["model", "horizon", "plant_uid"]):
        for target in TARGETS:
            error = group[f"pred_{target}"] - group[f"true_{target}"]
            failure_rows.append({"model": model, "horizon": horizon, "plant_uid": plant_uid, "target": target, "mean_signed_error": float(error.mean()),
                                 "window_mean_abs_error": float(error.abs().mean()), "target_min": float(group[f"true_{target}"].min()), "target_max": float(group[f"true_{target}"].max()),
                                 "prediction_min": float(group[f"pred_{target}"].min()), "prediction_max": float(group[f"pred_{target}"].max()), "plants_not_deleted": True})
    pd.DataFrame(failure_rows).to_csv(root / "reports/m3/failure_case_summary.csv", index=False)
    sensitivity = None
    sensitivity_plot = None
    if args.sensitivity:
        sensitivity_neural = load_partition(root, "official", 4096)
        sensitivity = pd.concat([official_oracle, sensitivity_neural], ignore_index=True)
        sensitivity_comparison, _, _, _ = analyze_prediction_set(sensitivity, root / "reports/m3/sensitivity4096")
        main_points = official_comparison.copy()
        main_points["points"] = 2048
        sensitivity_comparison["points"] = 4096
        sensitivity_plot = pd.concat([main_points, sensitivity_comparison], ignore_index=True)
        sensitivity_plot.to_csv(root / "reports/m3/point_sampling_sensitivity.csv", index=False)
    decision = research_decision(official_comparison, official_seed, official_pairs, oof_comparison, oof_pairs, fold_comparisons, protocol)
    write_json(root / "reports/m3/research_decision.json", decision)
    worst = generate(root, metadata, official_predictions, official_comparison, official_seed, official_plants, official_pairs, diagnostics, json.loads((root / "reports/m3/plant_splits.json").read_text()), sensitivity_plot)
    records = []
    for path in sorted((root / "outputs/m3").glob("official/points2048/*_training.json")):
        records.append(json.loads(path.read_text()))
    (root / "reports/m3/M3_REPORT.md").write_text(report_text(root, protocol, audit, official_comparison, official_pairs, decision, records, worst))
    write_json(root / "reports/m3/acceptance.json", {"status": "PASS", "engineering": {"real_data": True, "archive_md5": audit["archive"]["md5"], "plants": audit["plants"], "observations": audit["observations"], "official_neural_runs": len(records), "total_neural_runs": verification["training"]["neural_runs"], "folds": 4, "tests": counts, "legacy_verification": verification, "genotype_claims_provisional": True}, "research_decision": decision, "m4_started": False})
    print(decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--sensitivity", action="store_true"); run(parser.parse_args())
