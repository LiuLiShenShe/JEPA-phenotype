#!/usr/bin/env python3
"""Generate all seven M1 diagnostics with transparent selection and source data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from training.common import TARGET_NAMES, write_json


MODELS = ["B0","B1","B1b","B2a","B2b"]
LABELS = {"B0":"B0 Persistence (oracle)","B1":"B1 Linear (oracle)","B1b":"B1b Time only", "B2a":"B2a DINO + MLP","B2b":"B2b DINO + GRU"}
COLORS = {"B0":"#777777","B1":"#aaaaaa","B1b":"#b4916c","B2a":"#4c78a8","B2b":"#d17c2f"}
TARGET_LABELS = {"leaf_count":"Visible leaf count", "projected_area_ratio":"Projected image-occupancy ratio"}
FIGURES = ROOT / "reports/m1/figures"
RENDERED_QA = []


def save_figure(figure,name):
    axes_qa = []
    for axis in figure.axes:
        lower,upper = axis.get_ylim()
        values = [np.asarray(line.get_ydata(),dtype=float) for line in axis.lines if len(line.get_ydata())]
        outside = [float(value) for line_values in values for value in line_values if np.isfinite(value) and not lower-1e-10 <= value <= upper+1e-10]
        if outside:
            raise ValueError(f"Rendered data exceed axis bounds in {name}: {outside[:5]}")
        axes_qa.append({"title":axis.get_title(),"ylim":[float(lower),float(upper)],"data_outside_axis":len(outside)})
    RENDERED_QA.append({"figure":name,"status":"PASS","axes":axes_qa})
    figure.savefig(FIGURES / f"{name}.png",dpi=300,bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf",bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.svg",bbox_inches="tight")
    plt.close(figure)


def load_predictions():
    all_rows = []
    for path in sorted((ROOT / "outputs/m1/predictions").glob("*/test.csv")):
        all_rows.append(pd.read_csv(path))
    return pd.concat(all_rows,ignore_index=True)


def curve_figure(predictions,plant_uids,name,horizon=4):
    frames = pd.read_csv(ROOT / "data/metadata/frames.csv")
    starts = pd.to_datetime(frames.timestamp).groupby(frames.plant_uid).min()
    fig,axes = plt.subplots(len(plant_uids),2,figsize=(7.2,2*len(plant_uids)),squeeze=False,constrained_layout=True)
    for plant_position,plant_uid in enumerate(plant_uids):
        group = predictions.loc[predictions.plant_uid.eq(plant_uid)&predictions.forecast_horizon.eq(horizon)]
        for target_position,target in enumerate(TARGET_NAMES):
            axis = axes[plant_position,target_position]
            truth = group.groupby("target_timestamp")[f"true_{target}"].first().sort_index()
            elapsed = (pd.to_datetime(truth.index)-starts.loc[plant_uid]).total_seconds()/86400
            axis.plot(elapsed,truth.values,color="#222222",linewidth=1.6,label="Ground truth")
            for model in MODELS:
                values = group.loc[group.model.eq(model)].groupby("target_timestamp")[f"pred_{target}"].mean().sort_index()
                axis.plot((pd.to_datetime(values.index)-starts.loc[plant_uid]).total_seconds()/86400,values,
                          color=COLORS[model],linewidth=1,label=LABELS[model],linestyle="--" if model in {"B0","B1"} else "-")
            axis.set_title(f"{plant_uid} | +{horizon} observations",fontsize=8)
            axis.set_ylabel(TARGET_LABELS[target])
            axis.set_xlabel("Days since first recorded observation (not sowing age)")
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=7)
    save_figure(fig,name)


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":7,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.8,"legend.frameon":False,
                         "svg.fonttype":"none","pdf.fonttype":42})
    comparison = pd.read_csv(ROOT / "reports/m1/baseline_comparison.csv")
    plants = pd.read_csv(ROOT / "reports/m1/plant_level_errors.csv")
    predictions = load_predictions()
    for target in TARGET_NAMES:
        fig,axes = plt.subplots(1,3,figsize=(7.2,2.7),sharey=True,constrained_layout=True)
        upper_limit = float(comparison.loc[comparison.target.eq(target),"plant_macro_mae_ci95_high"].max())*1.18
        for axis,horizon in zip(axes,(2,4,8)):
            group = comparison.loc[comparison.target.eq(target)&comparison.horizon.eq(horizon)].set_index("model").loc[MODELS]
            means = group.plant_macro_mae_mean.to_numpy()
            errors = np.vstack([means-group.plant_macro_mae_ci95_low,group.plant_macro_mae_ci95_high-means])
            axis.bar(np.arange(5),means,yerr=errors,color=[COLORS[model] for model in MODELS],capsize=2)
            axis.set_xticks(np.arange(5),["B0*","B1*","B1b","B2a","B2b"],rotation=45)
            axis.set_title(f"+{horizon} original observations")
            axis.set_xlabel("*Oracle historical phenotypes")
            axis.set_ylim(0,upper_limit)
        axes[0].set_ylabel(f"Plant-macro MAE: {TARGET_LABELS[target]}")
        fig.suptitle("46 held-out plants | neural mean across 3 seeds | 95% plant-bootstrap CI",fontsize=8)
        save_figure(fig,f"baseline_mae_{target}")
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.7),constrained_layout=True)
    for axis,target in zip(axes,TARGET_NAMES):
        for model in MODELS:
            group = comparison.loc[comparison.target.eq(target)&comparison.model.eq(model)].sort_values("horizon")
            axis.plot(group.horizon,group.plant_macro_mae_mean,"o-",color=COLORS[model],label=LABELS[model],markersize=3)
        axis.set_xlabel("Original observation offset (not days)")
        axis.set_ylabel(f"Plant-macro MAE: {TARGET_LABELS[target]}")
        axis.set_xticks([2,4,8])
    axes[1].legend(fontsize=7)
    save_figure(fig,"horizon_comparison")
    test_plants = sorted(predictions.plant_uid.unique(),key=lambda plant:hashlib.sha256(f"42:{plant}".encode()).hexdigest())
    selected = test_plants[:6]
    curve_figure(predictions,selected,"growth_curves_preselected")
    independent = plants.groupby(["model","horizon","target","plant_uid","genotype"],as_index=False).mae.mean()
    failures = independent.loc[independent.model.eq("B2b")&independent.horizon.eq(8)].pivot(index=["plant_uid","genotype"],columns="target",values="mae").reset_index()
    train_rows = pd.read_csv(ROOT / "data/processed/temporal_windows.csv")
    scales = train_rows.loc[train_rows.split.eq("train")&train_rows.forecast_horizon.eq(8),["target_leaf_count","target_projected_area_ratio"]].std(ddof=0).to_numpy()
    failures["normalized_mean_mae"] = (failures.leaf_count/scales[0]+failures.projected_area_ratio/scales[1])/2
    failures = failures.sort_values("normalized_mean_mae",ascending=False)
    failures.to_csv(ROOT / "reports/m1/failure_cases.csv",index=False)
    curve_figure(predictions,failures.plant_uid.iloc[:3].tolist(),"growth_curves_failure_cases",horizon=8)
    fig,axes = plt.subplots(2,1,figsize=(7.2,5.2),constrained_layout=True)
    genotypes = sorted(independent.genotype.unique())
    for axis,target in zip(axes,TARGET_NAMES):
        for model,shift in (("B2a",-.18),("B2b",.18)):
            data = [independent.loc[independent.target.eq(target)&independent.horizon.eq(4)&independent.model.eq(model)&independent.genotype.eq(genotype),"mae"].to_numpy() for genotype in genotypes]
            boxes = axis.boxplot(data,positions=np.arange(len(genotypes))+shift,widths=.3,patch_artist=True,manage_ticks=False,
                                 flierprops={"markersize":2},medianprops={"color":"#222222"})
            for patch in boxes["boxes"]:
                patch.set_facecolor(COLORS[model])
                patch.set_alpha(.7)
        axis.set_xticks(np.arange(len(genotypes)),genotypes,rotation=60)
        axis.set_ylabel(f"+4 plant MAE: {TARGET_LABELS[target]}")
    axes[0].plot([],[],color=COLORS["B2a"],label=LABELS["B2a"])
    axes[0].plot([],[],color=COLORS["B2b"],label=LABELS["B2b"])
    axes[0].legend()
    save_figure(fig,"genotype_error_distributions")
    fig,axes = plt.subplots(2,3,figsize=(7.2,4.5),constrained_layout=True)
    for model_position,model in enumerate(("B2a","B2b")):
        for horizon_position,horizon in enumerate((2,4,8)):
            axis = axes[model_position,horizon_position]
            for seed in (42,43,44):
                history = pd.read_csv(ROOT / f"outputs/m1/checkpoints/{model}_h{horizon}_s{seed}/training_history.csv")
                axis.plot(history.epoch,history.train_weighted_huber,color=COLORS[model],alpha=.4,linewidth=.8)
                axis.plot(history.epoch,history.val_huber,color="#777777",alpha=.6,linestyle="--",linewidth=.8)
            axis.set_title(f"{model} | +{horizon} | 3 seeds")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Standardized Huber loss")
    axes[0,0].plot([],[],color=COLORS["B2a"],label="Train (plant weighted)")
    axes[0,0].plot([],[],color="#777777",linestyle="--",label="Val (window mean)")
    axes[0,0].legend()
    save_figure(fig,"training_validation_losses")
    temporal_sources = []
    fig,axes = plt.subplots(1,2,figsize=(7.2,3),constrained_layout=True)
    for axis,target in zip(axes,TARGET_NAMES):
        upper_values = []
        for model in MODELS:
            model_rows = predictions.loc[predictions.model.eq(model)].copy()
            model_rows["time_bin_hours"] = np.floor(model_rows.target_time_delta_seconds/21600)*6
            within = model_rows.groupby(["plant_uid","time_bin_hours"])[f"abs_error_{target}"].mean().reset_index()
            grouped = within.groupby("time_bin_hours")[f"abs_error_{target}"].agg(["mean","count"]).reset_index()
            grouped["model"] = model
            grouped["target"] = target
            temporal_sources.append(grouped)
            upper_values.append(float(grouped["mean"].max()))
            axis.plot(grouped.time_bin_hours/24,grouped["mean"],"o-",color=COLORS[model],label=LABELS[model],markersize=2)
        axis.set_xlabel("Actual forecast wall-clock interval (days; 6 h bins)")
        axis.set_ylabel(f"Within-bin plant-macro MAE: {TARGET_LABELS[target]}")
        axis.set_ylim(0,max(upper_values)*1.18)
    axes[1].legend(fontsize=7)
    save_figure(fig,"error_vs_actual_time")
    pd.concat(temporal_sources,ignore_index=True).to_csv(ROOT / "reports/m1/error_vs_time_source.csv",index=False)
    write_json(FIGURES / "selection_manifest.json",{
        "preselected_plants":selected,"rule":"first six SHA256(42:plant_uid); independent of prediction quality",
        "failure_plants":failures.plant_uid.iloc[:3].tolist(),"failure_rule":"worst +8 seed-averaged GRU target-normalized plant MAE; curves also displayed at +8",
        "curve_seed_aggregation":"average predictions for display, not the main metric",
        "ci_definition":"95% percentile bootstrap of 46 plants after averaging per-seed errors; 2000 replicates",
        "error_vs_time":"6-hour bins, seed/window errors averaged within each plant/bin then across plants; bins with small count remain visible",
        "source_data":["baseline_comparison.csv","plant_level_errors.csv","per_seed_metrics.csv","error_vs_time_source.csv","failure_cases.csv"],
        "figures":[path.name for path in sorted(FIGURES.glob("*.pdf"))],
    })
    write_json(ROOT / "reports/m1/figure_rendered_qa.json",{"status":"PASS","figures":RENDERED_QA,
               "check":"all plotted line/errorbar values within axes; comparison/time plots include 18% upper margin"})


if __name__ == "__main__":
    main()
