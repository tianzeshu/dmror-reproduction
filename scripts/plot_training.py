"""Plot unsmoothed full-model training histories and validation checkpoints.

Use saved histories only. Each seed stops at its own final recorded epoch; no
padding, interpolation, smoothing, test-set curves or synthetic values are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ICKG-Weak", "SIM-Auto", "SIM-Energy", "SIM-Semi")
SEEDS = (17, 29, 43, 71, 101)
COLORS = ("#376A92", "#D18B47", "#739378", "#9A79A3", "#737A86")
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_history(results: Path, dataset: str, seed: int) -> dict:
    run_dir = results / dataset / "full" / f"seed_{seed}"
    result_path, history_path = run_dir / "result.json", run_dir / "history.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (result["status"], result["dataset"], result["mode"], result["seed"]) != (
            "completed", dataset, "full", seed):
        raise ValueError(f"Unexpected result identity: {portable_path(result_path)}")
    if not history or len(history) != result["epochs_run"]:
        raise ValueError(f"Incomplete history: {portable_path(history_path)}")
    epochs = np.asarray([row["epoch"] for row in history], dtype=int)
    if not np.array_equal(epochs, np.arange(1, len(history) + 1)):
        raise ValueError(f"Epoch sequence has gaps: {portable_path(history_path)}")
    losses = np.asarray([row["train_loss"] for row in history], dtype=float)
    validation_ap = np.asarray([row["validation_auprc"] for row in history], dtype=float)
    if (not np.isfinite(losses).all() or not np.isfinite(validation_ap).all()
            or np.any(losses < 0) or np.any((validation_ap < 0) | (validation_ap > 1))):
        raise ValueError(f"Invalid history values: {portable_path(history_path)}")
    # Replay the actual checkpoint selection rule rather than assuming argmax:
    # an improvement must exceed min_delta, so a slightly higher score may not
    # replace a saved checkpoint.
    min_delta = float(result["config"]["min_delta"])
    selected_score, selected_epoch = -float("inf"), 0
    for epoch, score in zip(epochs, validation_ap):
        if float(score) > selected_score + min_delta:
            selected_score, selected_epoch = float(score), int(epoch)
    if (selected_epoch != result["best_epoch"] or not np.isclose(
            selected_score, result["best_validation_auprc"], rtol=0, atol=1e-12)):
        raise ValueError(f"Checkpoint/history mismatch: {portable_path(run_dir)}")
    if not np.isclose(result["validation"]["node"]["auprc"], selected_score,
                      rtol=0, atol=1e-12):
        raise ValueError(f"Restored checkpoint metric mismatch: {portable_path(run_dir)}")
    return {
        "dataset": dataset, "seed": seed, "history": history, "epochs": epochs,
        "loss": losses, "validation_ap": validation_ap,
        "best_epoch": selected_epoch, "best_validation_ap": selected_score,
        "epochs_run": len(history),
        "dataset_sha256": result["config"]["dataset_sha256"],
        "data_kind": result["config"]["dataset_metadata"]["data_kind"],
        "model_selection": result["config"]["model_selection"],
        "history_file": portable_path(history_path),
        "history_sha256": file_hash(history_path),
        "result_file": portable_path(result_path), "result_sha256": file_hash(result_path),
    }


def render(runs: list[dict], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 7, "axes.spines.right": False,
        "axes.spines.top": False, "axes.linewidth": 0.7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    figure, axes = plt.subplots(4, 2, figsize=(7.2, 8.2), sharex="row")
    for row_index, dataset in enumerate(DATASETS):
        dataset_runs = [run for run in runs if run["dataset"] == dataset]
        for seed_index, run in enumerate(dataset_runs):
            color, style = COLORS[seed_index], LINE_STYLES[seed_index]
            selected_index = run["best_epoch"] - 1
            for column, values in enumerate((run["loss"], run["validation_ap"])):
                axis = axes[row_index, column]
                axis.plot(run["epochs"], values, color=color, linestyle=style,
                          linewidth=1.05, alpha=0.9)
                axis.scatter(run["best_epoch"], values[selected_index], s=21,
                             marker="o", facecolors="white", edgecolors=color,
                             linewidths=0.85, zorder=4)
        max_epoch = max(run["epochs_run"] for run in dataset_runs)
        for column, axis in enumerate(axes[row_index]):
            panel = chr(ord("a") + row_index * 2 + column)
            axis.text(-0.14, 1.06, panel, transform=axis.transAxes,
                      fontsize=9, fontweight="bold", va="bottom")
            axis.set_title(dataset, loc="left", pad=8, fontweight="bold")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Training loss" if column == 0 else "Validation AP")
            axis.set_xlim(0.5, max_epoch + 0.5)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(axis="y", color="#E8E9EB", linewidth=0.5)
            axis.set_axisbelow(True)
            axis.margins(y=0.10)
            if column == 1:
                bottom, top = axis.get_ylim()
                axis.set_ylim(max(0.0, bottom), min(1.0, top))
    legend_handles = [
        Line2D([0], [0], color=COLORS[index], linestyle=LINE_STYLES[index],
               linewidth=1.2, label=f"Seed {seed}")
        for index, seed in enumerate(SEEDS)
    ]
    legend_handles.append(Line2D([0], [0], marker="o", color="#62666B",
                                 markerfacecolor="white", linewidth=0,
                                 markersize=4, label="Selected checkpoint"))
    figure.legend(handles=legend_handles, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 0.015), columnspacing=2.5, frameon=False)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.105, top=0.96,
                           hspace=0.64, wspace=0.29)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"full_model_training_histories.{suffix}", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/formal"))
    parser.add_argument("--output", type=Path, default=Path("reports/training_figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    # Contract is written before plotting to make the evidence scope explicit.
    write_json(args.output / "figure_contract.json", {
        "conclusion": "Recorded full-model training losses and validation AP trajectories show the actual training histories and validation-selected checkpoints across five seeds.",
        "archetype": "quantitative grid", "backend": "Python matplotlib",
        "panels": "Four datasets, training loss at left and validation AP at right; no test metrics are plotted.",
        "dimensions_inches": [7.2, 8.2], "png_dpi": 300,
        "exports": ["PNG", "SVG with editable text", "PDF with embedded TrueType fonts", "CSV Source Data"],
        "seeds": list(SEEDS), "model": "full",
        "statistics": "Five individual seed trajectories. No averaging, SD, confidence intervals or significance testing are shown.",
        "integrity": "No smoothing, padding, interpolation or extrapolation. Each curve ends at its actual final epoch; checkpoint markers replay validation AP selection with min_delta.",
        "scope": "Training diagnostics only. They do not establish robustness, sensitivity, causal propagation or replication of the paper's real-world datasets/results.",
        "review_risks": ["Training loss includes multiple objectives and should not be compared across different datasets as a common calibrated scale.", "Validation AP is repeatedly used for checkpoint selection, so its peak is not an unbiased test performance estimate.", "Five training seeds quantify optimisation variation, not population uncertainty."],
    })
    runs = [load_history(args.results, dataset, seed)
            for dataset in DATASETS for seed in SEEDS]
    for dataset in DATASETS:
        hashes = {run["dataset_sha256"] for run in runs if run["dataset"] == dataset}
        if len(hashes) != 1:
            raise ValueError(f"Different dataset hashes among seeds: {dataset}")
    rows = [
        {"dataset": run["dataset"], "mode": "full", "seed": run["seed"],
         "epoch": row["epoch"], "train_loss": row["train_loss"],
         "validation_ap": row["validation_auprc"],
         "validation_auc": row["validation_auc"],
         "selected_checkpoint": int(row["epoch"] == run["best_epoch"]),
         "epochs_run": run["epochs_run"], "dataset_sha256": run["dataset_sha256"],
         "history_file": run["history_file"]}
        for run in runs for row in run["history"]
    ]
    write_csv(source / "training_history.csv", rows, [
        "dataset", "mode", "seed", "epoch", "train_loss", "validation_ap",
        "validation_auc", "selected_checkpoint", "epochs_run", "dataset_sha256", "history_file",
    ])
    selections = [{key: run[key] for key in (
        "dataset", "seed", "best_epoch", "best_validation_ap", "epochs_run",
        "dataset_sha256", "data_kind", "model_selection", "history_file",
        "history_sha256", "result_file", "result_sha256")}
        for run in runs]
    write_csv(source / "checkpoint_selection.csv", selections, list(selections[0]))
    render(runs, args.output)
    write_json(args.output / "input_manifest.json", {
        "runs": selections, "n_runs": len(runs), "n_epoch_rows": len(rows),
        "checkpoint_rule_replay": "passed for all 20 full-model runs",
        "history_integrity": "epoch sequences, finite values, final epoch counts and restored validation metrics verified",
        "source_data_sha256": {
            path.name: file_hash(path) for path in sorted(source.glob("*.csv"))},
        "export_sha256": {
            path.name: file_hash(path) for path in sorted(args.output.glob("full_model_training_histories.*"))},
    })
    caption = """# 完整模型训练历史图说明

`full_model_training_histories` 的八个面板来自 20 次完整模型正式训练日志：四个数据集各运行种子 17、29、43、71、101。每行左图为训练目标的批次平均损失，右图为验证集节点 Average Precision（AP；本工程表格记作 AUPRC）。颜色和线型共同标识种子，空心圆标记按验证 AP 与 `min_delta` 规则实际选中的 checkpoint。训练达到早停条件后，各曲线在各自真实记录的最后 epoch 终止。

所有曲线直接绘制原始日志，没有平滑、插值、补齐或外推。图中不显示均值、标准差、置信区间或显著性检验；不把不等长历史补齐后求平均。五个训练种子是模型优化重复，不能当作五个独立研究样本。训练损失包含节点、对齐及可用传播监督目标，不同数据集之间的损失数值不代表统一校准的误差尺度。验证 AP 用于反复选择 checkpoint，其峰值不应作为独立测试结果。

本图展示训练与验证诊断，不是受控参数敏感性、噪声鲁棒性或真实供应链因果机制证据。SIM 数据集为仿真；ICKG-Weak 为真实文本来源的弱标签任务，不能视为真实中断金标准。正式测试指标及其五种子样本标准差请查主实验表。

Source Data：`source_data/training_history.csv` 保留每次正式训练的全部逐 epoch 数值与 checkpoint 标识；`source_data/checkpoint_selection.csv` 保留选择结果、数据文件和输入日志校验和。`input_manifest.json` 记录 20 次 checkpoint 选择规则复核及全部源数据与导出图校验和。SVG 中的文字保持可编辑，PDF 嵌入 TrueType 字体，PNG 按 7.2 × 8.2 英寸、300 dpi 导出。
"""
    (args.output / "captions.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(runs),
                      "epoch_rows": len(rows), "checkpoint_rule_replay": "passed"}))


if __name__ == "__main__":
    main()
