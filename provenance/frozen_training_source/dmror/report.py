"""Aggregate recorded run metrics; no paper values or placeholders are inserted."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD_ORDER = ("full", "text_mlp", "graph_only", "concat", "no_stm", "no_ltm", "no_time",
                "no_resgate", "no_overload", "avg_fusion", "scalar_gate", "no_align", "no_path")
PAPER_ABLATIONS = ("full", "no_stm", "no_ltm", "no_time", "no_resgate", "no_overload",
                   "avg_fusion", "scalar_gate")
METRICS = ("node.auc", "node.auprc", "node.macro_f1", "node.positive_f1", "node.recall_at_k",
           "scope.precision_at_k", "scope.recall_at_k", "scope.ndcg_at_k", "scope.jaccard", "scope.jaccard_at_k",
           "edge.auc", "edge.auprc", "edge.positive_f1", "path.path_precision",
           "path.path_recall", "path.path_f1", "path.path_at_k", "path.path_edge_precision", "path.path_edge_recall")


def nested(obj, key):
    for part in key.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def stats(values):
    values = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)])
    return {"n": int(len(values)), "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def dump_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(results):
    runs, failures, seen = [], [], set()
    for path in sorted(Path(results).rglob("result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or not record.get("test"):
            failures.append({"run": str(path.relative_to(results)), "status": record.get("status")})
            continue
        identity = record["dataset"], record["mode"], record["seed"]
        if identity in seen:
            raise ValueError(f"Duplicate dataset/method/seed: {identity}")
        seen.add(identity)
        record["_run_dir"] = path.parent
        runs.append(record)
    for path in sorted(Path(results).rglob("failure.json")):
        if not (path.parent / "result.json").exists():
            failures.append({"run": str(path.relative_to(results)), "status": "failed",
                             "error": json.loads(path.read_text(encoding="utf-8")).get("error")})
    return runs, failures


def aggregate(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["dataset"], run["mode"]].append(run)
    output = []
    for (dataset, mode), records in sorted(groups.items()):
        # Mixing hash, LLM, original or rewritten datasets would invalidate a
        # mean even if their folder names happen to agree.
        hashes = {row["config"].get("dataset_sha256") for row in records}
        if len(hashes) != 1:
            raise ValueError(f"Mixed dataset hashes in {dataset}/{mode}; separate result roots")
        row = {"dataset": dataset, "mode": mode, "n_seeds": len(records),
               "seeds": sorted(run["seed"] for run in records), "dataset_sha256": next(iter(hashes)),
               "data_kind": records[0]["config"].get("dataset_metadata", {}).get("data_kind", "unspecified"),
               "label_kind": records[0]["config"].get("dataset_metadata", {}).get("label_kind", "unspecified"),
               "text_encoder": records[0]["config"].get("dataset_metadata", {}).get("text_encoder", "unspecified"),
               "metrics": {metric: stats([nested(run["test"], metric) for run in records]) for metric in METRICS},
               "runtime_seconds": stats([run.get("runtime_seconds") for run in records])}
        output.append(row)
    return output


def paired_differences(runs):
    groups = {(run["dataset"], run["mode"], run["seed"]): run for run in runs}
    differences = defaultdict(list)
    for (dataset, mode, seed), record in groups.items():
        full = groups.get((dataset, "full", seed))
        if full is None or mode == "full":
            continue
        if full["config"].get("dataset_sha256") != record["config"].get("dataset_sha256"):
            raise ValueError("Paired difference attempted across different datasets")
        for metric in METRICS:
            a, b = nested(full["test"], metric), nested(record["test"], metric)
            if a is not None and b is not None:
                differences[dataset, mode, metric].append(float(a) - float(b))
    return [{"dataset": dataset, "comparator": mode, "metric": metric, "full_minus_comparator": stats(values)}
            for (dataset, mode, metric), values in sorted(differences.items())]


def display_stat(stat):
    if stat["mean"] is None:
        return "NA"
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}" if stat["std"] is not None else f"{stat['mean']:.4f} (n=1)"


def encoder_description(value):
    if isinstance(value, dict):
        return f"{value.get('model_identifier', value.get('encoder'))}，冻结编码，{value.get('projection_dim', '?')} 维缓存；完整权重与缓存摘要见 metadata.json"
    return str(value)


def plot(summary, runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 0.8, "legend.frameon": False})
    figures = Path(output) / "figures"
    figures.mkdir(exist_ok=True)
    source_rows = []
    def save(fig, name):
        for extension in ("svg", "pdf", "png"):
            fig.savefig(figures / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    datasets = sorted({row["dataset"] for row in summary})
    for dataset in datasets:
        rows = {row["mode"]: row for row in summary if row["dataset"] == dataset}
        for figure, methods, metrics in (
                ("node_comparison", METHOD_ORDER, ("node.auprc", "node.macro_f1")),
                ("paper_ablation", PAPER_ABLATIONS, ("node.auprc", "scope.recall_at_k")),
                ("path_scope", METHOD_ORDER, ("scope.ndcg_at_k", "path.path_f1"))):
            methods = [method for method in methods if method in rows]
            available = [metric for metric in metrics if any(rows[method]["metrics"][metric]["mean"] is not None for method in methods)]
            if not available:
                continue
            fig, axes = plt.subplots(1, len(available), figsize=(7.2, max(2.7, len(methods) * 0.25)), squeeze=False)
            for axis, metric in zip(axes[0], available):
                valid = [method for method in methods if rows[method]["metrics"][metric]["mean"] is not None]
                positions = np.arange(len(valid))
                means = [rows[method]["metrics"][metric]["mean"] for method in valid]
                # A single seed has no SD; matplotlib only receives zero error
                # for drawing, while CSV/report state undefined SD explicitly.
                deviations = [rows[method]["metrics"][metric]["std"] or 0.0 for method in valid]
                axis.errorbar(means, positions, xerr=deviations, fmt="none", color="#617588", capsize=2, linewidth=1)
                for position, method in zip(positions, valid):
                    color = "#2166AC" if method == "full" else "#9AAFC1"
                    axis.scatter(rows[method]["metrics"][metric]["mean"], position, color=color, s=28, zorder=3)
                    values = [nested(run["test"], metric) for run in runs if run["dataset"] == dataset and run["mode"] == method]
                    values = [value for value in values if value is not None]
                    axis.scatter(values, position + np.linspace(-0.065, 0.065, len(values)), color=color, s=8, alpha=.45, zorder=2)
                    source_rows.append({"dataset": dataset, "figure": figure, "metric": metric, "method": method,
                                        **rows[method]["metrics"][metric]})
                axis.set(yticks=positions, yticklabels=valid, xlabel=metric, xlim=(-.02, 1.02))
                axis.invert_yaxis()
                axis.grid(axis="x", linewidth=.4, alpha=.3)
            kind = next(iter(rows.values()))["data_kind"]
            fig.suptitle(f"{dataset} | {kind}; mean ± sample SD across training seeds", fontsize=9)
            fig.tight_layout()
            save(fig, dataset + "_" + figure)
    dump_csv(figures / "source_data.csv", source_rows, ["dataset", "figure", "metric", "method", "n", "mean", "std", "min", "max"])
    (figures / "figure_contract.json").write_text(json.dumps({
        "conclusion": "Display actual held-out method performance separately for each explicit dataset provenance; no superiority claim is assumed.",
        "evidence": {"node_comparison": "node ranking and classification", "paper_ablation": "seven architecture removals", "path_scope": "node scope and labelled-edge-induced path recovery"},
        "archetype": "quantitative grid", "backend": "Python matplotlib", "dimensions": "7.2 inch width; height depends on method count",
        "statistics": "training-seed mean and sample SD (ddof=1), with seed-level points; SD is undefined for n=1",
        "exports": ["editable text SVG", "TrueType PDF", "300dpi PNG", "source_data.csv"],
        "integrity": "No manuscript benchmark values inserted. Unknown metrics omitted. Simulator truth and weak public labels are not causal validation."
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_report(results, output=None, figures=True):
    results = Path(results).resolve()
    output = Path(output).resolve() if output else results / "report"
    output.mkdir(parents=True, exist_ok=True)
    runs, failures = collect(results)
    if not runs:
        raise ValueError("No completed result.json runs; run training before reporting")
    summary = aggregate(runs)
    paired = paired_differences(runs)
    payload = {"aggregation": "test metrics grouped by identical dataset hash and method; sample SD across seeds",
               "summary": summary, "paired_differences": paired, "incomplete_runs": failures}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    per_seed = [{"dataset": run["dataset"], "mode": run["mode"], "seed": run["seed"],
                 "best_epoch": run["best_epoch"], "runtime_seconds": run["runtime_seconds"],
                 **{metric: nested(run["test"], metric) for metric in METRICS}} for run in runs]
    dump_csv(output / "per_seed.csv", per_seed, ["dataset", "mode", "seed", "best_epoch", "runtime_seconds", *METRICS])
    flat_summary = [{"dataset": row["dataset"], "mode": row["mode"], "n_seeds": row["n_seeds"],
                     "data_kind": row["data_kind"], "text_encoder": row["text_encoder"],
                     **{metric + "_" + statistic: row["metrics"][metric][statistic]
                        for metric in METRICS for statistic in ("mean", "std", "n")}} for row in summary]
    dump_csv(output / "summary.csv", flat_summary, list(flat_summary[0]))
    lines = ["# 实验结果（实际运行）", "", "结果来自保存的测试集预测与 result.json；下表为训练随机种子的均值 ± 样本标准差。",
             "单次运行无法估计标准差；NA 表示缺少对应监督或无法计算，未填入论文数值。",
             "", "所有方法使用同一按时间排序且清除预测窗口重叠的划分。检查点按验证 AUPRC 选择；分类阈值只在验证集调节。",
             "模拟数据只能检查机制实现与可重复性。公开弱标签数据的指标是代理任务表现，无法验证真实供应链因果传播。",
             "重叠的预测窗口造成样本相关；种子标准差不是人群置信区间。", "",
             "text_mlp、graph_only、concat 是独立分类基线。它们的边/路径评分使用节点概率与依赖权重的启发式，未训练传播模型；不能作为论文命名基线的原实现。", ""]
    for dataset in sorted({row["dataset"] for row in summary}):
        rows = {row["mode"]: row for row in summary if row["dataset"] == dataset}
        sample = next(iter(rows.values()))
        lines.extend([f"## {dataset}", "", f"数据性质：{sample['data_kind']}；标签来源：{sample['label_kind']}；文本编码：{encoder_description(sample['text_encoder'])}。", "",
                      "| 方法 | 种子数 | AUC | AUPRC | Macro F1 | Recall@k | Scope NDCG@k | Path F1 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for mode in METHOD_ORDER:
            if mode in rows:
                row = rows[mode]
                fields = ["node.auc", "node.auprc", "node.macro_f1", "node.recall_at_k", "scope.ndcg_at_k", "path.path_f1"]
                lines.append("| " + " | ".join([mode, str(row["n_seeds"]), *[display_stat(row["metrics"][metric]) for metric in fields]]) + " |")
        lines.append("")
    if failures:
        lines.extend(["## 未完成的运行", "", *[f"- {row['run']}: {row['status']}" for row in failures], ""])
    lines.extend(["## 复核文件", "", "per_seed.csv 保存每个随机种子的指标；summary.csv/summary.json 保存汇总与有效样本数。",
                  "各运行目录保存检查点、训练历史、验证/测试概率、阈值、环境版本、数据 SHA-256 和参数。",
                  "figures/source_data.csv 对应图中统计值；SVG/PDF 文字可编辑，PNG 可直接预览。", ""])
    (output / "experiment_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    if figures:
        plot(summary, runs, output)
    print(f"report: {output} ({len(runs)} completed runs)")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    generate_report(args.results, args.output, not args.no_figures)


if __name__ == "__main__":
    main()
