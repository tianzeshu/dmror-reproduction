"""Report exactly 60 frozen-protocol Table-1-scale synthetic runs.

This reporter refuses partial, duplicate, stale or mixed-provenance results.
It reads recorded measurements; no manuscript benchmark values are inserted.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmror.report import (METRICS, aggregate, collect, dump_csv, nested,
                          paired_differences, stats)


DATASETS = ("SC-Auto-Synthetic", "SC-Semi-Synthetic", "SC-Energy-Synthetic")
MODES = ("full", "text_mlp", "graph_only", "concat")
SEEDS = (17, 29, 43, 71, 101)
EXPECTED = {
    "SC-Auto-Synthetic": (6482, 1126, 438, 76, 42, 58734, 124680, 38912, 8426),
    "SC-Semi-Synthetic": (4935, 842, 316, 54, 37, 46218, 96440, 31705, 6913),
    "SC-Energy-Synthetic": (5714, 973, 512, 68, 51, 52906, 108375, 34286, 7584),
}
STAT_FIELDS = ("firms", "products", "materials", "industries", "regions",
               "relations", "texts", "risk_signals", "risk_labels")
TYPE_NAMES = ("firm", "product", "material", "industry", "region")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def iso_day(day):
    return (date(1970, 1, 1) + timedelta(days=int(day))).isoformat()


def evidence(data_root, audit_path):
    audited = read_json(audit_path)
    require(audited.get("status") == "passed", "Independent actual-record audit must have passed")
    by_name = {item["name"]: item for item in audited["datasets"]}
    require(set(by_name) == set(DATASETS), "Actual-record audit must contain exactly the three new datasets")
    passport, statistic_rows = {}, []
    for name in DATASETS:
        directory = data_root / name
        audit = by_name[name]
        require(audit.get("status") == "passed" and not audit.get("error_counts"), f"Failed audit: {name}")
        # The audit parsed every JSONL record independently. These comparisons
        # prove that its findings still refer to the files being reported.
        for filename, recorded in audit["files"].items():
            path = directory / filename
            require(path.is_file(), f"Missing audited material: {name}/{filename}")
            require(path.stat().st_size == recorded["bytes"] and digest(path) == recorded["sha256"],
                    f"Material changed since independent audit: {name}/{filename}")
        metadata = read_json(directory / "metadata.json")
        require(metadata.get("is_real_world") is False and metadata.get("data_kind") == "synthetic",
                f"Dataset must explicitly be synthetic: {name}")
        require(metadata.get("table1_size_matched") is True, f"Reduced test fixture cannot be reported: {name}")
        require(metadata.get("retrieval_size") == 5 and metadata.get("signal_dim") == 64,
                f"Expected K=5, D=64: {name}")
        require(metadata.get("text_encoder") == "sha256_signed_hash_64_diagnostic_only",
                f"Unexpected encoder: {name}")
        with np.load(directory / "dataset.npz", allow_pickle=False) as data:
            node_type, labels, split = data["node_type"], data["labels"], data["split"]
            counts = np.bincount(node_type, minlength=5).astype(int).tolist()
            actual = [*counts, len(data["src"]), audit["actual_record_counts"]["texts"],
                      audit["actual_record_counts"]["signals"], int((labels >= 0).sum())]
            require(tuple(actual) == EXPECTED[name], f"Requested Table 1 counts disagree: {name}: {actual}")
            require(labels.shape == (96, len(node_type)), f"Expected 96 monthly queries: {name}")
            annotation_splits = {}
            for code, key in ((0, "train"), (1, "validation"), (2, "test"), (-1, "purged")):
                subset = labels[split == code]
                annotation_splits[key] = {"queries": int((split == code).sum()),
                    "known": int((subset >= 0).sum()), "positive": int((subset == 1).sum()),
                    "negative": int((subset == 0).sum()), "unknown": int((subset < 0).sum())}
            prediction_start, prediction_end = map(int, (data["prediction_days"].min(), data["prediction_days"].max()))
            last_target = int(data["target_days"].max())
        row = {"dataset": name, **dict(zip(STAT_FIELDS, actual)), "nodes": sum(counts),
               "time_span": "2018--2025", "data_kind": "synthetic", "is_real_world": False}
        statistic_rows.append(row)
        passport[name] = {"data_kind": "synthetic", "is_original_sc_recovery": False,
            "generator": metadata["generator"], "generator_seed": metadata["seed"],
            "statistics": row, "annotation_count_definition": metadata["risk_label_count_definition"],
            "annotation_sampling": metadata["label_sampling"], "annotation_splits": annotation_splits,
            "type_coverage": audit["source_summary"], "simulation_truth": audit["npz"].get("simulation"),
            "memory": {"K": 5, "D": 64, "encoder": metadata["text_encoder"],
                       "window_days": metadata["memory_window_days"],
                       "indices_reference": metadata["memory_indices_reference"]},
            "query_start": iso_day(prediction_start), "query_end": iso_day(prediction_end),
            "last_target": iso_day(last_target), "forecast_horizon_days": 30,
            "source_mask_rule": metadata["source_mask_rule"], "file_manifest": audit["files"],
            "dataset_sha256": audit["files"]["dataset.npz"]["sha256"],
            "independent_audit": {"status": "passed", "report_sha256": digest(audit_path)},
            "limitations": metadata["limitations"]}
    return passport, statistic_rows


def validate_runs(results, protocol_path, materials):
    protocol = read_json(protocol_path)
    require(protocol.get("stage") == "table_scale_synthetic_fixed_protocol", "Wrong experiment protocol")
    expected = {(dataset, mode, seed) for dataset in DATASETS for mode in MODES for seed in SEEDS}
    jobs = {(Path(job["dataset"]).parent.name, job["parameters"]["mode"], job["parameters"]["seed"])
            for job in protocol["jobs"]}
    require(jobs == expected and len(protocol["jobs"]) == 60, "Protocol must specify exactly the required 60 jobs")
    protocol_hash = digest(protocol_path)
    source = {path.name: digest(path) for path in sorted((ROOT / "dmror").glob("*.py"))}
    runs, failures = collect(results)
    require(not failures, f"Incomplete or failed runs cannot be aggregated: {failures[:8]}")
    actual = {(run["dataset"], run["mode"], run["seed"]) for run in runs}
    require(actual == expected and len(runs) == 60,
            f"Need exactly 60 completed new-scale runs; found {len(runs)}. Missing: {sorted(expected - actual)[:10]}; unexpected: {sorted(actual - expected)[:10]}")
    manifests = []
    shared_settings = dict(protocol["common"], weight_decay=.0001, grad_clip=5.0, lambda_reg=0.0)
    for run in runs:
        identity = f"{run['dataset']}/{run['mode']}/seed_{run['seed']}"
        directory = run["_run_dir"]
        require(directory.relative_to(results).as_posix() == identity, f"Pilot or unexpected directory: {directory}")
        config = run["config"]
        require(config["mode"] == run["mode"] and config["seed"] == run["seed"], f"Identity/config mismatch: {identity}")
        require(config.get("protocol_sha256") == protocol_hash, f"Mixed or stale protocol: {identity}")
        require(config.get("implementation_sha256") == source, f"Mixed or stale DM-ROR source: {identity}")
        require(config.get("dataset_sha256") == materials[run["dataset"]]["dataset_sha256"], f"Mixed or stale dataset: {identity}")
        require(config.get("dataset_metadata", {}).get("name") == run["dataset"], f"Metadata identity mismatch: {identity}")
        for key, expected_value in shared_settings.items():
            require(config.get(key) == expected_value, f"Protocol parameter {key} differs: {identity}")
        require(config.get("model_selection") == "validation AUPRC", f"Non-validation checkpoint selection: {identity}")
        require(config.get("threshold_selection") == "validation macro F1 for nodes; positive F1 for edges",
                f"Non-validation threshold selection: {identity}")
        unhashed = {key: value for key, value in config.items() if key != "configuration_sha256"}
        require(config.get("configuration_sha256") == canonical_hash(unhashed), f"Recorded config hash mismatch: {identity}")
        require(read_json(directory / "config.json") == config, f"Config file/result disagreement: {identity}")
        require(1 <= run["best_epoch"] <= run["epochs_run"] <= shared_settings["epochs"], f"Invalid epoch accounting: {identity}")
        files = ("result.json", "config.json", "history.json", "checkpoint.pt", "predictions.npz", "scaler.npz")
        for filename in files:
            require((directory / filename).is_file(), f"Missing completed-run artifact: {identity}/{filename}")
        require(np.isfinite(run["runtime_seconds"]) and run["runtime_seconds"] > 0, f"Invalid runtime: {identity}")
        for metric in METRICS:
            value = nested(run["test"], metric)
            require(value is None or np.isfinite(value), f"Nonfinite measured metric {metric}: {identity}")
        manifests.append({"run": identity, "dataset_sha256": config["dataset_sha256"],
                          "configuration_sha256": config["configuration_sha256"],
                          "artifacts": {filename: digest(directory / filename) for filename in files}})
    return runs, {"status": "passed", "complete_runs": 60, "expected_datasets": list(DATASETS),
        "expected_modes": list(MODES), "expected_seeds": list(SEEDS), "protocol_sha256": protocol_hash,
        "implementation_sha256": source, "implementation_map_sha256": canonical_hash(source),
        "common_settings": shared_settings, "run_artifact_manifest": manifests}


def display(statistic):
    if statistic["mean"] is None:
        return "NA"
    value = f"{statistic['mean']:.4f}"
    if statistic["std"] is not None:
        value += f" ± {statistic['std']:.4f}"
    if statistic["n"] != 5:
        value += f" (n={statistic['n']})"
    return value


def report(results, output, data_root, protocol_path, audit_path):
    materials, dataset_rows = evidence(data_root, audit_path)
    runs, validation = validate_runs(results, protocol_path, materials)
    summary = aggregate(runs)
    for row in summary:
        selected = [run for run in runs if run["dataset"] == row["dataset"] and run["mode"] == row["mode"]]
        row["epochs_run"] = stats([run["epochs_run"] for run in selected])
        row["best_epoch"] = stats([run["best_epoch"] for run in selected])
        row["cuda_peak_allocated_gib"] = stats([run.get("cuda_peak_allocated_bytes") / 2 ** 30
            if run.get("cuda_peak_allocated_bytes") is not None else None for run in selected])
    payload = {"aggregation": "Same dataset/protocol/source hashes; five optimizer seeds per method; sample SD ddof=1",
               "summary": summary, "paired_differences": paired_differences(runs), "incomplete_runs": [],
               "experiment_validation": validation, "material_passport": materials,
               "interpretation": "Explicitly synthetic fixed graphs and hash text features; no original SC results or population confidence intervals"}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", payload)
    write_json(output / "material_passport.json", {"status": "passed", "data": materials,
        "experiment": validation, "hardware": [run["config"].get("environment", {}) for run in runs[:1]]})
    per_seed = [{"dataset": run["dataset"], "mode": run["mode"], "seed": run["seed"],
        "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
        "runtime_seconds": run["runtime_seconds"], "cuda_peak_allocated_bytes": run.get("cuda_peak_allocated_bytes"),
        "dataset_sha256": run["config"]["dataset_sha256"], "protocol_sha256": validation["protocol_sha256"],
        "implementation_map_sha256": validation["implementation_map_sha256"],
        **{metric: nested(run["test"], metric) for metric in METRICS}} for run in runs]
    dump_csv(output / "per_seed.csv", per_seed, list(per_seed[0]))
    flat = [{"dataset": row["dataset"], "mode": row["mode"], "n_seeds": row["n_seeds"],
        "dataset_sha256": row["dataset_sha256"], "data_kind": row["data_kind"], "text_encoder": row["text_encoder"],
        **{metric + "_" + key: row["metrics"][metric][key] for metric in METRICS for key in ("mean", "std", "n")},
        **{metric + "_" + key: row[metric][key] for metric in ("runtime_seconds", "epochs_run", "best_epoch", "cuda_peak_allocated_gib")
           for key in ("mean", "std", "min", "max", "n")}} for row in summary]
    dump_csv(output / "summary.csv", flat, list(flat[0]))
    dump_csv(output / "dataset_statistics.csv", dataset_rows, list(dataset_rows[0]))
    table = [r"\begin{table*}[!tbp]", r"\centering",
        r"\caption{Statistics of explicitly synthetic supply-chain risk datasets constructed to match the requested Table 1 aggregate counts. These are generated engineering datasets, not recovered original observations.}",
        r"\label{tab:table_scale_synthetic_statistics}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrl}", r"\toprule",
        r"Dataset & \#Firms & \#Products & \#Materials & \#Industries & \#Regions & \#Relations & \#Texts & \#Risk Signals & \#Risk Labels & Time Span \\", r"\midrule"]
    table.extend(" & ".join([row["dataset"], *[f"{row[key]:,}" for key in STAT_FIELDS], "2018--2025"]) + r" \\" for row in dataset_rows)
    table.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}",
                  "% Risk Labels counts all known 0/1 entity-query annotations; unannotated pairs are -1.", ""])
    (output / "dataset_statistics.tex").write_text("\n".join(table), encoding="utf-8")
    lines = ["# 表 1 规模合成数据实验报告", "", "本报告来自新建三套合成数据上的 60 次完整实际运行：3 套数据 × 4 个方法 × 5 个训练种子。",
        "全部结果经过数据、DM-ROR 源码与固定实验协议 SHA-256 一致性检查；未混入旧 SIM、ICKG 或 pilot 结果。",
        "原始 SC-Auto/SC-Semi/SC-Energy 数据及论文原始实验结果仍未找回。匹配表 1 的数量不能证明原始分布、拓扑或论文指标已复现。", "",
        "## 数据材料与计数", "", "五类实体、唯一有向异构关系、原创合成文本、风险信号、有限标注及仿真事件均有实际文件。独立审计逐行检查原始文件，并从源扰动与三级事件日志重建节点和边真值。",
        "本报告重新核对全部被审计文件的摘要，统计不依赖填写的 metadata。Risk Labels 表示已知正负实体－查询实例合计；每个实体至少有一次标注，其他实例为 -1。", "",
        "| 数据集 | Firms | Products | Materials | Industries | Regions | Relations | Texts | Signals | Labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in dataset_rows:
        lines.append("| " + " | ".join([row["dataset"], *[f"{row[key]:,}" for key in STAT_FIELDS]]) + " |")
    lines.extend(["", "实际文本时间跨度为 2018-01-01 至 2025-12-31。96 个逐月查询，30 天预测窗；56/18/20 个训练/验证/测试查询，2 个边界查询清除。首月没有虚构的 2017 年预警。",
        "记忆仅检索查询日前 30 天信号，K=5、D=64。文本特征使用 SHA-256 有符号哈希，仅用于诊断；本轮未使用论文的 8B LLM。", "",
        "| 数据集 | 划分 | 查询数 | 已知正例 | 已知负例 | 未标注实例 |", "|---|---|---:|---:|---:|---:|"])
    for name in DATASETS:
        for split, values in materials[name]["annotation_splits"].items():
            lines.append(f"| {name} | {split} | {values['queries']} | {values['positive']} | {values['negative']} | {values['unknown']} |")
    lines.extend(["", "## 固定实验协议", "", "H=128，2 层，学习率 3e-4，batch size=1，最多 60 epochs，early stopping patience=12。检查点只按验证集 AUPRC 选择；节点和边阈值分别只按验证集 Macro F1 和 Positive F1 调节。未运行超参数搜索。",
        "训练随机种子为 17、29、43、71、101；每个领域只有一个固定生成图，均值 ± 样本标准差（ddof=1）反映优化过程的随机性，不是跨图泛化证据或人群置信区间。",
        "范围排序仅针对当日已标注节点，top-k=20（上限为已知节点数）；该 k 与记忆检索 K=5 含义不同。",
        "text_mlp、graph_only、concat 是工程比较基线，不是原文表 2 所有命名方法的复现。基线边及路径评分为未训练的节点概率 × 依赖权重启发式，不支持因果传播结论。", "",
        f"固定协议 SHA-256：`{validation['protocol_sha256']}`。源码摘要及每次运行材料摘要见 material_passport.json。", ""])
    for dataset in DATASETS:
        lines.extend([f"## {dataset}：测试集结果", "", "| 方法 | AUC | AUPRC | Macro F1 | Recall@20 | Scope NDCG@20 | Edge AUPRC | Path F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
        rows = {row["mode"]: row for row in summary if row["dataset"] == dataset}
        selected_metrics = ("node.auc", "node.auprc", "node.macro_f1", "node.recall_at_k", "scope.ndcg_at_k", "edge.auprc", "path.path_f1")
        for mode in MODES:
            lines.append("| " + " | ".join([mode, *[display(rows[mode]["metrics"][key]) for key in selected_metrics]]) + " |")
        lines.extend(["", "| 方法 | 每次运行秒数：均值 ± SD | 最大 CUDA allocated GiB | 最佳 epoch：均值 ± SD |",
            "|---|---:|---:|---:|"])
        for mode in MODES:
            row = rows[mode]
            gpu = row["cuda_peak_allocated_gib"]["max"]
            lines.append(f"| {mode} | {display(row['runtime_seconds'])} | {'NA' if gpu is None else f'{gpu:.3f}'} | {display(row['best_epoch'])} |")
        lines.append("")
    lines.extend(["## 解释与复核", "", "运行秒数是每个进程的实际墙钟时间；并发执行时包含资源竞争，求和不等于整个调度的耗时。CUDA 数值是 PyTorch 每个进程记录的峰值 allocated memory，不是整张 GPU 的保留内存或设备总占用。",
        "风险标签与传播边真值来自独立的延迟随机传输和库存耗尽过程，未使用 DM-ROR 神经公式生成标签。本轮证明工程规模、接口和可重复实验可运行；不能把结果作为真实供应链效果或论文既有数值。",
        "summary.json/summary.csv 保存实际五种子汇总及有效样本数；per_seed.csv 保存每个种子、时间、GPU 内存和指标。material_passport.json 保存数据、源代码、协议和运行文件摘要；dataset_statistics.csv/.tex 是明确标注 Synthetic 的表 1 数量副本。",
        "本报告工具没有生成图件，也没有修改论文稿件。", ""])
    (output / "experiment_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "passed", "runs": len(runs), "groups": len(summary), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/table_scale")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/table_scale")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/table_scale")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/table_scale.json")
    parser.add_argument("--data-audit", type=Path, default=ROOT / "reports/table_scale_validation.json")
    args = parser.parse_args(argv)
    try:
        outcome = report(args.results.resolve(), args.output.resolve(), args.data_root.resolve(),
                         args.protocol.resolve(), args.data_audit.resolve())
    except (OSError, KeyError, ValueError) as error:
        print(f"Report refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(outcome, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
