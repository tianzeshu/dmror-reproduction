"""Select configuration using validation AUPRC only and freeze five-seed suite."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["full", "text_mlp", "graph_only", "concat", "no_stm", "no_ltm", "no_time", "no_resgate", "no_overload", "avg_fusion", "scalar_gate", "no_align", "no_path"]
SEEDS = [17, 29, 43, 71, 101]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default="results/validation_search")
    parser.add_argument("--output", default="protocols/formal.json")
    parser.add_argument("--baseline-search", default="results/baseline_validation_search")
    args = parser.parse_args()
    search = ROOT / args.search
    selected = {}
    for dataset in sorted(search.iterdir()):
        if not dataset.is_dir():
            continue
        rows = []
        for result_path in sorted(dataset.glob("*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append({"configuration": result_path.parent.name, "validation_auprc": result["validation"]["node"]["auprc"], "lr": result["config"]["lr"], "hidden": result["config"]["hidden"], "best_epoch": result["best_epoch"]})
        if len(rows) != 4:
            raise ValueError(f"Expected four completed search runs for {dataset.name}; got {len(rows)}")
        winner = max(rows, key=lambda row: (row["validation_auprc"], -row["hidden"], -row["lr"]))
        selected[dataset.name] = {"winner": winner, "validation_only_candidates": rows}
    if len(selected) != 4:
        raise ValueError("Expected all four datasets")
    baseline_selected = {}
    for dataset in selected:
        baseline_selected[dataset] = {}
        for mode in ["text_mlp", "graph_only", "concat"]:
            rows = []
            for result_path in sorted((ROOT / args.baseline_search / dataset / mode).glob("*/result.json")):
                result = json.loads(result_path.read_text(encoding="utf-8"))
                rows.append({"configuration": result_path.parent.name, "validation_auprc": result["validation"]["node"]["auprc"], "lr": result["config"]["lr"], "hidden": result["config"]["hidden"], "best_epoch": result["best_epoch"]})
            if len(rows) != 4:
                raise ValueError(f"Expected four completed baseline search runs: {dataset}/{mode}")
            baseline_selected[dataset][mode] = {"winner": max(rows, key=lambda row: (row["validation_auprc"], -row["hidden"], -row["lr"])), "validation_only_candidates": rows}
    protocol = {"stage": "formal_five_seed_suite", "output": "results/formal", "selection": "validation AUPRC only; full and each independent baseline have equal 4-configuration budgets; ablations retain full settings", "selected": selected, "baseline_selected": baseline_selected, "common": {"epochs": 60, "patience": 12, "threads": 1, "batch_size": 8, "layers": 2}, "jobs": []}
    for dataset, selection in selected.items():
        winner = selection["winner"]
        for mode in MODES:
            winner = baseline_selected[dataset][mode]["winner"] if mode in baseline_selected[dataset] else selection["winner"]
            for seed in SEEDS:
                protocol["jobs"].append({"name": f"{dataset}/{mode}/seed_{seed}", "dataset": f"data/prepared/{dataset}/dataset.npz", "parameters": {"mode": mode, "seed": seed, "lr": winner["lr"], "hidden": winner["hidden"]}})
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for dataset, selection in selected.items():
        print(dataset, selection["winner"], flush=True)
    print("Frozen", len(protocol["jobs"]), "formal runs; no test score used in selection")


if __name__ == "__main__":
    main()
