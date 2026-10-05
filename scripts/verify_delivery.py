"""Check the complete frozen handoff without training or loading pickle weights.

Default verification is strict. --allow-incomplete is for an in-progress local
download only: missing outputs are deferred, but corrupt existing files fail.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import zipfile

import numpy as np

DATASETS = {"SIM-Auto", "SIM-Semi", "SIM-Energy", "ICKG-Weak"}
SEEDS = {17, 29, 43, 71, 101}
MODES = {"full", "text_mlp", "graph_only", "concat", "no_stm", "no_ltm",
         "no_time", "no_resgate", "no_overload", "avg_fusion", "scalar_gate",
         "no_align", "no_path"}
RUN_FILES = ("config.json", "result.json", "checkpoint.pt", "scaler.npz",
             "history.json", "predictions.npz", "console.log")
SCALAR_METRICS = {"auc", "auprc", "macro_f1", "positive_f1", "recall_at_k",
                  "precision_at_k", "ndcg_at_k", "jaccard_at_k", "jaccard",
                  "jaccard_positive_days", "path_at_k", "path_precision",
                  "path_recall", "path_f1", "path_edge_precision", "path_edge_recall"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class Audit:
    def __init__(self, allow_incomplete=False):
        self.allow_incomplete = allow_incomplete
        self.errors, self.pending, self.warnings = [], [], []
        self.checked_runs = 0
        self.checked_search_runs = 0
        self.unfingerprinted_early_validation_runs = []
        self.validation_threshold_selector = None
        self.dataset_info = {}
        self.source_sets = defaultdict(list)

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def missing(self, path, label):
        if Path(path).is_file() and Path(path).stat().st_size > 0:
            return False
        (self.pending if self.allow_incomplete else self.errors).append(f"{label}: missing/empty {path}")
        return True

    def guarded(self, label, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self.errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Protocol path escapes package root: {relative}")
    return path


def check_protocol(audit, protocol):
    jobs = protocol["jobs"]
    audit.require(len(jobs) == 260, f"Protocol requires exactly 260 jobs, found {len(jobs)}")
    names = [job["name"] for job in jobs]
    audit.require(len(set(names)) == len(names), "Duplicate protocol job names")
    triples = [(Path(job["dataset"]).parent.name, job["parameters"]["mode"],
                int(job["parameters"]["seed"])) for job in jobs]
    expected = {(dataset, mode, seed) for dataset in DATASETS for mode in MODES for seed in SEEDS}
    audit.require(set(triples) == expected and len(triples) == len(set(triples)),
                  "Protocol does not contain exactly four datasets x 13 modes x five prescribed seeds")
    for job, (dataset, mode, seed) in zip(jobs, triples):
        audit.require(job["name"] == f"{dataset}/{mode}/seed_{seed}",
                      f"Unexpected run name for dataset/mode/seed: {job['name']}")
        audit.require(job["dataset"] == f"data/prepared/{dataset}/dataset.npz",
                      f"Unexpected frozen dataset path: {job['dataset']}")
    return jobs


def check_dataset(audit, root, dataset):
    directory = root / "data" / "prepared" / dataset
    path = directory / "dataset.npz"
    if audit.missing(path, dataset) or audit.missing(directory / "metadata.json", dataset):
        return None
    data, metadata = load_npz(path), read_json(directory / "metadata.json")
    digest = sha256(path)
    audit.require(metadata["statistics"]["sha256"] == digest, f"{dataset}: metadata NPZ SHA mismatch")
    for key, array in data.items():
        if np.issubdtype(array.dtype, np.number):
            audit.require(np.isfinite(array).all(), f"{dataset}: nonfinite input array {key}")
    audit.require(set(np.unique(data["labels"])) <= {-1, 0, 1}, f"{dataset}: invalid node labels")
    audit.require(set(np.unique(data["edge_labels"])) <= {-1, 0, 1}, f"{dataset}: invalid edge labels")
    indices = {"train": np.flatnonzero(data["split"] == 0),
               "validation": np.flatnonzero(data["split"] == 1),
               "test": np.flatnonzero(data["split"] == 2)}
    for split, rows in indices.items():
        audit.require(len(rows) > 0, f"{dataset}: empty {split} split")
    audit.require(data["target_days"][indices["train"]].max() <= data["prediction_days"][indices["validation"]].min(),
                  f"{dataset}: training targets overlap validation query times")
    audit.require(data["target_days"][indices["validation"]].max() <= data["prediction_days"][indices["test"]].min(),
                  f"{dataset}: validation targets overlap test query times")
    if dataset == "ICKG-Weak":
        audit.require((data["edge_labels"] == -1).all(), "ICKG-Weak: edge truth must remain entirely unknown")
        audit.require(not data["source_mask"].any(), "ICKG-Weak: no observed causal propagation sources are available")
    audit.dataset_info[dataset] = {"sha256": digest, "nodes": len(data["node_features"]),
                                    "edges": len(data["src"]), "snapshots": len(data["split"]),
                                    "split_counts": {key: len(rows) for key, rows in indices.items()}}
    return {"data": data, "metadata": metadata, "digest": digest, "indices": indices}


def check_numeric_json(audit, value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SCALAR_METRICS and item is not None:
                audit.require(isinstance(item, (int, float)) and math.isfinite(item) and 0 <= item <= 1,
                              f"{label}.{key}: invalid bounded metric {item}")
            check_numeric_json(audit, item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_numeric_json(audit, item, f"{label}[{index}]")
    elif isinstance(value, float):
        audit.require(math.isfinite(value), f"{label}: nonfinite JSON value")


def check_binary_counts(audit, block, labels, label):
    known = labels >= 0
    audit.require(block["n_known"] == int(known.sum()), f"{label}: known-label denominator mismatch")
    audit.require(block["n_positive"] == int((labels == 1).sum()), f"{label}: positive denominator mismatch")
    if known.any():
        for name in ("macro_f1", "positive_f1"):
            audit.require(block.get(name) is not None, f"{label}: missing {name} despite known labels")
        if (labels == 1).any():
            audit.require(block.get("auprc") is not None, f"{label}: missing AUPRC despite positive labels")
        if (labels == 1).any() and (labels == 0).any():
            audit.require(block.get("auc") is not None, f"{label}: missing AUC despite both classes")
    else:
        for name in ("auc", "auprc", "macro_f1", "positive_f1"):
            audit.require(block.get(name) is None, f"{label}: unavailable metric {name} must be null")


def check_run(audit, root, output, protocol, job, frozen, track_formal_source=True,
              allow_early_validation_without_fingerprint=False):
    name = job["name"]
    directory = inside(output, name)
    absent = [filename for filename in RUN_FILES if audit.missing(directory / filename, name)]
    if absent or frozen is None:
        return
    result, config = read_json(directory / "result.json"), read_json(directory / "config.json")
    dataset = Path(job["dataset"]).parent.name
    expected_parameters = {**protocol.get("common", {}), **job["parameters"]}
    audit.require(result["status"] == "completed", f"{name}: result status is not completed")
    audit.require((result["dataset"], result["mode"], result["seed"]) ==
                  (dataset, expected_parameters["mode"], expected_parameters["seed"]),
                  f"{name}: result identity differs from protocol")
    audit.require(result["config"] == config, f"{name}: saved config differs from result's config")
    for key, value in expected_parameters.items():
        audit.require(config.get(key) == value, f"{name}: {key} differs from frozen protocol")
    for cfg in (config, result["config"]):
        audit.require(cfg["dataset_sha256"] == frozen["digest"], f"{name}: run/dataset NPZ SHA mismatch")
        audit.require(cfg["dataset_metadata"] == frozen["metadata"], f"{name}: dataset metadata differs from frozen input")
    if "implementation_sha256" not in config:
        # The initial 16 full-model validation runs predate source-fingerprint
        # logging. This exception is authorized only by the exact early search
        # protocol below, never by a formal run or the later baseline search.
        allowed = (allow_early_validation_without_fingerprint and not track_formal_source
                   and protocol.get("stage") == "validation_search")
        audit.require(allowed, f"{name}: required source fingerprint map is missing")
        if allowed:
            audit.unfingerprinted_early_validation_runs.append(name)
    else:
        implementation = config["implementation_sha256"]
        audit.require({"model.py", "train.py", "losses.py", "metrics.py", "paths.py", "data.py"} <= set(implementation),
                      f"{name}: source fingerprint map omits training implementation files")
        audit.require(all(Path(key).name == key and re.fullmatch(r"[0-9a-f]{64}", value)
                          for key, value in implementation.items()), f"{name}: invalid source fingerprint map")
        source_signature = json.dumps(implementation, sort_keys=True)
        if track_formal_source:
            audit.source_sets[source_signature].append(name)

    data, indices = frozen["data"], frozen["indices"]
    predictions = load_npz(directory / "predictions.npz")
    for split in ("validation", "test"):
        rows = indices[split]
        audit.require(np.array_equal(predictions[f"{split}_indices"], rows), f"{name}: {split} indices differ")
        audit.require(np.array_equal(predictions[f"{split}_labels"], data["labels"][rows]), f"{name}: {split} labels differ")
        for key, width in (("node_prob", len(data["node_features"])), ("edge_prob", len(data["src"])),
                           ("edge_score", len(data["src"]))):
            array = predictions[f"{split}_{key}"]
            audit.require(array.shape == (len(rows), width), f"{name}: invalid {split}_{key} shape {array.shape}")
            audit.require(np.isfinite(array).all() and (array >= 0).all(), f"{name}: invalid {split}_{key} values")
            if key.endswith("prob"):
                audit.require((array <= 1).all(), f"{name}: probability exceeds one in {split}_{key}")
        block = result[split]
        check_numeric_json(audit, block, f"{name}/{split}")
        check_binary_counts(audit, block["node"], data["labels"][rows], f"{name}/{split}/node")
        check_binary_counts(audit, block["edge"], data["edge_labels"][rows], f"{name}/{split}/edge")
        for key in ("precision_at_k", "recall_at_k", "ndcg_at_k", "jaccard_at_k", "jaccard"):
            audit.require(block["scope"].get(key) is not None, f"{name}/{split}: missing scope metric {key}")
        if dataset == "ICKG-Weak":
            audit.require(block["path"]["status"] == "skipped_no_edge_labels", f"{name}/{split}: weak path metric not skipped")
            for key in ("path_at_k", "path_precision", "path_recall", "path_f1"):
                audit.require(block["path"].get(key) is None, f"{name}/{split}: weak path metric {key} must be null")
        else:
            audit.require(block["path"]["status"] == "computed", f"{name}/{split}: simulator path metrics not computed")
            audit.require(block["path"].get("path_f1") is not None, f"{name}/{split}: missing simulator Path-F1")
    if audit.validation_threshold_selector is not None:
        for target, label_key, probability_key, objective in (
                ("node", "labels", "node_prob", "macro_f1"),
                ("edge", "edge_labels", "edge_prob", "positive_f1")):
            selected_threshold = audit.validation_threshold_selector(
                data[label_key][indices["validation"]], predictions[f"validation_{probability_key}"], objective)
            for split in ("validation", "test"):
                saved = result[split][target]["threshold"]
                audit.require(math.isfinite(saved) and 0 <= saved <= 1 and
                              math.isclose(saved, selected_threshold, rel_tol=0, abs_tol=1e-7),
                              f"{name}/{split}/{target}: threshold does not match validation-only {objective} selection")
            if target == "node":
                for split in ("validation", "test"):
                    audit.require(math.isclose(result[split]["scope"]["jaccard_threshold"], selected_threshold,
                                               rel_tol=0, abs_tol=1e-7),
                                  f"{name}/{split}: scope threshold differs from selected node threshold")
    test_rows = indices["test"]
    for key, expected in (("test_edge_labels", data["edge_labels"][test_rows]),
                          ("prediction_days", data["prediction_days"][test_rows]),
                          ("target_days", data["target_days"][test_rows])):
        audit.require(np.array_equal(predictions[key], expected), f"{name}: saved {key} differs from frozen input")

    scaler = load_npz(directory / "scaler.npz")
    for key, dimension in (("node_mean", data["node_features"].shape[-1]),
                           ("node_std", data["node_features"].shape[-1]),
                           ("signal_mean", data["signals"].shape[-1]),
                           ("signal_std", data["signals"].shape[-1])):
        audit.require(scaler[key].shape == (dimension,) and np.isfinite(scaler[key]).all(), f"{name}: invalid scaler {key}")
        if key.endswith("std"):
            audit.require((scaler[key] > 0).all(), f"{name}: nonpositive scaler {key}")
    history = read_json(directory / "history.json")
    audit.require(len(history) == result["epochs_run"] > 0, f"{name}: history/epochs_run mismatch")
    audit.require([row["epoch"] for row in history] == list(range(1, len(history) + 1)), f"{name}: discontinuous history epochs")
    check_numeric_json(audit, history, f"{name}/history")
    selected = [row for row in history if row["epoch"] == result["best_epoch"]]
    audit.require(len(selected) == 1 and selected[0]["validation_auprc"] == result["best_validation_auprc"],
                  f"{name}: selected epoch/AP not in saved history")
    audit.require(math.isfinite(result["runtime_seconds"]) and result["runtime_seconds"] > 0, f"{name}: invalid runtime")
    # torch.save's zip container can be checked without unpickling arbitrary code.
    with zipfile.ZipFile(directory / "checkpoint.pt") as checkpoint:
        audit.require(any(member.endswith("/data.pkl") for member in checkpoint.namelist()), f"{name}: invalid torch checkpoint structure")
        audit.require(checkpoint.testzip() is None, f"{name}: checkpoint ZIP CRC failure")
    audit.require(not (directory / "failure.json").exists(), f"{name}: unresolved failure.json exists")
    if track_formal_source:
        audit.checked_runs += 1
    else:
        audit.checked_search_runs += 1


def check_manifest(audit, output, jobs):
    path = output / "dispatch_manifest.json"
    if audit.missing(path, "Formal dispatcher"):
        return
    manifest = read_json(path)
    records = manifest["jobs"]
    planned = {job["name"] for job in jobs}
    actual = [record["name"] for record in records]
    audit.require(len(actual) == len(set(actual)), "Duplicate dispatcher records")
    audit.require(set(actual) <= planned, "Dispatcher contains unplanned jobs")
    not_completed = [record["name"] for record in records if record["status"] != "completed"]
    failed = [record["name"] for record in records if record["status"] == "failed" or record.get("returncode", 0) != 0]
    audit.require(not failed, f"Formal dispatcher has failed jobs: {failed}")
    audit.require(all(record.get("returncode") == 0 for record in records if record["status"] == "completed"),
                  "A completed dispatcher record lacks successful returncode=0")
    if set(actual) != planned or not_completed or manifest["status"] != "completed":
        message = f"Formal dispatcher incomplete: {len(actual)}/{len(planned)} dispatched; {len(not_completed)} unfinished"
        (audit.pending if audit.allow_incomplete else audit.errors).append(message)


def check_sources(audit, source_dir):
    audit.require(len(audit.source_sets) <= 1, f"Formal runs have {len(audit.source_sets)} different frozen implementation maps")
    for signature, runs in audit.source_sets.items():
        recorded = json.loads(signature)
        for name, digest in recorded.items():
            path = source_dir / name
            if audit.missing(path, "Frozen source"):
                continue
            audit.require(sha256(path) == digest, f"Frozen source {name} differs from {len(runs)} formal run fingerprints")
        added = sorted(path.name for path in source_dir.glob("*.py") if path.name not in recorded)
        if added:
            audit.warnings.append(f"Additive modules absent from training-time map (not used for these runs): {added}")


def check_validation_selection(audit, root, formal, frozen):
    """Rebuild frozen winners from validation fields; do not read test scores."""
    gathered = defaultdict(list)
    summary = {}
    for filename, expected_count in (("validation_search.json", 16), ("baseline_validation_search.json", 48)):
        path = root / "protocols" / filename
        if audit.missing(path, "Validation protocol"):
            continue
        protocol = read_json(path)
        jobs = protocol["jobs"]
        audit.require(len(jobs) == expected_count, f"{filename}: expected {expected_count} planned jobs")
        names = [job["name"] for job in jobs]
        audit.require(len(set(names)) == len(names), f"{filename}: duplicate search jobs")
        output = inside(root, protocol["output"])
        actual = set(path.parent.relative_to(output).as_posix() for path in output.rglob("result.json")) if output.exists() else set()
        audit.require(actual <= set(names), f"{filename}: unplanned search results {sorted(actual - set(names))}")
        summary[filename] = {"planned": len(jobs), "present": len(actual)}
        audit.guarded(filename + " manifest", check_manifest, audit, output, jobs)
        budgets = defaultdict(list)
        for job in jobs:
            dataset = Path(job["dataset"]).parent.name
            parameters = {**protocol.get("common", {}), **job["parameters"]}
            mode = parameters["mode"]
            budgets[(dataset, mode)].append((parameters["hidden"], parameters["lr"], parameters["seed"]))
            audit.guarded(filename + "/" + job["name"], check_run, audit, root, output, protocol,
                          job, frozen.get(dataset), False,
                          filename == "validation_search.json" and protocol.get("stage") == "validation_search")
            result_path = output / job["name"] / "result.json"
            if not result_path.is_file():
                continue
            # This extraction intentionally selects only validation values.
            result = read_json(result_path)
            row = {"configuration": result_path.parent.name,
                   "validation_auprc": result["validation"]["node"]["auprc"],
                   "lr": result["config"]["lr"], "hidden": result["config"]["hidden"],
                   "best_epoch": result["best_epoch"]}
            gathered[(dataset, mode)].append(row)
        expected_modes = {"full"} if expected_count == 16 else {"text_mlp", "graph_only", "concat"}
        audit.require(set(budgets) == {(dataset, mode) for dataset in DATASETS for mode in expected_modes},
                      f"{filename}: dataset/mode search coverage differs")
        for identity, settings in budgets.items():
            audit.require(set(settings) == {(hidden, lr, 17) for hidden in (64, 128) for lr in (.0003, .001)}
                          and len(settings) == 4, f"{filename}: unequal four-configuration budget for {identity}")
    for dataset in sorted(DATASETS):
        for mode in ("full", "text_mlp", "graph_only", "concat"):
            identity = (dataset, mode)
            rows = sorted(gathered[identity], key=lambda row: row["configuration"])
            if len(rows) != 4:
                continue  # Missing outputs have already been reported by check_run.
            selected = formal["selected"][dataset] if mode == "full" else formal["baseline_selected"][dataset][mode]
            winner = max(rows, key=lambda row: (row["validation_auprc"], -row["hidden"], -row["lr"]))
            audit.require(selected["validation_only_candidates"] == rows,
                          f"{dataset}/{mode}: frozen candidate list differs from saved validation results")
            audit.require(selected["winner"] == winner,
                          f"{dataset}/{mode}: frozen winner does not follow validation-AUPRC selection/tie policy")
            covered_modes = MODES - {"text_mlp", "graph_only", "concat"} if mode == "full" else {mode}
            for job in formal["jobs"]:
                if Path(job["dataset"]).parent.name == dataset and job["parameters"]["mode"] in covered_modes:
                    audit.require((job["parameters"]["hidden"], job["parameters"]["lr"]) == (winner["hidden"], winner["lr"]),
                                  f"{job['name']}: formal settings do not inherit validation winner")
    return summary


def markdown_targets(text):
    # Ignore code fences. Check regular inline/image links and reference links.
    text = re.sub(r"(?ms)^\s*(```|~~~).*?^\s*\1\s*$", "", text)
    references = {}
    for match in re.finditer(r"(?m)^\s*\[([^\]]+)\]:\s*(<[^>]+>|\S+)", text):
        references[match.group(1).casefold()] = match.group(2).strip("<>")
    targets = []
    for match in re.finditer(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text):
        targets.append(match.group(1).strip("<>"))
    for match in re.finditer(r"!?\[([^\]]+)\]\[([^\]]*)\]", text):
        key = (match.group(2) or match.group(1)).casefold()
        if key in references:
            targets.append(references[key])
    return targets


def check_links(audit, root):
    files = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    checked = 0
    for file in files:
        if audit.missing(file, "Documentation"):
            continue
        for target in markdown_targets(file.read_text(encoding="utf-8")):
            if target.startswith(("#", "//")) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            relative = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not relative:
                continue
            linked = (file.parent / relative).resolve()
            checked += 1
            if not linked.exists():
                message = f"Broken local Markdown link in {file.relative_to(root)}: {target}"
                (audit.pending if audit.allow_incomplete else audit.errors).append(message)
    return {"markdown_files": len(files), "local_links_checked": checked}


def verify(root, protocol_path, source_dir, allow_incomplete=False):
    audit = Audit(allow_incomplete)
    # metrics.py imports only NumPy/sklearn; loading its named routine does not
    # load torch, checkpoint pickle objects, or start training. Use the declared
    # frozen source to recompute selection from saved validation arrays only.
    def load_threshold_selector():
        specification = importlib.util.spec_from_file_location("_frozen_delivery_metrics", source_dir / "metrics.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module.tune_threshold
    audit.validation_threshold_selector = audit.guarded("Frozen validation threshold routine", load_threshold_selector)
    protocol = audit.guarded("Protocol", read_json, protocol_path)
    if protocol is None:
        jobs = []
        output = root / "results" / "formal"
    else:
        jobs = audit.guarded("Protocol coverage", check_protocol, audit, protocol) or []
        output = inside(root, protocol["output"])
    frozen = {dataset: audit.guarded(f"Dataset {dataset}", check_dataset, audit, root, dataset) for dataset in sorted(DATASETS)}
    actual_results = set(path.parent.relative_to(output).as_posix() for path in output.rglob("result.json")) if output.exists() else set()
    planned_names = {job["name"] for job in jobs}
    audit.require(actual_results <= planned_names, f"Unplanned formal results: {sorted(actual_results - planned_names)}")
    for job in jobs:
        audit.guarded(job["name"], check_run, audit, root, output, protocol, job, frozen.get(Path(job["dataset"]).parent.name))
    audit.guarded("Dispatcher manifest", check_manifest, audit, output, jobs)
    searches = audit.guarded("Validation searches and frozen selection", check_validation_selection, audit, root, protocol, frozen) if protocol else None
    if audit.unfingerprinted_early_validation_runs:
        audit.warnings.append(f"The {len(audit.unfingerprinted_early_validation_runs)} early full-model validation_search runs predate implementation_sha256 logging. Their original source cannot be verified by a saved hash; no fingerprint was invented or backfilled. Actual configuration/data/predictions/history/checkpoints and validation winner selection were checked. Names are recorded in unfingerprinted_early_validation_runs.")
    audit.guarded("Source fingerprints", check_sources, audit, source_dir)
    links = audit.guarded("Markdown local links", check_links, audit, root)
    groups = Counter((Path(name).parts[0], Path(name).parts[1]) for name in actual_results)
    if actual_results:
        missing_groups = {f"{dataset}/{mode}": count for (dataset, mode), count in groups.items() if count != 5}
        if missing_groups or len(groups) != 52:
            message = f"Incomplete five-seed groups: {len(groups)}/52 groups; sizes differing from five: {missing_groups}"
            (audit.pending if allow_incomplete else audit.errors).append(message)
    status = "failed" if audit.errors else "incomplete" if audit.pending else "passed"
    return {"status": status, "strict": not allow_incomplete,
            "audit_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": str(protocol_path.relative_to(root)) if protocol_path.is_relative_to(root) else str(protocol_path),
            "protocol_sha256": sha256(protocol_path) if protocol_path.exists() else None,
            "source_directory": str(source_dir), "planned_runs": len(jobs),
            "present_results": len(actual_results), "checked_complete_runs": audit.checked_runs,
            "checked_validation_search_runs": audit.checked_search_runs, "validation_searches": searches,
            "unfingerprinted_early_validation_runs": sorted(audit.unfingerprinted_early_validation_runs),
            "present_dataset_mode_groups": len(groups), "source_fingerprint_maps": len(audit.source_sets),
            "datasets": audit.dataset_info, "documentation": links,
            "errors": audit.errors, "pending": audit.pending, "warnings": audit.warnings,
            "scope": "Integrity and execution completeness only; not validation of weak-label facts, causal paths, publication claims, or external dependency availability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path("protocols/formal.json"))
    parser.add_argument("--frozen-source", type=Path, help="Original source directory when delivered tooling was changed after the frozen run; default dmror/")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional full JSON audit record; no training/results are modified")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = args.protocol.resolve() if args.protocol.is_absolute() else root / args.protocol
    source = (args.frozen_source.resolve() if args.frozen_source.is_absolute() else root / args.frozen_source) if args.frozen_source else root / "dmror"
    result = verify(root, protocol, source, args.allow_incomplete)
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key not in ("datasets", "errors", "pending", "warnings")}
    compact.update(error_count=len(result["errors"]), pending_count=len(result["pending"]), warnings=result["warnings"],
                   first_errors=result["errors"][:10], first_pending=result["pending"][:5])
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "failed" or (result["status"] == "incomplete" and not args.allow_incomplete) else 0


if __name__ == "__main__":
    sys.exit(main())
