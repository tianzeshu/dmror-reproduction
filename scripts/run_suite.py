"""Run every dataset/method/seed through the same temporal evaluation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmror.train import DEFAULT_SEEDS, EXPERIMENT_MODES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(ROOT / "data" / "prepared"))
    p.add_argument("--output", default=str(ROOT / "results"))
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Names relative to data-root. Default: every */dataset.npz")
    p.add_argument("--modes", nargs="+", default=list(EXPERIMENT_MODES), choices=EXPERIMENT_MODES)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()
    data_root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    datasets = [data_root / name / "dataset.npz" for name in args.datasets] if args.datasets else sorted(data_root.glob("*/dataset.npz"))
    if not datasets:
        raise FileNotFoundError(f"No dataset.npz under {data_root}")
    for dataset in datasets:
        if not dataset.exists():
            raise FileNotFoundError(dataset)
    output.mkdir(parents=True, exist_ok=True)
    entries, started = [], time.perf_counter()
    for dataset in datasets:
        for mode in args.modes:
            for seed in args.seeds:
                run = output / dataset.parent.name / mode / f"seed_{seed}"
                if (run / "result.json").exists() and not args.overwrite:
                    existing = json.loads((run / "result.json").read_text(encoding="utf-8"))
                    # Only skip a completed/skipped run if its data and requested
                    # training budget agree. Stale results must be explicit.
                    old = existing.get("config", {})
                    from dmror.train import sha256
                    compatible = old.get("dataset_sha256") == sha256(dataset)
                    compatible &= all(old.get(key) == getattr(args, key) for key in
                                      ("epochs", "patience", "hidden", "layers", "lr", "batch_size"))
                    if not compatible:
                        raise RuntimeError(f"Stale result: {run}; use a new output or --overwrite")
                    print(f"resume: retained {run}", flush=True)
                    entries.append({"dataset": dataset.parent.name, "mode": mode, "seed": seed,
                                    "status": "retained", "run": str(run)})
                    continue
                run.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "dmror.train", "--dataset", str(dataset),
                           "--output", str(run), "--mode", mode, "--seed", str(seed)]
                for key in ("epochs", "patience", "device", "threads", "batch_size", "hidden", "layers", "lr"):
                    command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
                if args.overwrite:
                    command.append("--overwrite")
                print(f"run {dataset.parent.name} {mode} seed={seed}", flush=True)
                with (run / "console.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                               errors="replace", bufsize=1)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    returncode = process.wait()
                entries.append({"dataset": dataset.parent.name, "mode": mode, "seed": seed,
                                "status": "completed" if returncode == 0 else "failed",
                                "returncode": returncode, "run": str(run)})
                manifest = {"requested": vars(args), "runs": entries,
                            "elapsed_seconds": time.perf_counter() - started}
                (output / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                if returncode and not args.continue_on_error:
                    raise RuntimeError(f"Training failed: {run}; inspect console.log and failure.json")
    (output / "suite_manifest.json").write_text(json.dumps({"requested": vars(args), "runs": entries,
                   "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    if any(row.get("returncode", 0) for row in entries):
        raise SystemExit("Suite finished with failed runs; failures are preserved in suite_manifest.json")
    print(f"Suite complete; report with: python -m dmror.report --results {output}", flush=True)


if __name__ == "__main__":
    main()
