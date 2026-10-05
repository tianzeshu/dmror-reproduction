"""Dispatch a frozen research protocol over exclusive GPU slots; preserve failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--devices", nargs="+", default=["cpu"])
    p.add_argument("--slots-per-device", type=int, default=1,
                   help="Independent workers sharing each device; budget total RAM/VRAM before increasing")
    p.add_argument("--resume", action="store_true",
                   help="Retain completed runs only when data, implementation and frozen protocol match")
    args = p.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if args.slots_per_device < 1 or len(set(args.devices)) != len(args.devices):
        raise ValueError("slots-per-device must be positive and devices must be unique")
    slots = [(device, index) for device in args.devices for index in range(args.slots_per_device)]
    jobs = list(protocol["jobs"])
    for job in jobs:
        if not (ROOT / job["dataset"]).is_file():
            raise FileNotFoundError(ROOT / job["dataset"])
    manifest = {"protocol": str(protocol_path), "protocol_sha256": protocol_digest,
                "slots_per_device": args.slots_per_device, "jobs": [], "started": time.time(), "status": "running"}
    output = ROOT / protocol["output"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dispatch_manifest.json"
    waiting, running = jobs.copy(), {}
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    while waiting or running:
        for slot in slots:
            if slot in running or not waiting:
                continue
            device, slot_index = slot
            job = waiting.pop(0)
            run = output / job["name"]
            if (run / "result.json").exists():
                if not args.resume:
                    raise FileExistsError(f"Use --resume to explicitly verify a completed run: {run}")
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                config = result.get("config", {})
                from dmror.train import sha256
                compatible = result.get("status") == "completed" and config.get("protocol_sha256") == protocol_digest
                compatible &= config.get("dataset_sha256") == sha256(ROOT / job["dataset"])
                compatible &= all(config.get(key) == value for key, value in
                                  {**protocol.get("common", {}), **job.get("parameters", {})}.items())
                current_code = {file.name: sha256(file) for file in sorted((ROOT / "dmror").glob("*.py"))}
                compatible &= config.get("implementation_sha256") == current_code
                if not compatible:
                    raise RuntimeError(f"Stale or incompatible run; use a fresh protocol output: {run}")
                manifest["jobs"].append({"name": job["name"], "status": "retained", "configuration_sha256": config.get("configuration_sha256")})
                print(f"RETAIN {job['name']}", flush=True)
                save()
                continue
            run.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "dmror.train", "--dataset", str(ROOT / job["dataset"]), "--output", str(run), "--device", device,
                       "--protocol-sha256", protocol_digest]
            for key, value in {**protocol.get("common", {}), **job.get("parameters", {})}.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            log = open(run / "console.log", "w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record = {"name": job["name"], "device": device, "slot": slot_index,
                      "pid": process.pid, "started": time.time(), "status": "running", "command": command}
            manifest["jobs"].append(record)
            running[slot] = (process, log, record)
            print(f"START {job['name']} on {device} slot={slot_index}", flush=True)
            save()
        for slot, (process, log, record) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            record.update(status="completed" if code == 0 else "failed", returncode=code, ended=time.time())
            print(f"END {record['name']} exit={code}", flush=True)
            del running[slot]
            save()
        if running:
            time.sleep(5)
    manifest.update(status="failed" if any(record['status'] == 'failed' for record in manifest['jobs']) else "completed", ended=time.time())
    save()
    if manifest["status"] == "failed":
        raise SystemExit("Failures preserved; inspect failure.json and console.log before any rerun")


if __name__ == "__main__":
    main()
