"""Dispatch a frozen research protocol over exclusive GPU slots; preserve failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--devices", nargs="+", default=["cpu"])
    args = p.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    jobs = list(protocol["jobs"])
    manifest = {"protocol": str(protocol_path), "jobs": [], "started": time.time(), "status": "running"}
    output = ROOT / protocol["output"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dispatch_manifest.json"
    waiting, running = jobs.copy(), {}
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    while waiting or running:
        for device in args.devices:
            if device in running or not waiting:
                continue
            job = waiting.pop(0)
            run = output / job["name"]
            if (run / "result.json").exists():
                raise FileExistsError(f"Use a fresh protocol output to avoid silent stale results: {run}")
            run.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "dmror.train", "--dataset", str(ROOT / job["dataset"]), "--output", str(run), "--device", device]
            for key, value in {**protocol.get("common", {}), **job.get("parameters", {})}.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            log = open(run / "console.log", "w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record = {"name": job["name"], "device": device, "pid": process.pid, "started": time.time(), "status": "running", "command": command}
            manifest["jobs"].append(record)
            running[device] = (process, log, record)
            print(f"START {job['name']} on {device}", flush=True)
            save()
        for device, (process, log, record) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            record.update(status="completed" if code == 0 else "failed", returncode=code, ended=time.time())
            print(f"END {record['name']} exit={code}", flush=True)
            del running[device]
            save()
        if running:
            time.sleep(5)
    manifest.update(status="failed" if any(record['status'] == 'failed' for record in manifest['jobs']) else "completed", ended=time.time())
    save()
    if manifest["status"] == "failed":
        raise SystemExit("Failures preserved; inspect failure.json and console.log before any rerun")


if __name__ == "__main__":
    main()
