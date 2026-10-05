"""Collect documented US government EIA bulk releases, preserving provenance."""
import concurrent.futures
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parent / "energy_sources"
UA = "SupplyChainResearchCollector/1.0 (https://github.com/tianzeshu/dmror-reproduction)"


def fetch(url, relative):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    proof = target.with_name(target.name + ".provenance.json")
    if target.exists() and proof.exists():
        recorded = json.loads(proof.read_text(encoding="utf-8"))
        if hashlib.sha256(target.read_bytes()).hexdigest() == recorded["sha256"]:
            return recorded
        raise ValueError(f"Existing raw file changed: {target}")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        partial = target.with_name(target.name + ".partial")
        count = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 256):
                handle.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        partial.replace(target)
        result = {"source_url": url, "response_url": response.url,
                  "fetched_at_utc": started, "status_code": response.status_code,
                  "size_bytes": count, "sha256": digest.hexdigest(),
                  "content_type": response.headers.get("Content-Type"),
                  "last_modified": response.headers.get("Last-Modified"),
                  "etag": response.headers.get("ETag"),
                  "relative_path": str(target.relative_to(ROOT)).replace("\\", "/")}
    proof.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": relative, "bytes": count, "sha256": result["sha256"]}), flush=True)
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source = "https://www.eia.gov/electricity/data/eia860/"
    records = [fetch(source, "raw/eia860_index.html"),
               fetch("https://www.eia.gov/about/copyrights_reuse.php", "raw/eia_reuse_policy.html")]
    html = (ROOT / "raw/eia860_index.html").read_text(encoding="utf-8")
    urls = {}
    for href in re.findall(r'href=[\"\x27]([^\"\x27]+)', html, re.I):
        matched = re.search(r"eia860(201[89]|202[0-5])\.zip$", href, re.I)
        if matched:
            urls[int(matched.group(1))] = urljoin(source, href)
    if set(urls) != set(range(2018, 2026)):
        raise RuntimeError(f"Official index did not list all requested releases: {sorted(urls)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(fetch, url, f"raw/eia860{year}.zip"): year for year, url in urls.items()}
        for job in concurrent.futures.as_completed(jobs):
            records.append(job.result())
    (ROOT / "download_manifest.json").write_text(json.dumps({
        "source": "US Energy Information Administration, Form EIA-860 annual detailed survey data",
        "requested_years": list(range(2018, 2026)), "is_synthetic": False,
        "source_policy_url": "https://www.eia.gov/about/copyrights_reuse.php",
        "scope_note": "Generator, plant and utility facts; not product-specific supply links or risk gold labels.",
        "downloads": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
