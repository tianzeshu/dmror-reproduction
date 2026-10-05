"""Download only the annual files listed by DOE's public summary archive."""
import json
from pathlib import Path
import concurrent.futures
from collect_public_energy import fetch

ROOT = Path(__file__).resolve().parent / "energy_sources"

def main():
    years = list(range(2018, 2024))
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(fetch, f"https://doe417.pnnl.gov/summaries/{year}_Annual_Summary.xls",
                            f"raw/doe417_{year}_Annual_Summary.xls"): year for year in years}
        for job in concurrent.futures.as_completed(jobs):
            records.append(job.result())
    (ROOT / "doe417_download_manifest.json").write_text(json.dumps({
        "source": "US DOE-417 public annual electric emergency incidents and disturbances",
        "archive_index": "https://doe417.pnnl.gov/",
        "archive_link_evidence": "../additional_source_review/doe417_index-DtrryTHD.js",
        "listed_years_downloaded": years, "missing_requested_years": [2024, 2025],
        "is_synthetic": False, "downloads": records,
        "scope_note": "Incident reports, not causal propagation paths; absence of report is not a verified negative."}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
