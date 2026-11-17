"""Download GH Archive hour files with wget. Default: 2026-03-01-0 .. 2026-11-18-15 (UTC) = 03/01 08:00 - 11/19
00:00 Taiwan, the period used for the submitted output.

    python src/download.py --raw RAW_DATA_FOLDER

Existing files are skipped, a file is kept only if wget finished, and missing hours are listed at the end.
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

URL = "https://data.gharchive.org/{}.json.gz"


def name(h):
    return f"{h:%Y-%m-%d}-{h.hour}"  # hour without leading zero, as GH Archive names files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder for the GH Archive hour files")
    ap.add_argument("--start", default="2026-03-01 00:00", help="first hour (UTC)")
    ap.add_argument("--end", default="2026-11-18 15:00", help="last hour (UTC), inclusive")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.raw, exist_ok=True)
    hours = pd.date_range(a.start, a.end, freq="1h")
    path = {h: os.path.join(a.raw, name(h) + ".json.gz") for h in hours}
    todo = [h for h in hours if not os.path.exists(path[h])]
    print(f"{len(hours)} hours {name(hours[0])} .. {name(hours[-1])} (UTC); {len(todo)} to download", flush=True)

    def get(h):
        part = path[h] + ".part"
        for _ in range(3):  # retry failed downloads
            if subprocess.run(["wget", "-q", "--tries=3", "--timeout=60", "-O", part, URL.format(name(h))]).returncode == 0:
                os.replace(part, path[h])
                return
        if os.path.exists(part):
            os.remove(part)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(get, todo))
    missing = [name(h) for h in hours if not os.path.exists(path[h])]
    print(f"done: {len(hours) - len(missing)} of {len(hours)} hours present"
          + (f"; missing on GH Archive or failed: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}" if missing else ""))


if __name__ == "__main__":
    main()
