"""Pull resolved binary questions from Metaculus to test assumptions against real data.

The calibration layer shrinks toward a prior of 0.35, chosen on the reasoning that most
"will X happen by date Y" questions resolve No. That was an assumption. This measures it.

Reads METACULUS_TOKEN from the environment. Never hard-code the token.

    python fetch_resolved.py --pages 12
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter

API = "https://www.metaculus.com/api/posts/"
OUT = "resolved_binary.json"


def fetch_page(token: str, offset: int, limit: int = 100) -> dict:
    url = (
        f"{API}?statuses=resolved&forecast_type=binary"
        f"&limit={limit}&offset={offset}&order_by=-resolve_time"
    )
    # Metaculus 403s the default python-urllib user agent, with the token attached and all.
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {token}",
            "User-Agent": "Mozilla/5.0 (calibration-research-script)",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        raise SystemExit("METACULUS_TOKEN not set in environment")

    rows = []
    for page in range(args.pages):
        offset = page * args.limit
        try:
            data = fetch_page(token, offset, args.limit)
        except urllib.error.HTTPError as e:
            print(f"  page {page}: HTTP {e.code}, stopping")
            break
        results = data.get("results") or []
        if not results:
            break
        for post in results:
            q = post.get("question") or {}
            if q.get("type") != "binary":
                continue
            rows.append(
                {
                    "id": post.get("id"),
                    "title": post.get("title"),
                    "resolution": q.get("resolution"),
                    "score_type": q.get("default_score_type"),
                    "agg_method": q.get("default_aggregation_method"),
                    "resolve_time": q.get("actual_resolve_time"),
                    "nr_forecasters": post.get("nr_forecasters"),
                }
            )
        print(f"  page {page}: +{len(results)} (total {len(rows)})", flush=True)
        if not data.get("next"):
            break
        time.sleep(0.4)  # be polite to the API

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    print(f"\nSaved {len(rows)} rows to {OUT}\n")

    res = Counter(str(r["resolution"]) for r in rows)
    print("resolution values:")
    for k, v in res.most_common():
        print(f"  {k!r:12} {v:5d}")

    yes = res.get("yes", 0)
    no = res.get("no", 0)
    decided = yes + no
    if decided:
        print(f"\nBASE RATE over {decided} cleanly resolved binary questions: "
              f"{yes / decided:.4f} YES")
        print(f"  (calibration.py currently assumes prior = 0.35)")

    print("\nscore types:", dict(Counter(str(r['score_type']) for r in rows)))
    print("aggregation methods:", dict(Counter(str(r['agg_method']) for r in rows)))


if __name__ == "__main__":
    main()
