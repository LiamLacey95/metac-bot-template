"""Verify every OpenRouter model slug in main.py actually exists.

Needs no API key - the model list is public. Run after editing any model, and in CI.

A wrong slug is the expensive kind of bug: it does not crash, it removes one model family from
the ensemble and every subsequent forecast is quietly worse. `google/gemini-3.1-pro` looks
entirely reasonable and does not exist; the real slug is `google/gemini-3.1-pro-preview`.

This checks the ROLE models - researcher, summarizer, parser - as well as the ensemble, because
the ensemble was never where the damage came from. A dead researcher slug stops the bot outright,
and a parser that resolves but cannot extract a schema is worse than either: it fails silently on
every question while looking like a working run.

    python check_models.py     # exit 0 if all slugs resolve, 1 otherwise
"""

from __future__ import annotations

import json
import sys
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_available() -> set[str]:
    req = urllib.request.Request(
        MODELS_URL, headers={"User-Agent": "Mozilla/5.0 (model-slug-check)"}
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {m["id"] for m in data.get("data", [])}


def main() -> int:
    # Imported lazily so this script runs without the forecasting-tools dependency installed.
    import re
    from pathlib import Path

    source = (Path(__file__).parent / "main.py").read_text(encoding="utf-8")
    block = re.search(r"ENSEMBLE_MODELS = \[(.*?)\]", source, re.S)
    if not block:
        print("Could not find ENSEMBLE_MODELS in main.py")
        return 1
    slugs = re.findall(r'"([^"]+)"', block.group(1))

    # Every other paid `openrouter/...` string in the file - the researcher, summarizer and
    # parser. Free-tier slugs end in `:free` and are a separate, deliberately disposable path.
    roles = [
        s
        for s in re.findall(r'"(openrouter/[^"]+)"', source)
        if s not in slugs and not s.endswith(":free")
    ]
    slugs += sorted(set(roles))

    available = fetch_available()
    print(f"OpenRouter lists {len(available)} models.\n")

    ok = True
    for slug in slugs:
        # The bot prefixes slugs with the litellm provider; OpenRouter's own ids omit it.
        bare = slug.removeprefix("openrouter/")
        exists = bare in available
        print(f"  {'OK  ' if exists else 'FAIL'}  {slug}")
        if not exists:
            ok = False
            near = sorted(m for m in available if m.split("/")[0] == bare.split("/")[0])[:8]
            print(f"        not on OpenRouter. Same provider: {near}")

    print()
    if ok:
        print(f"All {len(slugs)} model slugs resolve.")
        return 0
    print("At least one slug is wrong. Fix it in main.py before running the bot.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
