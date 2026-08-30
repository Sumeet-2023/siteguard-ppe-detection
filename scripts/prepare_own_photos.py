"""Search Wikimedia Commons for candidate helmet/head photos and download
CC-licensed real photographs for manual labeling review.

Used as a substitute for the project spec's "shoot 60-80 photos yourself"
step, when no camera/site access is available. See reports/own_photos/README.md
for why this is a materially weaker substitute and the labeling methodology
caveat -- this script only gets you candidate images, not ground truth.

Wikimedia's API rate-limits aggressively without a descriptive User-Agent;
this one identifies the project per their UA policy.
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "SiteGuardResearch/1.0 (educational PPE-detection project; no contact configured)"

DEFAULT_QUERIES = [
    "construction worker hard hat safety",
    "site worker helmet",
    "warehouse worker",
    "farm worker head",
    "factory worker",
]

BAD_TITLE_PAT = re.compile(
    r"icon|logo|diagram|drawing|map|flag|coat of arms|clipart|cartoon|sign\.|symbol",
    re.I,
)
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
FREE_LICENSES = ("cc0", "cc-by", "cc by", "public domain", "pd-")


def api_get(params: dict, retries: int = 3) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** (attempt + 2))
                continue
            raise


def search(query: str, limit: int) -> list[str]:
    data = api_get({
        "action": "query", "list": "search", "srsearch": query,
        "srnamespace": 6, "srlimit": limit, "format": "json",
    })
    return [hit["title"] for hit in data["query"]["search"]]


def get_info(titles: list[str]) -> dict:
    data = api_get({
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|extmetadata|size", "iiurlwidth": 800, "format": "json",
    })
    return data["query"]["pages"]


def is_free_license(extmeta: dict) -> bool:
    lic = extmeta.get("LicenseShortName", {}).get("value", "").lower()
    return any(tag in lic for tag in FREE_LICENSES)


def find_candidates(queries: list[str], limit_per_query: int) -> list[dict]:
    seen_titles: set[str] = set()
    candidates = []
    for q in queries:
        try:
            titles = search(q, limit_per_query)
        except Exception as e:
            print(f"search failed for {q!r}: {e}")
            continue
        new_titles = [t for t in titles if t not in seen_titles]
        seen_titles.update(new_titles)
        if not new_titles:
            continue
        try:
            pages = get_info(new_titles)
        except Exception as e:
            print(f"imageinfo failed for {q!r}: {e}")
            continue

        for page in pages.values():
            title = page.get("title", "")
            if BAD_TITLE_PAT.search(title):
                continue
            info = page.get("imageinfo")
            if not info:
                continue
            info = info[0]
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if ext not in ALLOWED_EXT:
                continue
            if info.get("width", 0) < 400 or info.get("height", 0) < 300:
                continue
            extmeta = info.get("extmetadata", {})
            if not is_free_license(extmeta):
                continue
            candidates.append({"title": title, "url": url, "query": q,
                                "license": extmeta.get("LicenseShortName", {}).get("value", "?")})
        time.sleep(0.5)
    return candidates


def download(candidates: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, c in enumerate(candidates):
        url_ext = Path(urllib.parse.urlparse(c["url"]).path).suffix.lower()
        fname = f"commons_{i:03d}{url_ext}"
        dst = out_dir / fname
        try:
            req = urllib.request.Request(c["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r, open(dst, "wb") as f:
                f.write(r.read())
        except Exception as e:
            print(f"download failed {c['url']}: {e}")
            continue
        manifest.append({**c, "file": fname})
        time.sleep(0.5)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("data/own_photos"))
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--limit-per-query", type=int, default=15)
    a = ap.parse_args()

    candidates = find_candidates(a.queries, a.limit_per_query)
    print(f"{len(candidates)} candidates passed filtering")

    manifest = download(candidates, a.out_root / "raw")
    (a.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Downloaded {len(manifest)} images to {a.out_root / 'raw'}")
    print("Next: manually review each image (irrelevant results are common -- "
          "free-text search on Commons is noisy), then run find_failure_cases.py-style "
          "model-assisted labeling. See reports/own_photos/README.md for the methodology.")


if __name__ == "__main__":
    main()
