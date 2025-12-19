# fetch_wikidata_texts.py
# Lädt kurze Wikidata-Beschreibungen (schema:description) für Filme
# und speichert sie als CSV: outputs/text_wikidata_cache.csv

from __future__ import annotations
import certifi
import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("wikidata-fetch")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def read_titles_from_joined(joined_csv: Path, limit: int) -> list[str]:
    df = pd.read_csv(joined_csv)
    if "title" not in df.columns:
        raise ValueError("CSV muss eine Spalte 'title' enthalten.")
    titles = (
        df["title"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )
    return titles[:limit]


def chunk(lst: list[str], n: int) -> list[list[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def clean_title(title: str) -> str:
    """Entfernt problematische Zeichen für SPARQL"""
    return title.replace('"', "").replace("\n", " ").strip()


def fetch_wikidata_descriptions(titles: list[str], lang: str = "en") -> pd.DataFrame:
    if not titles:
        return pd.DataFrame(columns=["title", "wd_item", "description"])

    cleaned = [clean_title(t) for t in titles if t]

    values = " ".join([f'"{t}"@{lang}' for t in cleaned])

    query = f"""
    SELECT ?title ?item ?desc WHERE {{
      VALUES ?title {{ {values} }}
      ?item rdfs:label ?title .
      ?item wdt:P31/wdt:P279* wd:Q11424 .
      OPTIONAL {{
        ?item schema:description ?desc .
        FILTER(LANG(?desc) = "{lang}")
      }}
    }}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "movie-dashboard-portfolio/1.0"
    }

    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": query},
        headers=headers,
        timeout=30,
        verify=False
    )

    r.raise_for_status()
    data = r.json()

    rows = []
    for b in data.get("results", {}).get("bindings", []):
        rows.append({
            "title": b.get("title", {}).get("value", ""),
            "wd_item": b.get("item", {}).get("value", ""),
            "description": b.get("desc", {}).get("value", "")
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["title", "wd_item", "description"])

    return df.drop_duplicates(subset=["title"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined_csv", type=str, default="outputs/joined_imdb_rt.csv")
    ap.add_argument("--out_csv", type=str, default="outputs/text_wikidata_cache.csv")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--lang", type=str, default="en")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    joined = (base / args.joined_csv).resolve()
    out_csv = (base / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not joined.exists():
        raise FileNotFoundError(f"joined_csv nicht gefunden: {joined}")

    titles = read_titles_from_joined(joined, limit=args.limit)
    LOG.info(f"Titles loaded: {len(titles)}")

    all_rows = []
    for i, part in enumerate(chunk(titles, args.batch), start=1):
        LOG.info(f"Fetching batch {i} ({len(part)} titles)")
        try:
            df = fetch_wikidata_descriptions(part, lang=args.lang)
            all_rows.append(df)
        except Exception as e:
            LOG.exception(f"Batch {i} failed: {e}")

    if all_rows:
        res = pd.concat(all_rows, ignore_index=True)
    else:
        res = pd.DataFrame(columns=["title","wd_item","description"])

    res = res.drop_duplicates(subset=["title"])

    found = (res["description"].fillna("").str.strip() != "").sum()
    LOG.info(f"Rows total: {len(res)} | with description: {found}")

    res.to_csv(out_csv, index=False)
    LOG.info(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
