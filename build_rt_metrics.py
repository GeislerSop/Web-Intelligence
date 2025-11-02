from pathlib import Path
import re
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs"
SRC = OUT_DIR / "rotten_tomatoes_movies.csv"
DST = OUT_DIR / "rt_metrics.csv"

def norm_title(t: str) -> str:
    if pd.isna(t):
        return ""
    t = str(t)
    # häufige Klammerjahre am Ende entfernen, z.B. "Movie (1994)"
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
    # Sonderzeichen raus, lower-case, Mehrfachspaces -> 1 Space
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(t.split())

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Quelle nicht gefunden: {SRC}")

    # Nur die Spalten ziehen, die wir wirklich brauchen
    usecols = [
        "movie_title",
        "original_release_date",
        "tomatometer_rating",
        "audience_rating",
        "tomatometer_count",
        "audience_count",
        "tomatometer_status",
        "audience_status",
    ]

    df = pd.read_csv(SRC, usecols=lambda c: c in usecols, low_memory=False)

    # Titel + Normalisierung
    df = df.rename(columns={"movie_title": "title"})
    df["title_norm"] = df["title"].map(norm_title)

    # Jahr aus Datum ziehen
    df["year"] = pd.to_datetime(df["original_release_date"], errors="coerce").dt.year

    # Bewertungen/Counts numerisch machen
    for c in ["tomatometer_rating", "audience_rating", "tomatometer_count", "audience_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Benennen wie in der App erwartet
    df = df.rename(
        columns={
            "tomatometer_rating": "rt_tomato",
            "audience_rating": "rt_audience",
        }
    )

    # Plausibilitätsfilter (Ratings 0..100)
    for c in ["rt_tomato", "rt_audience"]:
        df.loc[~df[c].between(0, 100, inclusive="both"), c] = np.nan

    # Ergebnis-Spalten zusammenstellen
    keep = [
        "title",
        "title_norm",
        "year",
        "rt_tomato",
        "rt_audience",
        "tomatometer_count",
        "audience_count",
        "tomatometer_status",
        "audience_status",
    ]
    df_out = df[keep].drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

    OUT_DIR.mkdir(exist_ok=True)
    df_out.to_csv(DST, index=False)
    print(f"✅ Geschrieben: {DST} — Zeilen: {len(df_out)}")
    print("Beispiel:")
    print(df_out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
