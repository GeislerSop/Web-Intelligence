from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parent
JOINED = next((p for p in [
    BASE/"outputs/joined_imdb_rt.csv",
    BASE/"joined_imdb_rt.csv",
] if p.exists()), None)

OUT = BASE/"outputs/edges.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def norm_title(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+"," ", str(t).lower())
    return " ".join(t.split())

if JOINED is None:
    raise FileNotFoundError("joined_imdb_rt.csv nicht gefunden (outputs/ oder root).")

df = pd.read_csv(JOINED)
if "title" not in df.columns:
    raise ValueError("Spalte 'title' fehlt in joined_imdb_rt.csv")

# Genres finden (dein File hat meist 'genres' als comma-separated)
if "genres" not in df.columns:
    raise ValueError("Spalte 'genres' fehlt in joined_imdb_rt.csv (für Movie↔Genre Edges).")

d = df[["title","genres"]].dropna()
d["title"] = d["title"].astype(str)

# split + explode
d["genre"] = d["genres"].astype(str).str.split(",")
d = d.explode("genre")
d["genre"] = d["genre"].astype(str).str.strip()
d = d[d["genre"].ne("")]

edges = pd.DataFrame({
    "source": d["title"],
    "target": d["genre"],
    "relation": "has_genre"
})

# optional: norm spalten (hilft beim Matching in der App)
edges["source_norm"] = edges["source"].map(norm_title)
edges["target_norm"] = edges["target"].map(norm_title)

edges = edges.drop_duplicates()
edges.to_csv(OUT, index=False)
print(f"✅ saved {OUT} shape={edges.shape}")
