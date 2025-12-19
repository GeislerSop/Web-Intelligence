from __future__ import annotations

from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parent

# ---- Eingaben: Wir versuchen mehrere typische Pfade ----
JOINED = next((p for p in [
    BASE/"outputs/joined_imdb_rt.csv",
    BASE/"joined_imdb_rt.csv",
] if p.exists()), None)

# Optional: IMDb "title.principals" + "name.basics" (falls du die hast)
TITLE_PRINCIPALS = next((p for p in [
    BASE/"outputs/title_principals.csv",
    BASE/"raw/title_principals.csv",
    BASE/"title_principals.csv",
    BASE/"outputs/title.principals.tsv",
    BASE/"raw/title.principals.tsv",
] if p.exists()), None)

NAME_BASICS = next((p for p in [
    BASE/"outputs/name_basics.csv",
    BASE/"raw/name_basics.csv",
    BASE/"name_basics.csv",
    BASE/"outputs/name.basics.tsv",
    BASE/"raw/name.basics.tsv",
] if p.exists()), None)

OUT = BASE/"outputs/edges.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def norm_text(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+"," ", str(t).lower())
    return " ".join(t.split())

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)

if JOINED is None:
    raise FileNotFoundError("joined_imdb_rt.csv nicht gefunden (outputs/ oder root).")

df = pd.read_csv(JOINED)

# ---- Title-Spalte finden ----
if "title" not in df.columns:
    # Fallbacks
    for c in ["primaryTitle", "Title", "movie_title", "name"]:
        if c in df.columns:
            df = df.rename(columns={c: "title"})
            break
if "title" not in df.columns:
    raise ValueError("Keine Titelspalte gefunden (erwarte 'title' oder ähnlich).")

# ---- tconst finden (für Actor-Merge super wichtig) ----
if "tconst" not in df.columns:
    for c in ["imdb_id", "titleId", "const"]:
        if c in df.columns:
            df = df.rename(columns={c: "tconst"})
            break

# ---- Genres finden ----
if "genres" not in df.columns:
    for c in ["genre", "Genres"]:
        if c in df.columns:
            df = df.rename(columns={c: "genres"})
            break
if "genres" not in df.columns:
    raise ValueError("Keine Genres-Spalte gefunden (erwarte 'genres' oder ähnlich).")

movies = df[["title"] + (["tconst"] if "tconst" in df.columns else []) + ["genres"]].copy()
movies["title"] = movies["title"].astype(str)
movies["title_norm"] = movies["title"].map(norm_text)

# ---- Movie → Genre edges ----
mg = movies.dropna(subset=["genres"]).copy()
mg["genre"] = mg["genres"].astype(str).str.split(",")
mg = mg.explode("genre")
mg["genre"] = mg["genre"].astype(str).str.strip()
mg = mg[mg["genre"].ne("")]

edges_mg = pd.DataFrame({
    "source": mg["title"],
    "target": mg["genre"],
    "relation": "has_genre",
    "source_type": "movie",
    "target_type": "genre",
})
edges_mg["source_norm"] = edges_mg["source"].map(norm_text)
edges_mg["target_norm"] = edges_mg["target"].map(norm_text)

# ==========================================================
# Schauspieler (2 Wege):
# A) Wenn joined eine Actor-Spalte hat (cast/stars/actors)
# B) Sonst über IMDb title.principals + name.basics (falls vorhanden)
# ==========================================================

actor_df = None

# ---- Weg A: Actor-Spalte direkt im joined suchen ----
actor_cols = [c for c in df.columns if c.lower() in {"cast", "stars", "actors", "actor", "performers"}]
if actor_cols:
    c = actor_cols[0]
    tmp = df[["title"] + (["tconst"] if "tconst" in df.columns else []) + [c]].copy()
    tmp = tmp.dropna(subset=[c])
    tmp[c] = tmp[c].astype(str)

    # Trennzeichen: probiere ; | , (robust)
    # Wir splitten zunächst an ; oder |, sonst fallback an , wenn viele Namen drin
    def split_names(s: str):
        if "|" in s:
            parts = s.split("|")
        elif ";" in s:
            parts = s.split(";")
        else:
            parts = s.split(",")
        return [p.strip() for p in parts if p.strip()]

    tmp["actor"] = tmp[c].map(split_names)
    tmp = tmp.explode("actor")
    tmp = tmp[tmp["actor"].notna() & tmp["actor"].astype(str).str.strip().ne("")]
    actor_df = tmp.rename(columns={"actor": "primaryName"})

# ---- Weg B: IMDb TSV/CSV title.principals + name.basics ----
if actor_df is None and ("tconst" in df.columns) and (TITLE_PRINCIPALS is not None) and (NAME_BASICS is not None):
    tp = read_any(TITLE_PRINCIPALS)
    nb = read_any(NAME_BASICS)

    # Spalten normalisieren
    # title.principals: tconst, nconst, category, ordering
    if "tconst" not in tp.columns:
        # bei manchen Exporten heißt es titleId
        for c in ["titleId"]:
            if c in tp.columns:
                tp = tp.rename(columns={c: "tconst"})
    if "nconst" not in tp.columns:
        for c in ["nameId"]:
            if c in tp.columns:
                tp = tp.rename(columns={c: "nconst"})
    if "category" not in tp.columns:
        for c in ["job", "role"]:
            if c in tp.columns:
                tp = tp.rename(columns={c: "category"})

    if "primaryName" not in nb.columns:
        for c in ["name", "Name"]:
            if c in nb.columns:
                nb = nb.rename(columns={c: "primaryName"})
    if "nconst" not in nb.columns:
        for c in ["nameId"]:
            if c in nb.columns:
                nb = nb.rename(columns={c: "nconst"})

    need_tp = {"tconst", "nconst", "category"}
    need_nb = {"nconst", "primaryName"}
    if need_tp.issubset(tp.columns) and need_nb.issubset(nb.columns):
        # nur actors/actress
        tp2 = tp[tp["category"].isin(["actor", "actress"])].copy()

        # optional: pro Film begrenzen (sonst riesig)
        if "ordering" in tp2.columns:
            tp2["ordering"] = pd.to_numeric(tp2["ordering"], errors="coerce")
            tp2 = tp2.sort_values(["tconst", "ordering"])
            tp2 = tp2.groupby("tconst").head(12)

        tp2 = tp2.merge(nb[["nconst", "primaryName"]], on="nconst", how="left")
        tp2 = tp2.dropna(subset=["primaryName"])

        # mit joined-Filmen verbinden (damit wir title haben)
        film_map = df[["tconst", "title"]].dropna().drop_duplicates()
        actor_df = tp2.merge(film_map, on="tconst", how="inner")[["title", "tconst", "primaryName"]].copy()

# ---- Wenn wir keine Actor-Daten bekommen: nur Movie-Genre edges raus ----
edges_all = [edges_mg]

if actor_df is not None and not actor_df.empty:
    actor_df["primaryName"] = actor_df["primaryName"].astype(str).str.strip()
    actor_df = actor_df[actor_df["primaryName"].ne("")]


    edges_ma = pd.DataFrame({
        "source": actor_df["title"],
        "target": actor_df["primaryName"],
        "relation": "has_actor",
        "source_type": "movie",
        "target_type": "actor",
    })
    edges_ma["source_norm"] = edges_ma["source"].map(norm_text)
    edges_ma["target_norm"] = edges_ma["target"].map(norm_text)
    edges_all.append(edges_ma)

    # Actor → Genre (über Filme)
    # join: actor_df (title) + mg (title→genre)
    mg_for_join = mg[["title", "genre"]].dropna().copy()
    ag = actor_df.merge(mg_for_join, on="title", how="inner")

    # Optional: Gewichte = wie oft Actor in Genre vorkommt
    ag_counts = (
        ag.groupby(["primaryName", "genre"])
          .size()
          .reset_index(name="weight")
          .sort_values(["primaryName", "weight"], ascending=[True, False])
    )

    # Optional: pro Actor nur Top-N Genres, damit Graph lesbar bleibt
    TOPN = 8
    ag_counts = ag_counts.groupby("primaryName").head(TOPN)

    edges_ag = pd.DataFrame({
        "source": ag_counts["primaryName"],
        "target": ag_counts["genre"],
        "relation": "acts_in_genre",
        "weight": ag_counts["weight"],
        "source_type": "actor",
        "target_type": "genre",
    })
    edges_ag["source_norm"] = edges_ag["source"].map(norm_text)
    edges_ag["target_norm"] = edges_ag["target"].map(norm_text)
    edges_all.append(edges_ag)

# ---- final speichern ----
edges_out = pd.concat(edges_all, ignore_index=True)
edges_out = edges_out.drop_duplicates(subset=["source","target","relation"])
edges_out.to_csv(OUT, index=False)

print(f"✅ saved {OUT} shape={edges_out.shape}")
print("Relations:", edges_out["relation"].value_counts().to_dict())
