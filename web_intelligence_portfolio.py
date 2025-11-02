import os, io, gzip, json, re, time, random, zipfile, pathlib, datetime, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
BASE = Path('.').resolve()
RAW  = BASE / 'raw';  RAW.mkdir(exist_ok=True)
OUT  = BASE / 'outputs'; OUT.mkdir(exist_ok=True)

print('Run UTC:', datetime.datetime.now(datetime.timezone.utc).isoformat())

# --- Helpers ---

def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9 ]+"," ", t)
    return " ".join(t.split())


def save_fig(name: str):
    p = OUT / f"{name}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    print("Saved:", p)


def show_df(df: pd.DataFrame, n: int = 3, title: str | None = None):
    """Print a small preview of a DataFrame in script environments."""
    if df is None:
        print("<None>")
        return
    if title:
        print(title)
    try:
        print(df.head(n).to_string(index=False))
    except Exception:
        # Fallback without to_string if very wide
        print(df.head(n))


# 1) Rotten Tomatoes via Kaggle (wie in Colab) — KaggleHub **oder** Kaggle API, plus Local-Fallback
# --- Optional: Kaggle-Credentials inline setzen (wie in Colab) ---
# Tipp: Setze diese ENV-Variablen VOR dem Import/Authentifizieren.
# Du kannst sie auch außerhalb des Skripts setzen. Niemals echte Keys committen!
KAGGLE_USERNAME_INLINE = "sophiegeisler" #os.environ.get("KAGGLE_USERNAME_INLINE")  # oder "dein_user"
KAGGLE_KEY_INLINE      = "893a7a9b8250acf3aceb6d596c1e38b5" #os.environ.get("KAGGLE_KEY_INLINE")       # oder "dein_api_key"
if KAGGLE_USERNAME_INLINE and KAGGLE_KEY_INLINE:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME_INLINE
    os.environ["KAGGLE_KEY"]      = KAGGLE_KEY_INLINE

import kagglehub
from kagglehub import KaggleDatasetAdapter

rt_dataset_id = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
rt_file_path  = "rotten_tomatoes_movies.csv"  # falls anders: hier anpassen

print("Lade RT-Dataset …")


# Optionaler Local-Fallback: Wenn die CSV lokal vorhanden ist (oder via ENV angegeben), erst diese nutzen.
LOCAL_RT = os.environ.get("RT_CSV_PATH") or str((RAW/rt_file_path))

def try_read_local_rt(path: str):
    if not path or not Path(path).exists():
        return None
    print(f"Lese lokale RT-CSV: {path}")
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="warn")
        except Exception as e:
            print(f"Warnung: Lokales Lesen mit encoding='{enc}' fehlgeschlagen: {e}")
    try:
        return pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"Letzter Versuch (skip) fehlgeschlagen: {e}")
        return None

rt_df_raw = try_read_local_rt(LOCAL_RT)

if rt_df_raw is None:
    # 1) Versuch: KaggleHub dataset_load (benötigt Extras)
    try:
        def load_rt_df_kagglehub():
            encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
            for enc in encodings_to_try:
                try:
                    df = kagglehub.dataset_load(
                        KaggleDatasetAdapter.PANDAS,
                        rt_dataset_id,
                        rt_file_path,
                        pandas_kwargs={
                            "encoding": enc,
                            "encoding_errors": "replace",
                            "low_memory": False,
                            "engine": "python",
                            "on_bad_lines": "warn",
                            "sep": ",",
                            "quotechar": '"',
                            "doublequote": True,
                        },
                    )
                    print(f"KaggleHub: CSV gelesen mit encoding='{enc}'")
                    return df
                except Exception as e:
                    print(f"KaggleHub: Lesen mit '{enc}' fehlgeschlagen: {e}")
            # letzter Versuch mit skip
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                rt_dataset_id,
                rt_file_path,
                pandas_kwargs={
                    "encoding": "latin-1",
                    "encoding_errors": "replace",
                    "low_memory": False,
                    "engine": "python",
                    "on_bad_lines": "skip",
                },
            )
            print("KaggleHub: CSV gelesen mit Fallback (latin-1, skip)")
            return df
        rt_df_raw = load_rt_df_kagglehub()
    except Exception as e_hub:
        print("Hinweis: KaggleHub dataset_load nicht verfügbar oder fehlgeschlagen:", e_hub)
        print("Versuche stattdessen Kaggle API (wie in Colab)…")
        # 2) Versuch: Kaggle API (nutzt ENV KAGGLE_USERNAME/KAGGLE_KEY, die oben gesetzt werden können)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()  # nutzt ENV oder ~/.kaggle/kaggle.json
            # Lade gezielt die eine Datei in RAW/
            api.dataset_download_file(
                dataset=rt_dataset_id,
                file_name=rt_file_path,
                path=str(RAW),
                force=True,
            )
            # Die API lädt als .zip; entpacken
            zpath = RAW / f"{Path(rt_file_path).name}.zip"
            if zpath.exists():
                with zipfile.ZipFile(zpath, 'r') as zf:
                    zf.extractall(RAW)
                zpath.unlink(missing_ok=True)
            # Jetzt lokal einlesen
            local_csv = RAW/rt_file_path
            if not local_csv.exists():
                # manche Datasets haben Pfadpräfixe – suche die CSV
                cand = list(RAW.rglob(Path(rt_file_path).name))
                if cand:
                    local_csv = cand[0]
            rt_df_raw = try_read_local_rt(str(local_csv))
            if rt_df_raw is None:
                raise RuntimeError("Kaggle API: CSV gefunden, aber Einlesen schlug fehl")
            print("Kaggle API: CSV erfolgreich geladen und eingelesen")
        except Exception as e_api:
            raise RuntimeError(
                "Konnte RT-CSV nicht laden. Optionen: 1) pip install \"kagglehub[pandas-datasets]\" "
                "oder 2) Kaggle-API verwenden (ENV KAGGLE_USERNAME/KAGGLE_KEY setzen), oder 3) CSV manuell in raw/ legen."
                f"Letzte Fehler – KaggleHub: {e_hub} Kaggle API: {e_api}"
            )


print("RT-DF shape:", rt_df_raw.shape)
show_df(rt_df_raw, 3, title="Rotten Tomatoes (Rohdaten):")


# 2) IMDb Rohdaten (offiziell) – basics + ratings
IMDB_BASE = "https://datasets.imdbws.com"


def read_tsv_gz(name, usecols=None):
    url = f"{IMDB_BASE}/{name}"
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return pd.read_csv(
        io.BytesIO(gzip.decompress(r.content)),
        sep="\t", na_values="\\N",
        usecols=usecols, low_memory=False
    )


title_basics  = read_tsv_gz("title.basics.tsv.gz",
                            ["tconst","titleType","primaryTitle","startYear","runtimeMinutes","genres"])
title_ratings = read_tsv_gz("title.ratings.tsv.gz",
                            ["tconst","averageRating","numVotes"])

imdb_movies = title_basics[title_basics["titleType"]=="movie"].copy()
imdb_movies.rename(columns={"primaryTitle":"title","startYear":"year"}, inplace=True)
imdb_movies["year"]           = pd.to_numeric(imdb_movies["year"], errors="coerce")
imdb_movies["runtimeMinutes"] = pd.to_numeric(imdb_movies["runtimeMinutes"], errors="coerce")
imdb_movies["title_norm"]     = imdb_movies["title"].map(norm_title)

imdb_full = imdb_movies.merge(title_ratings, on="tconst", how="left")
print("IMDb movies:", imdb_full.shape)
show_df(imdb_full, 3, title="IMDb (Rohdaten, Filme):")


# 3) Rotten Tomatoes CSV standardisieren -> title_norm, year, rt_tomato (0-100), rt_audience (0-100)

def parse_year_from_text(s: str):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return float(m.group(0)) if m else np.nan


def std_rt_kagglehub(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lower = {c.lower(): c for c in d.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    c_title = pick("movie_title","title","name")
    c_year  = pick("original_release_year","year","release_year","original_release_date","dvd_release_date","theatrical_release_date")
    c_info  = pick("movie_info")
    c_tomato   = pick("tomatometer_rating","tomato_score","tomatometer","rotten_tomatoes","rt","rt_score")
    c_audience = pick("audience_rating","audience_score","audiencescore")

    keep = [c for c in [c_title, c_year, c_info, c_tomato, c_audience] if c]
    if not keep:
        return pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])
    d = d[keep].copy()

    ren = {}
    if c_title:   ren[c_title]   = "title"
    if c_year:    ren[c_year]    = "year_raw"
    if c_info:    ren[c_info]    = "movie_info_raw"
    if c_tomato:  ren[c_tomato]  = "rt_tomato_raw"
    if c_audience:ren[c_audience]= "rt_audience_raw"
    d.rename(columns=ren, inplace=True)

    d["title_norm"] = d["title"].map(norm_title) if "title" in d else ""

    year = pd.to_numeric(d.get("year_raw", pd.Series(index=d.index)), errors="coerce")
    if year.isna().all() and "movie_info_raw" in d:
        year = d["movie_info_raw"].map(parse_year_from_text)
    d["year"] = year

    def to_100(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().replace("%", "")
        try:
            v = float(s)
            if 0 <= v <= 10:
                v *= 10.0
            return v
        except Exception:
            return np.nan

    d["rt_tomato"]   = d.get("rt_tomato_raw",   pd.Series(index=d.index)).map(to_100)
    d["rt_audience"] = d.get("rt_audience_raw", pd.Series(index=d.index)).map(to_100)

    out = d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()
    return out


rt_df = std_rt_kagglehub(rt_df_raw)
print("RT std:", rt_df.shape, rt_df.columns.tolist())
show_df(rt_df, 3, title="RT (standardisiert):")


# 4) Join: IMDb × Rotten Tomatoes (Title + Year)
imdb_sel = imdb_full[["tconst","title","title_norm","year","averageRating","numVotes","genres"]].copy()
joined = imdb_sel.merge(rt_df, on=["title_norm","year"], how="left")
print("joined:", joined.shape, "| RT non-null:", joined["rt_tomato"].notna().sum())
show_df(joined.sample(5, random_state=0), 5, title="Join-Beispiele:")


# 5) Plots & Auswertungen

# 5.1 Durchschnitt nach Quelle
vals = {}
if "averageRating" in joined and joined["averageRating"].notna().any():
    vals["IMDb (x10)"] = joined["averageRating"].dropna().mean()*10
if "rt_tomato" in joined and joined["rt_tomato"].notna().any():
    vals["Rotten Tomatoes (Tomatometer)"] = joined["rt_tomato"].dropna().mean()
if "rt_audience" in joined and joined["rt_audience"].notna().any():
    vals["Rotten Tomatoes (Audience)"] = joined["rt_audience"].dropna().mean()

if vals:
    plt.figure(figsize=(8,5))
    plt.bar(list(vals.keys()), list(vals.values()))
    plt.title("Durchschnittliche Bewertung nach Quelle (0–100)")
    plt.ylabel("Score")
    save_fig("avg_by_source_rt_imdb")

# 5.2 Scatter IMDb vs Tomatometer
tmp = joined.dropna(subset=["averageRating","rt_tomato"]).copy()
if not tmp.empty:
    plt.figure(figsize=(6,6))
    plt.scatter(tmp["averageRating"]*10, tmp["rt_tomato"], alpha=0.5)
    plt.xlabel("IMDb (x10)")
    plt.ylabel("Rotten Tomatoes: Tomatometer (%)")
    plt.title("IMDb vs Rotten Tomatoes (Tomatometer)")
    x = (tmp["averageRating"]*10).values; y = tmp["rt_tomato"].values
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100); ys = m*xs + b
        plt.plot(xs, ys)
    save_fig("scatter_imdb_tomato")

# 5.3 Verteilungen
if not tmp.empty:
    plt.figure(figsize=(9,5))
    plt.hist(tmp["averageRating"]*10, bins=30, alpha=0.5, density=True, label="IMDb")
    plt.hist(tmp["rt_tomato"], bins=30, alpha=0.5, density=True, label="RT (Tomatometer)")
    plt.title("Bewertungsverteilung: IMDb vs Rotten Tomatoes (Tomatometer)")
    plt.xlabel("Score (0–100)"); plt.legend()
    save_fig("dist_imdb_tomato")

# 5.4 IMDb: Genre-Ø (≥50k Stimmen)
imdb_g = imdb_full[imdb_full["numVotes"].fillna(0) >= 50000].dropna(subset=["genres"]).copy()
if not imdb_g.empty:
    g = imdb_g.assign(genre=imdb_g["genres"].str.split(",")).explode("genre")
    avg_by_genre = (g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12)) * 10
    plt.figure(figsize=(9,5))
    plt.bar(avg_by_genre.index.tolist(), avg_by_genre.values)
    plt.title("IMDb (≥50k Stimmen): Ø Bewertung pro Genre (Top 12)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ø Bewertung (0–100)")
    save_fig("imdb_avg_by_genre")

# 5.5 IMDb: Ø nach Jahrzehnt
dec = imdb_full.dropna(subset=["year"]).copy()
if not dec.empty:
    dec["year"] = dec["year"].astype(int)
    dec["decade"] = (dec["year"]//10)*10
    dec_avg = (dec.groupby("decade")["averageRating"].mean().sort_index()) * 10
    plt.figure(figsize=(9,5))
    plt.plot(dec_avg.index.values, dec_avg.values, marker="o")
    plt.title("IMDb: Ø Bewertung nach Jahrzehnt")
    plt.xlabel("Jahrzehnt")
    plt.ylabel("Ø Bewertung (0–100)")
    save_fig("imdb_avg_by_decade")

# 5.6 Top-20 (IMDb)
top20 = imdb_full.sort_values("numVotes", ascending=False).loc[:, ["title","year","averageRating","numVotes"]].head(20)
top20.to_csv(OUT/"top20_by_votes_imdb.csv", index=False)
print("Top-20 saved.")

# 5.7 Export
joined.to_csv(OUT/"joined_imdb_rt.csv", index=False)
print("joined export:", OUT/"joined_imdb_rt.csv")

# 6) Google Trends (robust mit Retry/Backoff)
try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None
    print("Hinweis: pytrends nicht installiert – Trends werden übersprungen. Installiere mit 'pip install pytrends'.")


def google_trends_top5(imdb_df: pd.DataFrame, max_retries=4, wait_base=4.0):
    if TrendReq is None:
        return None, []
    pytrends = TrendReq(hl="de-DE", tz=0)
    top5 = imdb_df.sort_values("numVotes", ascending=False).dropna(subset=["title","year"]).head(5)
    kw_primary  = [f"{r['title']} {int(r['year'])}" for _, r in top5.iterrows()]
    kw_fallback = [f"{r['title']}" for _, r in top5.iterrows()]
    # primary
    for attempt in range(max_retries):
        try:
            pytrends.build_payload(kw_list=kw_primary, timeframe="today 5-y")
            d = pytrends.interest_over_time()
            if d is not None and not d.empty:
                return d, kw_primary
        except Exception as e:
            sleep_s = wait_base * (2**attempt) + random.uniform(0, 1.0)
            print(f"Trends Versuch {attempt+1} fehlgeschlagen. Warte {sleep_s:.1f}s … ({e})")
            time.sleep(sleep_s)
    # fallback
    for attempt in range(max_retries):
        try:
            pytrends.build_payload(kw_list=kw_fallback, timeframe="today 5-y")
            d = pytrends.interest_over_time()
            if d is not None and not d.empty:
                return d, kw_fallback
        except Exception as e:
            sleep_s = wait_base * (2**attempt) + random.uniform(0, 1.0)
            print(f"Trends Fallback-Versuch {attempt+1} fehlgeschlagen. Warte {sleep_s:.1f}s … ({e})")
            time.sleep(sleep_s)
    return None, kw_primary


d_trends, kw_used = google_trends_top5(imdb_full, max_retries=4, wait_base=4.0)
if d_trends is not None and hasattr(d_trends, 'empty') and not d_trends.empty:
    d_trends.reset_index().to_csv(OUT/"google_trends_top5.csv", index=False)
    plt.figure(figsize=(10,5))
    for c in kw_used:
        if c in d_trends.columns:
            plt.plot(d_trends.index, d_trends[c], label=c)  # Fix: d_trends statt d
    plt.title("Google Trends: Suchinteresse (Top 5 IMDb nach Stimmen)")
    plt.xlabel("Datum"); plt.ylabel("Relatives Interesse (0–100)"); plt.legend()
    save_fig("google_trends_top5")
else:
    print("⚠️ Google Trends: Keine Daten nach Retries oder pytrends nicht verfügbar.")

# 7) Übersicht der erzeugten Artefakte
print("Erzeugte Dateien im outputs/-Ordner:")
for f in sorted(OUT.iterdir()):
    print(" -", f.name)
