# app.py
# 🎬 Web-Intelligence: IMDb × Rotten Tomatoes (Kagglehub) — Shiny for Python
# Interaktive Auswertung + Visualisierungen + CSV-Download
# ----------------------------------------------------------
# Anforderungen:
#   pip install shiny pandas numpy matplotlib requests kagglehub
# Optional: Setze die Umgebungsvariable RT_DATASET_FILE auf eine andere Datei im Kaggle-Dataset
# ----------------------------------------------------------

from __future__ import annotations
import io, gzip, re, os, json, datetime, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from shiny import App, render, ui, reactive, Inputs, Outputs, Session

# ---------- Daten-Beschaffung ----------

IMDB_BASE = "https://datasets.imdbws.com"
RT_DATASET_ID = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
RT_FILE_DEFAULT = os.getenv("RT_DATASET_FILE", "rotten_tomatoes_movies.csv")

def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9 ]+"," ", t)
    return " ".join(t.split())

def read_imdb_tsv(name: str, usecols=None) -> pd.DataFrame:
    url = f"{IMDB_BASE}/{name}"
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return pd.read_csv(
        io.BytesIO(gzip.decompress(r.content)),
        sep="\t", na_values="\\N", usecols=usecols, low_memory=False
    )

def parse_year_from_text(s: str):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return float(m.group(0)) if m else np.nan

def to_100(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%","")
    try:
        v = float(s)
        if 0 <= v <= 10: v *= 10.0
        return v
    except:
        return np.nan

def std_rt_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lower = {c.lower(): c for c in d.columns}
    def pick(*names):
        for n in names:
            if n in lower: return lower[n]
        return None

    c_title   = pick("movie_title","title","name")
    c_year    = pick("original_release_year","year","release_year","original_release_date","dvd_release_date","theatrical_release_date")
    c_info    = pick("movie_info")
    c_tomato  = pick("tomatometer_rating","tomato_score","tomatometer","rotten_tomatoes","rt","rt_score")
    c_aud     = pick("audience_rating","audience_score","audiencescore")

    keep = [c for c in [c_title, c_year, c_info, c_tomato, c_aud] if c]
    if not keep:
        return pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])
    d = d[keep].copy()

    ren = {}
    if c_title:  ren[c_title]  = "title"
    if c_year:   ren[c_year]   = "year_raw"
    if c_info:   ren[c_info]   = "movie_info_raw"
    if c_tomato: ren[c_tomato] = "rt_tomato_raw"
    if c_aud:    ren[c_aud]    = "rt_audience_raw"
    d.rename(columns=ren, inplace=True)

    d["title_norm"] = d["title"].map(norm_title) if "title" in d else ""

    year = pd.to_numeric(d.get("year_raw", pd.Series(index=d.index)), errors="coerce")
    if year.isna().all() and "movie_info_raw" in d:
        year = d["movie_info_raw"].map(parse_year_from_text)
    d["year"] = year

    d["rt_tomato"]   = d.get("rt_tomato_raw",   pd.Series(index=d.index)).map(to_100)
    d["rt_audience"] = d.get("rt_audience_raw", pd.Series(index=d.index)).map(to_100)

    out = d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()
    return out

def load_rt_via_kagglehub(dataset_id=RT_DATASET_ID, file_path=RT_FILE_DEFAULT) -> pd.DataFrame:
    # Lazy import, damit App ohne kagglehub starten kann (bei rein manueller Datei)
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as e:
        raise RuntimeError("kagglehub ist nicht installiert. Bitte: pip install kagglehub") from e

    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_id,
            file_path,
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Kagglehub-Download fehlgeschlagen: {e}")

def load_data(initial_rt: pd.DataFrame | None = None):
    # IMDb
    basics  = read_imdb_tsv("title.basics.tsv.gz",  ["tconst","titleType","primaryTitle","startYear","runtimeMinutes","genres"])
    ratings = read_imdb_tsv("title.ratings.tsv.gz", ["tconst","averageRating","numVotes"])
    movies  = basics[basics["titleType"]=="movie"].copy()
    movies.rename(columns={"primaryTitle":"title","startYear":"year"}, inplace=True)
    movies["year"]           = pd.to_numeric(movies["year"], errors="coerce")
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
    movies["title_norm"]     = movies["title"].map(norm_title)
    imdb_full = movies.merge(ratings, on="tconst", how="left")

    # RT
    if initial_rt is None:
        try:
            rt_raw = load_rt_via_kagglehub()
        except Exception as e:
            rt_raw = pd.DataFrame()  # Fallback: UI erlaubt Upload
            print("Hinweis:", e)
    else:
        rt_raw = initial_rt

    rt_std = std_rt_kaggle(rt_raw) if not rt_raw.empty else pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])

    # Join (IMDb als Basis)
    imdb_sel = imdb_full[["tconst","title","title_norm","year","averageRating","numVotes","genres"]].copy()
    joined   = imdb_sel.merge(rt_std, on=["title_norm","year"], how="left")
    return imdb_full, rt_std, joined

# ---------- Shiny UI ----------

app_ui = ui.page_fluid(
    ui.panel_title("🎬 Web-Intelligence: IMDb × Rotten Tomatoes (Kagglehub)"),
    ui.row(
        ui.column(4,
            ui.card(
                ui.card_header("Daten & Filter"),
                ui.input_slider("year_range", "Jahre", min=1920, max=2025, value=(1980, 2025), step=1),
                ui.input_numeric("min_votes", "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
                ui.input_checkbox("use_audience", "Audience-Score anzeigen (zusätzlich zu Tomatometer)", False),
                ui.input_action_button("reload_kaggle", "RT aus Kagglehub neu laden"),
                ui.input_file("rt_upload", "Oder: Rotten-Tomatoes CSV manuell hochladen", accept=[".csv"], multiple=False),
                ui.tags.small("Tipp: Wenn Kagglehub blockiert ist, kannst du hier eine CSV aus dem Kaggle-Dataset hochladen.")
            ),
            ui.card(
                ui.card_header("Downloads"),
                ui.download_button("dl_joined", "Ergebnistabelle (CSV) herunterladen")
            ),
        ),
        ui.column(8,
            ui.navset_tab(
                ui.nav(
                    "Überblick",
                    ui.layout_column_wrap(
                        ui.value_box("n_movies", "Filme (gefiltert)"),
                        ui.value_box("avg_imdb", "Ø IMDb (x10)"),
                        ui.value_box("avg_rt", "Ø RT Tomatometer"),
                        fill=False
                    ),
                    ui.output_plot("plot_avg_by_source", height="350px"),
                ),
                ui.nav(
                    "IMDb ↔ Rotten Tomatoes",
                    ui.output_plot("plot_scatter", height="380px"),
                    ui.output_plot("plot_dist", height="380px"),
                ),
                ui.nav(
                    "Genres & Jahrzehnte",
                    ui.output_plot("plot_genre", height="380px"),
                    ui.output_plot("plot_decade", height="380px"),
                ),
                ui.nav(
                    "Tabelle",
                    ui.output_data_frame("table_movies")
                )
            )
        )
    ),
    ui.tags.hr(),
    ui.tags.small(
        "Daten: IMDb (offizielle TSVs) & Rotten Tomatoes (Kagglehub). ",
        "Bei RT-Problemen CSV hochladen. ",
        f"Build: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z"
    )
)

# ---------- Shiny Server ----------

def server(input: Inputs, output: Outputs, session: Session):

    # Reactive: globaler Datensatz (mit optionalem manuellem Upload)
    rt_manual = reactive.Value(pd.DataFrame())

    @reactive.effect
    @reactive.event(input.rt_upload)
    def _on_upload():
        f = input.rt_upload()
        if f is None:
            return
        try:
            df = pd.read_csv(f[0]["datapath"])
            rt_manual.set(df)
        except Exception as e:
            print("Upload-Fehler:", e)

    @reactive.Calc
    def data_all():
        manual = rt_manual.get()
        if manual is not None and not manual.empty:
            imdb, rt, joined = load_data(initial_rt=manual)
        else:
            imdb, rt, joined = load_data()
        return imdb, rt, joined

    # Gefilterte Sicht
    @reactive.Calc
    def data_filtered():
        imdb, rt, joined = data_all()
        y1, y2 = input.year_range()
        mv = input.min_votes()
        df = joined.copy()
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
        df = df[df["numVotes"].fillna(0) >= mv]
        return df

    # KPIs
    @output
    @render.value_box
    def n_movies():
        df = data_filtered()
        return ui.value_box(showcase=None, title="Filme (gefiltert)", value=f"{len(df):,}")

    @output
    @render.value_box
    def avg_imdb():
        df = data_filtered()
        val = (df["averageRating"].dropna().mean()*10) if not df["averageRating"].dropna().empty else np.nan
        txt = f"{val:.1f}" if pd.notna(val) else "—"
        return ui.value_box(showcase=None, title="Ø IMDb (x10)", value=txt)

    @output
    @render.value_box
    def avg_rt():
        df = data_filtered()
        val = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns else np.nan
        txt = f"{val:.1f}" if pd.notna(val) else "—"
        return ui.value_box(showcase=None, title="Ø RT Tomatometer", value=txt)

    # Plots
    def _plt_to_obj(fig):
        # Shiny zeigt matplotlib-Fig direkt; Helper falls nötig
        return fig

    @output
    @render.plot
    def plot_avg_by_source():
        df = data_filtered()
        vals = {}
        if "averageRating" in df:
            v = df["averageRating"].dropna()
            if not v.empty: vals["IMDb (x10)"] = v.mean()*10
        if "rt_tomato" in df:
            r = df["rt_tomato"].dropna()
            if not r.empty: vals["RT Tomatometer"] = r.mean()
        if input.use_audience() and "rt_audience" in df:
            a = df["rt_audience"].dropna()
            if not a.empty: vals["RT Audience"] = a.mean()
        if not vals:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.text(0.5,0.5,"Keine Daten für Balkendiagramm", ha="center", va="center")
            ax.axis("off")
            return _plt_to_obj(fig)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_title("Durchschnittliche Bewertung nach Quelle (0–100)")
        ax.set_ylabel("Score")
        ax.set_ylim(0,100)
        return _plt_to_obj(fig)

    @output
    @render.plot
    def plot_scatter():
        df = data_filtered()
        if "rt_tomato" not in df.columns:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Kein RT verfügbar", ha="center", va="center"); ax.axis("off")
            return _plt_to_obj(fig)
        tmp = df.dropna(subset=["averageRating","rt_tomato"]).copy()
        if tmp.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Schnittmenge IMDb × RT", ha="center", va="center"); ax.axis("off")
            return _plt_to_obj(fig)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(tmp["averageRating"]*10, tmp["rt_tomato"], alpha=0.5)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer (%)")
        ax.set_title("IMDb vs Rotten Tomatoes")
        # einfache Trendlinie
        x = (tmp["averageRating"]*10).values; y = tmp["rt_tomato"].values
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100); ys = m*xs + b
            ax.plot(xs, ys)
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        return _plt_to_obj(fig)

    @output
    @render.plot
    def plot_dist():
        df = data_filtered()
        if "rt_tomato" not in df.columns:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Kein RT verfügbar", ha="center", va="center"); ax.axis("off")
            return _plt_to_obj(fig)
        tmp = df.dropna(subset=["averageRating","rt_tomato"]).copy()
        if tmp.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Daten für Verteilung", ha="center", va="center"); ax.axis("off")
            return _plt_to_obj(fig)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(tmp["averageRating"]*10, bins=30, alpha=0.5, density=True, label="IMDb")
        ax.hist(tmp["rt_tomato"], bins=30, alpha=0.5, density=True, label="RT Tomatometer")
        if input.use_audience() and "rt_audience" in tmp:
            if tmp["rt_audience"].notna().any():
                ax.hist(tmp["rt_audience"], bins=30, alpha=0.3, density=True, label="RT Audience")
        ax.set_title("Bewertungsverteilung")
        ax.set_xlabel("Score (0–100)"); ax.legend()
        ax.set_xlim(0,100)
        return _plt_to_obj(fig)

    @output
    @render.plot
    def plot_genre():
        imdb, rt, joined = data_all()
        imdb_g = imdb[imdb["numVotes"].fillna(0) >= 50000].dropna(subset=["genres"]).copy()
        if imdb_g.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Genre-Daten", ha="center", va="center"); ax.axis("off")
            return _plt_to_obj(fig)
        g = imdb_g.assign(genre=imdb_g["genres"].str.split(",")).explode("genre")
        avg_by_genre = (g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12))*10
        fig, ax = plt.subplots(figsize=(9,4))
        ax.bar(avg_by_genre.index.tolist(), avg_by_genre.values)
        ax.set_title("IMDb (≥50k Stimmen): Ø Bewertung pro Genre (Top 12)")
        ax.set_xticklabels(avg_by_genre.index.tolist(), rotation=45, ha="right")
        ax.set_ylabel("Ø Bewertung (0–100)"); ax.set_ylim(0,100)
        return _plt_to_obj(fig)

    @output
    @render.plot
    def plot_decade():
        imdb, rt, joined = data_all()
        dec = imdb.dropna(subset=["year"]).copy()
        if dec.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Jahresdaten", ha="center", va="center"); ax.axis("off")
            return _plt_to_obj(fig)
        dec["year"] = dec["year"].astype(int)
        dec["decade"] = (dec["year"]//10)*10
        dec_avg = (dec.groupby("decade")["averageRating"].mean().sort_index())*10
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(dec_avg.index.values, dec_avg.values, marker="o")
        ax.set_title("IMDb: Ø Bewertung nach Jahrzehnt")
        ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø Bewertung (0–100)")
        ax.set_ylim(0,100)
        return _plt_to_obj(fig)

    @output
    @render.data_frame
    def table_movies():
        df = data_filtered().copy()
        cols = ["title","year","averageRating","numVotes","rt_tomato","rt_audience","genres","tconst"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols].sort_values(["numVotes"], ascending=False)
        # kleine Skalenangleichung
        df["IMDb (x10)"] = (df["averageRating"]*10).round(1)
        df["RT Tomatometer"] = df["rt_tomato"].round(1)
        df["RT Audience"] = df["rt_audience"].round(1)
        df = df.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        return df[["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]]

    @output
    @render.download(filename=lambda: f"joined_movies_{datetime.date.today().isoformat()}.csv")
    def dl_joined():
        _, _, joined = data_all()
        yield joined.to_csv(index=False).encode("utf-8")

    # Kagglehub-Reload
    @reactive.effect
    @reactive.event(input.reload_kaggle)
    def _reload_rt():
        # Reset manuelles Upload-DF -> nächste data_all lädt wieder via kagglehub
        rt_manual.set(pd.DataFrame())
        ui.notification_show("Kagglehub-Download wird beim nächsten Zugriff neu ausgeführt.", duration=4, type="message")


app = App(app_ui, server)
