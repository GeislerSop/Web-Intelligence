# app.py
# 🎬 Movie Ratings Analytics — IMDb × Rotten Tomatoes (Kagglehub)
# Shiny (Python) 1.5 kompatibel | helle UI | seitliche Tabs (navset_pill_list)
# Tabs: Überblick | IMDb↔RT | Genres & Jahrzehnte | Google Trends | Downloads | Tabelle

from __future__ import annotations
import io, gzip, re, os, datetime, time, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# --------------------------
# Daten-Beschaffung & Utils
# --------------------------

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
    r = requests.get(url, timeout=180); r.raise_for_status()
    return pd.read_csv(io.BytesIO(gzip.decompress(r.content)),
                       sep="\t", na_values="\\N", usecols=usecols, low_memory=False)

def parse_year_from_text(s: str):
    if pd.isna(s): return np.nan
    m = re.search(r"\b(19|20)\d{2}\b", str(s))
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

    d["title_norm"] = d["title"].map(norm_title) if "title" in d.columns else ""

    year = pd.to_numeric(d.get("year_raw", pd.Series(index=d.index)), errors="coerce")
    if year.isna().all() and "movie_info_raw" in d.columns:
        year = d["movie_info_raw"].map(parse_year_from_text)
    d["year"] = year

    d["rt_tomato"]   = d.get("rt_tomato_raw",   pd.Series(index=d.index)).map(to_100)
    d["rt_audience"] = d.get("rt_audience_raw", pd.Series(index=d.index)).map(to_100)

    return d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()

def load_rt_via_kagglehub(dataset_id=RT_DATASET_ID, file_path=RT_FILE_DEFAULT) -> pd.DataFrame:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_id, file_path)

def build_data(initial_rt: pd.DataFrame | None = None):
    basics  = read_imdb_tsv("title.basics.tsv.gz",  ["tconst","titleType","primaryTitle","startYear","runtimeMinutes","genres"])
    ratings = read_imdb_tsv("title.ratings.tsv.gz", ["tconst","averageRating","numVotes"])
    movies  = basics[basics["titleType"]=="movie"].copy()
    movies.rename(columns={"primaryTitle":"title","startYear":"year"}, inplace=True)
    movies["year"]           = pd.to_numeric(movies["year"], errors="coerce")
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
    movies["title_norm"]     = movies["title"].map(norm_title)
    imdb_full = movies.merge(ratings, on="tconst", how="left")

    if initial_rt is None:
        try:
            rt_raw = load_rt_via_kagglehub()
        except Exception as e:
            print("Kagglehub Hinweis:", e)
            rt_raw = pd.DataFrame()
    else:
        rt_raw = initial_rt

    rt_std = std_rt_kaggle(rt_raw) if not rt_raw.empty else pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])
    imdb_sel = imdb_full[["tconst","title","title_norm","year","averageRating","numVotes","genres"]].copy()
    joined   = imdb_sel.merge(rt_std, on=["title_norm","year"], how="left")
    return imdb_full, rt_std, joined

# -------------
# UI – Helles Theme + Seiten-Navigation (links)
# -------------

app_ui = ui.page_fluid(
    # Light look & feel
    ui.tags.style("""
        body{ background:#f7f8fb; color:#1a1a1a; }
        .card, .value-box, .form-control, .navbar, .nav-pills, .nav{ background:#ffffff !important; color:#1a1a1a !important; }
        .card{ border:1px solid #e8ebf3; box-shadow:0 8px 18px rgba(16,24,40,.06); border-radius:16px; }
        .value-box{ border:1px solid #e8ebf3; border-radius:16px; }
        .btn, .download-btn{ background:#2563eb !important; color:#fff !important; border-radius:10px; border:none; }
        h1{ font-weight:700; letter-spacing:.2px; color:#0f172a; }
        h2,h3,h4{ color:#0f172a; }
        .sidebar-title{ font-weight:600; color:#334155; }
        .shiny-input-container{ margin-bottom:14px; }
        .nav-pills .nav-link.active{ background:#e8efff !important; color:#1e3a8a !important; }
        .nav-pills .nav-link{ color:#334155 !important; border-radius:10px; }
    """),

    # kurze, prägnante Überschrift
    ui.row(
        ui.column(12, ui.tags.h1("Movie Ratings Analytics"))
    ),

    # Layout: linke Spalte = Navigation + Filter, rechte Spalte = Inhalte
    ui.layout_columns(
        # linke Spalte (Navigation + Filter)
        ui.card(
            ui.card_header(ui.tags.div("Navigation & Filter", class_="sidebar-title")),
            ui.navset_pill_list(  # vertikale Tabs
                ui.nav_panel("Überblick", ui.output_ui("kpis"), value="tab_overview"),
                ui.nav_panel("IMDb ↔ RT",  ui.output_plot("plot_scatter", height="380px"),
                                          ui.output_plot("plot_dist", height="380px"),
                             value="tab_compare"),
                ui.nav_panel("Genres & Jahrzehnte", ui.output_plot("plot_genre", height="360px"),
                                                   ui.output_plot("plot_decade", height="360px"),
                             value="tab_trends"),
                ui.nav_panel("Google Trends", ui.card(
                                ui.input_text("gt_kw1", "Keyword 1", "The Dark Knight 2008"),
                                ui.input_text("gt_kw2", "Keyword 2", "Inception 2010"),
                                ui.input_text("gt_kw3", "Keyword 3", "Oppenheimer 2023"),
                                ui.input_text("gt_kw4", "Keyword 4", "Fight Club 1999"),
                                ui.input_text("gt_kw5", "Keyword 5", "Forrest Gump 1994"),
                                ui.input_radio_buttons("gt_range", "Zeitraum",
                                                       choices=["today 5-y","today 12-m","today 3-m"],
                                                       selected="today 5-y", inline=True),
                                ui.input_action_button("gt_fetch", "Trends abrufen"),
                                ui.tags.small("Hinweis: In Uni/Cloud-Netzen kann Trends blockieren. Es wird mit Retry/Fallback gearbeitet.")
                              ),
                              ui.output_text("gt_status"),
                              ui.output_plot("gt_plot", height="320px"),
                              ui.download_button("dl_trends", "Google-Trends (CSV)"),
                              value="tab_gtrends"),
                ui.nav_panel("Downloads", ui.download_button("dl_joined", "Ergebnistabelle (IMDb × RT)"),
                                          ui.download_button("dl_top20", "Top-20 nach Stimmen (IMDb)"),
                             value="tab_downloads"),
                ui.nav_panel("Tabelle", ui.output_data_frame("table_movies"), value="tab_table"),
                id="sidebar_nav"
            ),
            ui.tags.hr(),
            ui.tags.div("Filter", class_="sidebar-title"),
            ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
            ui.input_numeric("year_end",   "Jahr bis",  2025, min=1920, max=2025, step=1),
            ui.input_numeric("min_votes", "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
            ui.input_checkbox("use_audience", "Audience-Score (zusätzlich zu Tomatometer)", False),
            ui.input_action_button("reload_kaggle", "RT aus Kagglehub neu laden"),
            ui.input_file("rt_upload", "oder: Rotten-Tomatoes CSV hochladen", accept=[".csv"], multiple=False),
            col_widths=4
        ),

        # rechte Spalte (Arbeitsfläche – hier wird pro Tab gerendert)
        ui.card(
            ui.card_header("Arbeitsfläche"),
            ui.output_plot("plot_avg_by_source", height="360px"),
            col_widths=8
        ),
        col_gap="24px"
    ),

    ui.tags.hr(),
    ui.tags.small(
        "Daten: IMDb TSVs & Rotten Tomatoes (Kagglehub). ",
        "Build: ", datetime.datetime.now(datetime.timezone.utc).isoformat(), "Z"
    ),
)

# -----------------
# Server-Logik
# -----------------

def server(input: Inputs, output: Outputs, session: Session):

    # Lazy-Store: wird erst beim ersten Zugriff gebaut (vermeidet lange Blocker -> weniger 'Reload'-Hinweise)
    _cache = reactive.Value({"imdb": None, "rt": None, "joined": None})

    def ensure_data(initial_rt_df: pd.DataFrame | None = None):
        store = _cache.get()
        if store["joined"] is None or initial_rt_df is not None:
            imdb, rt, joined = build_data(initial_rt_df)
            _cache.set({"imdb": imdb, "rt": rt, "joined": joined})
        return _cache.get()["imdb"], _cache.get()["rt"], _cache.get()["joined"]

    # manueller RT-Upload
    rt_manual = reactive.Value(pd.DataFrame())

    @reactive.effect
    @reactive.event(input.rt_upload)
    def _on_upload():
        f = input.rt_upload()
        if f:
            try:
                df = pd.read_csv(f[0]["datapath"])
                rt_manual.set(df)
                ensure_data(df)  # sofort neu aufbauen
                ui.notification_show("RT-CSV geladen & Daten aktualisiert.", type="message", duration=4)
            except Exception as e:
                ui.notification_show(f"Upload-Fehler: {e}", type="warning", duration=6)

    # gefilterte Daten
    @reactive.Calc
    def data_filtered():
        manual = rt_manual.get()
        imdb, rt, joined = ensure_data(manual if not manual.empty else None)
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2: y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = joined.copy()
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
        df = df[df["numVotes"].fillna(0) >= mv]
        return df

    # KPI-Boxen (als ein UI-Block, damit die Karte oben rechts sauber wirkt)
    @output
    @render.ui
    def kpis():
        df = data_filtered()
        imdb_mean = (df["averageRating"].dropna().mean()*10) if "averageRating" in df.columns and df["averageRating"].notna().any() else np.nan
        rt_mean   = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns and df["rt_tomato"].notna().any() else np.nan
        return ui.layout_column_wrap(
            ui.value_box(title="Filme (gefiltert)", value=f"{len(df):,}".replace(",", ".")),
            ui.value_box(title="Ø IMDb (x10)", value=("—" if pd.isna(imdb_mean) else f"{imdb_mean:.1f}")),
            ui.value_box(title="Ø RT Tomatometer", value=("—" if pd.isna(rt_mean) else f"{rt_mean:.1f}")),
            fill=False
        )

    # Balken: Ø-Bewertungen je Quelle
    @output
    @render.plot
    def plot_avg_by_source():
        df = data_filtered()
        vals = {}
        if "averageRating" in df.columns and df["averageRating"].notna().any():
            vals["IMDb (x10)"] = df["averageRating"].mean()*10
        if "rt_tomato" in df.columns and df["rt_tomato"].notna().any():
            vals["RT Tomatometer"] = df["rt_tomato"].mean()
        if input.use_audience() and "rt_audience" in df.columns and df["rt_audience"].notna().any():
            vals["RT Audience"] = df["rt_audience"].mean()

        if not vals:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.text(0.5,0.5,"Keine Daten für Vergleich", ha="center", va="center"); ax.axis("off")
            return fig

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_title("Durchschnittliche Bewertung nach Quelle (0–100)")
        ax.set_ylabel("Score"); ax.set_ylim(0,100)
        return fig

    # Scatter IMDb vs RT
    @output
    @render.plot
    def plot_scatter():
        df = data_filtered()
        if "rt_tomato" not in df.columns:
            fig, ax = plt.subplots(figsize=(4,3)); ax.axis("off")
            ax.text(0.5,0.5,"RT nicht verfügbar", ha="center", va="center"); return fig
        tmp = df.dropna(subset=["averageRating","rt_tomato"])
        if tmp.empty:
            fig, ax = plt.subplots(figsize=(4,3)); ax.axis("off")
            ax.text(0.5,0.5,"Keine Schnittmenge IMDb × RT", ha="center", va="center"); return fig
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(tmp["averageRating"]*10, tmp["rt_tomato"], alpha=0.5)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer (%)"); ax.set_xlim(0,100); ax.set_ylim(0,100)
        ax.set_title("IMDb vs Rotten Tomatoes")
        x, y = (tmp["averageRating"]*10).values, tmp["rt_tomato"].values
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100); ax.plot(xs, m*xs+b)
        return fig

    # Verteilungen
    @output
    @render.plot
    def plot_dist():
        df = data_filtered()
        if "rt_tomato" not in df.columns:
            fig, ax = plt.subplots(figsize=(4,3)); ax.axis("off")
            ax.text(0.5,0.5,"RT nicht verfügbar", ha="center", va="center"); return fig
        tmp = df.dropna(subset=["averageRating","rt_tomato"])
        if tmp.empty:
            fig, ax = plt.subplots(figsize=(4,3)); ax.axis("off")
            ax.text(0.5,0.5,"Keine Daten für Verteilung", ha="center", va="center"); return fig
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(tmp["averageRating"]*10, bins=30, alpha=0.5, density=True, label="IMDb")
        ax.hist(tmp["rt_tomato"], bins=30, alpha=0.5, density=True, label="RT Tomatometer")
        if input.use_audience() and "rt_audience" in tmp.columns and tmp["rt_audience"].notna().any():
            ax.hist(tmp["rt_audience"], bins=30, alpha=0.3, density=True, label="RT Audience")
        ax.set_title("Bewertungsverteilung"); ax.set_xlabel("Score (0–100)"); ax.legend(); ax.set_xlim(0,100)
        return fig

    # Genres / Jahrzehnte
    @output
    @render.plot
    def plot_genre():
        imdb, _, _ = ensure_data()
        imdb_g = imdb[imdb["numVotes"].fillna(0) >= 50000].dropna(subset=["genres"]).copy()
        if imdb_g.empty:
            fig, ax = plt.subplots(figsize=(4,3)); ax.axis("off")
            ax.text(0.5,0.5,"Keine Genre-Daten", ha="center", va="center"); return fig
        g = imdb_g.assign(genre=imdb_g["genres"].str.split(",")).explode("genre")
        avg_by_genre = (g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12))*10
        fig, ax = plt.subplots(figsize=(9,4))
        ax.bar(avg_by_genre.index.tolist(), avg_by_genre.values)
        ax.set_title("IMDb (≥50k Stimmen): Ø Bewertung pro Genre (Top 12)")
        ax.set_xticklabels(avg_by_genre.index.tolist(), rotation=45, ha="right")
        ax.set_ylabel("Ø Bewertung (0–100)"); ax.set_ylim(0,100)
        return fig

    @output
    @render.plot
    def plot_decade():
        imdb, _, _ = ensure_data()
        dec = imdb.dropna(subset=["year"]).copy()
        if dec.empty:
            fig, ax = plt.subplots(figsize=(4,3)); ax.axis("off")
            ax.text(0.5,0.5,"Keine Jahresdaten", ha="center", va="center"); return fig
        dec["year"] = dec["year"].astype(int)
        dec["decade"] = (dec["year"]//10)*10
        dec_avg = (dec.groupby("decade")["averageRating"].mean().sort_index())*10
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(dec_avg.index.values, dec_avg.values, marker="o")
        ax.set_title("IMDb: Ø Bewertung nach Jahrzehnt"); ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø Bewertung (0–100)")
        ax.set_ylim(0,100)
        return fig

    # Tabelle
    @output
    @render.data_frame
    def table_movies():
        df = data_filtered().copy()
        cols = ["title","year","averageRating","numVotes","rt_tomato","rt_audience","genres","tconst"]
        for c in cols:
            if c not in df.columns: df[c] = pd.NA
        df = df[cols].sort_values(["numVotes"], ascending=False)
        df["IMDb (x10)"] = (df["averageRating"]*10).round(1)
        df["RT Tomatometer"] = df["rt_tomato"].round(1)
        df["RT Audience"] = df["rt_audience"].round(1)
        df = df.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        return df[["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]]

    # Downloads
    @output
    @render.download(filename=lambda: f"joined_movies_{datetime.date.today().isoformat()}.csv")
    def dl_joined():
        _, _, joined = ensure_data()
        yield joined.to_csv(index=False).encode("utf-8")

    @output
    @render.download(filename=lambda: f"top20_imdb_{datetime.date.today().isoformat()}.csv")
    def dl_top20():
        imdb, _, _ = ensure_data()
        top20 = imdb.sort_values("numVotes", ascending=False).loc[:,["tconst","title","year","averageRating","numVotes"]].head(20)
        yield top20.to_csv(index=False).encode("utf-8")

    # Google Trends (robust)
    gt_store_df = reactive.Value(pd.DataFrame())
    gt_store_cols = reactive.Value([])
    gt_status_txt = reactive.Value("Noch keine Abfrage.")

    def _trends_try_fetch(keywords: list[str], timeframe: str, max_retries=4, wait_base=2.5):
        try:
            from pytrends.request import TrendReq
        except Exception as e:
            return None, f"Pytrends nicht verfügbar: {e}"
        pytrends = TrendReq(hl="de-DE", tz=0)
        for attempt in range(max_retries):
            try:
                pytrends.build_payload(kw_list=keywords, timeframe=timeframe)
                d = pytrends.interest_over_time()
                if d is not None and not d.empty:
                    return d, "OK"
            except Exception:
                time.sleep(wait_base * (2**attempt) + random.uniform(0,0.8))
        return None, f"Keine Daten (Rate-Limit/Block?)."

    def _keywords():
        kws = [input.gt_kw1(), input.gt_kw2(), input.gt_kw3(), input.gt_kw4(), input.gt_kw5()]
        return [k for k in kws if k and k.strip()][:5]

    @reactive.effect
    @reactive.event(input.gt_fetch)
    def _gt_fetch():
        gt_status_txt.set("Hole Trends…")
        timeframe = input.gt_range()
        kws = _keywords()
        if not kws:
            gt_status_txt.set("Bitte mindestens ein Keyword eingeben.")
            gt_store_df.set(pd.DataFrame()); gt_store_cols.set([]); return
        d, msg = _trends_try_fetch(kws, timeframe)
        if d is None or d.empty:
            # Fallback: ohne Jahreszahl
            k2 = [re.sub(r"\b(19|20)\d{2}\b","", k).strip() for k in kws]
            k2 = [re.sub(r"\s+"," ", k) for k in k2]
            d, msg2 = _trends_try_fetch(k2, timeframe)
            if d is None or d.empty:
                gt_status_txt.set(f"Trends fehlgeschlagen. Hinweis: {msg2}")
                gt_store_df.set(pd.DataFrame()); gt_store_cols.set([]); return
            gt_status_txt.set("Trends OK (Fallback ohne Jahr)."); gt_store_df.set(d.reset_index()); gt_store_cols.set(k2)
        else:
            gt_status_txt.set("Trends OK."); gt_store_df.set(d.reset_index()); gt_store_cols.set(kws)

    @output
    @render.text
    def gt_status():
        return gt_status_txt.get()

    @output
    @render.plot
    def gt_plot():
        d = gt_store_df.get()
        if d is None or d.empty:
            fig, ax = plt.subplots(figsize=(6,3)); ax.axis("off")
            ax.text(0.5,0.5,"Noch keine Daten", ha="center", va="center"); return fig
        cols = gt_store_cols.get()
        fig, ax = plt.subplots(figsize=(9,4))
        d = d.set_index("date") if "date" in d.columns else d.set_index(d.columns[0])
        for c in cols:
            if c in d.columns: ax.plot(d.index, d[c], label=c)
        ax.set_title("Google Trends: Suchinteresse"); ax.set_xlabel("Datum"); ax.set_ylabel("Relatives Interesse (0–100)"); ax.legend()
        return fig

    @reactive.effect
    @reactive.event(input.reload_kaggle)
    def _reload_rt():
        # löscht Cache -> nächster Zugriff lädt neu
        _cache.set({"imdb": None, "rt": None, "joined": None})
        rt_manual.set(pd.DataFrame())
        ui.notification_show("Kagglehub-Daten werden beim nächsten Zugriff neu geladen.", type="message", duration=4)

    @output
    @render.download(filename=lambda: f"google_trends_{datetime.date.today().isoformat()}.csv")
    def dl_trends():
        d = gt_store_df.get()
        yield (d if d is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8")


app = App(app_ui, server)
