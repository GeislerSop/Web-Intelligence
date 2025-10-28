# app.py
# 🎬 Web-Intelligence: IMDb × Rotten Tomatoes (Kagglehub) — Shiny for Python
# Kompatibel mit Shiny ≤ 1.5.0 (Posit Connect Cloud)
# Tabs: Überblick | IMDb↔RT | Genres & Jahrzehnte | Google Trends | Downloads | Tabelle
# ------------------------------------------------------------------------------------

from __future__ import annotations
import io, gzip, re, os, json, datetime, time, random
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
    r = requests.get(url, timeout=180); r.raise_for_status()
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

    d["title_norm"] = d["title"].map(norm_title) if "title" in d.columns else ""

    year = pd.to_numeric(d.get("year_raw", pd.Series(index=d.index)), errors="coerce")
    if year.isna().all() and "movie_info_raw" in d.columns:
        year = d["movie_info_raw"].map(parse_year_from_text)
    d["year"] = year

    d["rt_tomato"]   = d.get("rt_tomato_raw",   pd.Series(index=d.index)).map(to_100)
    d["rt_audience"] = d.get("rt_audience_raw", pd.Series(index=d.index)).map(to_100)

    out = d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()
    return out

def load_rt_via_kagglehub(dataset_id=RT_DATASET_ID, file_path=RT_FILE_DEFAULT) -> pd.DataFrame:
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
            rt_raw = pd.DataFrame()
            print("Hinweis:", e)
    else:
        rt_raw = initial_rt

    rt_std = std_rt_kaggle(rt_raw) if not rt_raw.empty else pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])

    imdb_sel = imdb_full[["tconst","title","title_norm","year","averageRating","numVotes","genres"]].copy()
    joined   = imdb_sel.merge(rt_std, on=["title_norm","year"], how="left")
    return imdb_full, rt_std, joined

# ---------- UI (Dark Mode + Branding) ----------

app_ui = ui.page_fluid(
    # Dark mode styles
    ui.tags.style("""
        body { background-color: #1e1e1e; color: #eaeaea; }
        .card, .value-box, .form-control, .navbar, .nav-tabs {
            background-color: #2c2c2c !important; color: #eaeaea !important; border-color: #444 !important;
        }
        .btn, .download-btn { background-color: #0a84ff !important; color: white !important; border-radius: 8px; }
        h1, h2, h3, h4 { color: #f5f5f5; }
        a { color: #66aaff; }
    """),

    ui.panel_title("🎬 Web-Intelligence: IMDb × Rotten Tomatoes (Kagglehub)"),

    ui.tags.div(
        ui.tags.img(src="https://upload.wikimedia.org/wikipedia/commons/6/6e/Film_Reel_Icon.png", height="60px"),
        ui.tags.h2("Movie Analytics Dashboard", style="display:inline; margin-left:15px; color:#fff;"),
        class_="d-flex align-items-center mb-3"
    ),

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
        ),
        ui.column(8,
            ui.navset_tab(
                ui.nav_panel(
                    "Überblick",
                    ui.layout_column_wrap(
                        ui.output_ui("n_movies"),
                        ui.output_ui("avg_imdb"),
                        ui.output_ui("avg_rt"),
                        fill=False
                    ),
                    ui.output_plot("plot_avg_by_source", height="350px"),
                ),
                ui.nav_panel(
                    "IMDb ↔ Rotten Tomatoes",
                    ui.output_plot("plot_scatter", height="380px"),
                    ui.output_plot("plot_dist", height="380px"),
                ),
                ui.nav_panel(
                    "Genres & Jahrzehnte",
                    ui.output_plot("plot_genre", height="380px"),
                    ui.output_plot("plot_decade", height="380px"),
                ),
                ui.nav_panel(
                    "Google Trends",
                    ui.card(
                        ui.card_header("Abruf"),
                        ui.input_text("gt_kw1", "Keyword 1", "The Dark Knight 2008"),
                        ui.input_text("gt_kw2", "Keyword 2", "Inception 2010"),
                        ui.input_text("gt_kw3", "Keyword 3", "Oppenheimer 2023"),
                        ui.input_text("gt_kw4", "Keyword 4", "Fight Club 1999"),
                        ui.input_text("gt_kw5", "Keyword 5", "Forrest Gump 1994"),
                        ui.input_radio_buttons("gt_range", "Zeitraum", choices=["today 5-y","today 12-m","today 3-m"], selected="today 5-y", inline=True),
                        ui.input_action_button("gt_fetch", "Trends abrufen"),
                        ui.tags.small("Hinweis: In Uni/Cloud-Netzen kann Google Trends blockieren. Die App versucht mehrfach (Backoff) und nutzt notfalls Keywords ohne Jahr.")
                    ),
                    ui.output_text("gt_status"),
                    ui.output_plot("gt_plot", height="380px"),
                    ui.download_button("dl_trends", "Google-Trends (CSV)")
                ),
                ui.nav_panel(
                    "Downloads",
                    ui.card(
                        ui.card_header("CSV-Exporte"),
                        ui.download_button("dl_joined", "Ergebnistabelle (IMDb × RT)"),
                        ui.download_button("dl_top20", "Top-20 nach Stimmen (IMDb)"),
                        ui.tags.small("Tipp: Falls Google Trends nicht geht, exportiere direkt auf trends.google.com und nutze die CSV in deiner Doku.")
                    )
                ),
                ui.nav_panel(
                    "Tabelle",
                    ui.output_data_frame("table_movies")
                )
            )
        )
    ),
    ui.tags.hr(),
    ui.tags.small(
        "Daten: IMDb (offizielle TSVs) & Rotten Tomatoes (Kagglehub). ",
        "Google Trends mit robustem Retry/Fallback. ",
        f"Build: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z"
    )
)

# ---------- Server ----------

def server(input: Inputs, output: Outputs, session: Session):

    # Manuell hochgeladene RT-CSV
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
            ui.notification_show("RT-CSV geladen. Daten werden neu aufgebaut.", duration=4, type="message")
        except Exception as e:
            ui.notification_show(f"Upload-Fehler: {e}", duration=6, type="warning")

    @reactive.Calc
    def data_all():
        manual = rt_manual.get()
        if manual is not None and not manual.empty:
            imdb, rt, joined = load_data(initial_rt=manual)
        else:
            imdb, rt, joined = load_data()
        return imdb, rt, joined

    @reactive.Calc
    def data_filtered():
        imdb, rt, joined = data_all()
        y1, y2 = input.year_range()
        mv = input.min_votes()
        df = joined.copy()
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
        df = df[df["numVotes"].fillna(0) >= mv]
        return df

    # --- KPIs (Shiny ≥1.5: via render.ui + ui.value_box) ---
    @output
    @render.ui
    def n_movies():
        df = data_filtered()
        return ui.value_box(title="Filme (gefiltert)", value=f"{len(df):,}")

    @output
    @render.ui
    def avg_imdb():
        df = data_filtered()
        val = (df["averageRating"].dropna().mean()*10) if "averageRating" in df.columns and not df["averageRating"].dropna().empty else np.nan
        txt = f"{val:.1f}" if pd.notna(val) else "—"
        return ui.value_box(title="Ø IMDb (x10)", value=txt)

    @output
    @render.ui
    def avg_rt():
        df = data_filtered()
        val = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns and not df["rt_tomato"].dropna().empty else np.nan
        txt = f"{val:.1f}" if pd.notna(val) else "—"
        return ui.value_box(title="Ø RT Tomatometer", value=txt)

    @output
    @render.plot
    def plot_avg_by_source():
        df = data_filtered()
        vals = {}
        if "averageRating" in df.columns:
            v = df["averageRating"].dropna()
            if not v.empty: vals["IMDb (x10)"] = v.mean()*10
        if "rt_tomato" in df.columns:
            r = df["rt_tomato"].dropna()
            if not r.empty: vals["RT Tomatometer"] = r.mean()
        if input.use_audience() and "rt_audience" in df.columns:
            a = df["rt_audience"].dropna()
            if not a.empty: vals["RT Audience"] = a.mean()
        if not vals:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.text(0.5,0.5,"Keine Daten für Balkendiagramm", ha="center", va="center")
            ax.axis("off")
            return fig
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_title("Durchschnittliche Bewertung nach Quelle (0–100)")
        ax.set_ylabel("Score"); ax.set_ylim(0,100)
        return fig

    @output
    @render.plot
    def plot_scatter():
        df = data_filtered()
        if "rt_tomato" not in df.columns:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Kein RT verfügbar", ha="center", va="center"); ax.axis("off")
            return fig
        tmp = df.dropna(subset=["averageRating","rt_tomato"]).copy()
        if tmp.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Schnittmenge IMDb × RT", ha="center", va="center"); ax.axis("off")
            return fig
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(tmp["averageRating"]*10, tmp["rt_tomato"], alpha=0.5)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer (%)")
        ax.set_title("IMDb vs Rotten Tomatoes")
        x = (tmp["averageRating"]*10).values; y = tmp["rt_tomato"].values
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100); ys = m*xs + b
            ax.plot(xs, ys)
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        return fig

    @output
    @render.plot
    def plot_dist():
        df = data_filtered()
        if "rt_tomato" not in df.columns:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Kein RT verfügbar", ha="center", va="center"); ax.axis("off")
            return fig
        tmp = df.dropna(subset=["averageRating","rt_tomato"]).copy()
        if tmp.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Daten für Verteilung", ha="center", va="center"); ax.axis("off")
            return fig
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(tmp["averageRating"]*10, bins=30, alpha=0.5, density=True, label="IMDb")
        ax.hist(tmp["rt_tomato"], bins=30, alpha=0.5, density=True, label="RT Tomatometer")
        if input.use_audience() and "rt_audience" in tmp.columns:
            if tmp["rt_audience"].notna().any():
                ax.hist(tmp["rt_audience"], bins=30, alpha=0.3, density=True, label="RT Audience")
        ax.set_title("Bewertungsverteilung")
        ax.set_xlabel("Score (0–100)"); ax.legend(); ax.set_xlim(0,100)
        return fig

    @output
    @render.plot
    def plot_genre():
        imdb, rt, joined = data_all()
        imdb_g = imdb[imdb["numVotes"].fillna(0) >= 50000].dropna(subset=["genres"]).copy()
        if imdb_g.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Genre-Daten", ha="center", va="center"); ax.axis("off")
            return fig
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
        imdb, rt, joined = data_all()
        dec = imdb.dropna(subset=["year"]).copy()
        if dec.empty:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.text(0.5,0.5,"Keine Jahresdaten", ha="center", va="center"); ax.axis("off")
            return fig
        dec["year"] = dec["year"].astype(int)
        dec["decade"] = (dec["year"]//10)*10
        dec_avg = (dec.groupby("decade")["averageRating"].mean().sort_index())*10
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(dec_avg.index.values, dec_avg.values, marker="o")
        ax.set_title("IMDb: Ø Bewertung nach Jahrzehnt")
        ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø Bewertung (0–100)")
        ax.set_ylim(0,100)
        return fig

    @output
    @render.data_frame
    def table_movies():
        df = data_filtered().copy()
        cols = ["title","year","averageRating","numVotes","rt_tomato","rt_audience","genres","tconst"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
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
        _, _, joined = data_all()
        yield joined.to_csv(index=False).encode("utf-8")

    @output
    @render.download(filename=lambda: f"top20_imdb_{datetime.date.today().isoformat()}.csv")
    def dl_top20():
        imdb, _, _ = data_all()
        top20 = imdb.sort_values("numVotes", ascending=False).loc[:,["tconst","title","year","averageRating","numVotes"]].head(20)
        yield top20.to_csv(index=False).encode("utf-8")

    # Google Trends (robust mit Retry/Fallback)
    gt_store_df = reactive.Value(pd.DataFrame())
    gt_store_cols = reactive.Value([])
    gt_status_txt = reactive.Value("Noch keine Abfrage gestartet.")

    def _trends_try_fetch(keywords: list[str], timeframe: str, max_retries=4, wait_base=3.0):
        try:
            from pytrends.request import TrendReq
        except Exception as e:
            return None, f"Pytrends nicht installiert/verfügbar: {e}"

        pytrends = TrendReq(hl="de-DE", tz=0)
        for attempt in range(max_retries):
            try:
                pytrends.build_payload(kw_list=keywords, timeframe=timeframe)
                d = pytrends.interest_over_time()
                if d is not None and not d.empty:
                    return d, "OK"
            except Exception:
                sleep_s = wait_base * (2**attempt) + random.uniform(0, 0.8)
                time.sleep(sleep_s)
        return None, f"Keine Daten (Rate-Limit/Block?). Versuche: {max_retries}."

    def _keywords():
        kws = [input.gt_kw1(), input.gt_kw2(), input.gt_kw3(), input.gt_kw4(), input.gt_kw5()]
        return [k for k in kws if k and k.strip()][:5]

    @reactive.effect
    @reactive.event(input.gt_fetch)
    def _gt_fetch():
        gt_status_txt.set("Hole Trends …")
        timeframe = input.gt_range()
        kws = _keywords()
        if not kws:
            gt_status_txt.set("Bitte mindestens ein Keyword eingeben.")
            gt_store_df.set(pd.DataFrame()); gt_store_cols.set([])
            return

        d, msg = _trends_try_fetch(kws, timeframe)
        if d is None or d.empty:
            # Fallback: Keywords ohne Jahr
            kws_fallback = [re.sub(r"\b(19|20)\d{2}\b","", k).strip() for k in kws]
            kws_fallback = [re.sub(r"\s+"," ", k) for k in kws_fallback]
            d, msg2 = _trends_try_fetch(kws_fallback, timeframe)
            if d is None or d.empty:
                gt_status_txt.set(f"Trends fehlgeschlagen. Hinweis: {msg2}")
                gt_store_df.set(pd.DataFrame()); gt_store_cols.set([])
                return
            else:
                gt_status_txt.set("Trends OK (Fallback ohne Jahr).")
                gt_store_df.set(d.reset_index()); gt_store_cols.set(kws_fallback)
        else:
            gt_status_txt.set("Trends OK.")
            gt_store_df.set(d.reset_index()); gt_store_cols.set(kws)

    @output
    @render.text
    def gt_status():
        return gt_status_txt.get()

    @output
    @render.plot
    def gt_plot():
        d = gt_store_df.get()
        if d is None or d.empty:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.text(0.5,0.5,"Noch keine Daten", ha="center", va="center"); ax.axis("off")
            return fig
        cols = gt_store_cols.get()
        fig, ax = plt.subplots(figsize=(9,4))
        d = d.set_index("date") if "date" in d.columns else d.set_index(d.columns[0])
        for c in cols:
            if c in d.columns:
                ax.plot(d.index, d[c], label=c)
        ax.set_title("Google Trends: Suchinteresse")
        ax.set_xlabel("Datum"); ax.set_ylabel("Relatives Interesse (0–100)")
        ax.legend()
        return fig

    @reactive.effect
    @reactive.event(input.reload_kaggle)
    def _reload_rt():
        rt_manual.set(pd.DataFrame())
        ui.notification_show("Kagglehub-Download wird beim nächsten Zugriff neu ausgeführt.", duration=4, type="message")

    @output
    @render.download(filename=lambda: f"google_trends_{datetime.date.today().isoformat()}.csv")
    def dl_trends():
        d = gt_store_df.get()
        yield (d if d is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8")


app = App(app_ui, server)
