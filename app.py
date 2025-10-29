# app.py
# Movie Ratings — IMDb × Rotten Tomatoes (Kagglehub)
# Helles UI • Sidebar-Navigation • Non-blocking Data Load • Trends-Fallback

from __future__ import annotations
import io, gzip, re, os, datetime, time, random, threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# -------------------- Config --------------------
IMDB_BASE = "https://datasets.imdbws.com"
RT_DATASET_ID = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
RT_FILE_DEFAULT = os.getenv("RT_DATASET_FILE", "rotten_tomatoes_movies.csv")
REQUEST_TIMEOUT = 120  # Sekunden
SAMPLE_MAX = 120000    # harte Obergrenze für erste Anzeige (Performance)

# -------------------- Utilities --------------------
def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9 ]+"," ", t)
    return " ".join(t.split())

def to_100(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%","")
    try:
        v = float(s)
        if 0 <= v <= 10: v *= 10.0
        return v
    except:
        return np.nan

def read_imdb_tsv(name: str, usecols=None) -> pd.DataFrame:
    url = f"{IMDB_BASE}/{name}"
    r = requests.get(url, timeout=REQUEST_TIMEOUT); r.raise_for_status()
    return pd.read_csv(io.BytesIO(gzip.decompress(r.content)),
                       sep="\t", na_values="\\N", usecols=usecols, low_memory=False)

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
    if c_year:   ren[c_year]   = "year"
    if c_info:   ren[c_info]   = "info"
    if c_tomato: ren[c_tomato] = "rt_tomato_raw"
    if c_aud:    ren[c_aud]    = "rt_audience_raw"
    d.rename(columns=ren, inplace=True)

    d["title_norm"] = d["title"].map(norm_title) if "title" in d.columns else ""
    d["year"] = pd.to_numeric(d.get("year", pd.Series(index=d.index)), errors="coerce")
    if "info" in d.columns and d["year"].isna().all():
        d["year"] = d["info"].astype(str).str.extract(r"\b(19|20)\d{2}\b", expand=False).astype(float)

    d["rt_tomato"]   = d.get("rt_tomato_raw",   pd.Series(index=d.index)).map(to_100)
    d["rt_audience"] = d.get("rt_audience_raw", pd.Series(index=d.index)).map(to_100)

    return d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()

def load_rt_via_kagglehub(dataset_id=RT_DATASET_ID, file_path=RT_FILE_DEFAULT) -> pd.DataFrame:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_id, file_path)

def build_joined(rt_raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    basics  = read_imdb_tsv("title.basics.tsv.gz",  ["tconst","titleType","primaryTitle","startYear","runtimeMinutes","genres"])
    ratings = read_imdb_tsv("title.ratings.tsv.gz", ["tconst","averageRating","numVotes"])
    movies  = basics[basics["titleType"]=="movie"].copy()
    movies.rename(columns={"primaryTitle":"title","startYear":"year"}, inplace=True)
    movies["year"]           = pd.to_numeric(movies["year"], errors="coerce")
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
    movies["title_norm"]     = movies["title"].map(norm_title)
    imdb = movies.merge(ratings, on="tconst", how="left")

    if rt_raw is None or rt_raw.empty:
        try:
            rt_raw = load_rt_via_kagglehub()
        except Exception:
            rt_raw = pd.DataFrame()
    rt_std = std_rt_kaggle(rt_raw) if not rt_raw.empty else pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])

    joined = imdb[["tconst","title","title_norm","year","averageRating","numVotes","genres"]].merge(
        rt_std, on=["title_norm","year"], how="left"
    )

    # Soft-Sampling für schnelle Erstanzeige
    if len(joined) > SAMPLE_MAX:
        joined = joined.sample(SAMPLE_MAX, random_state=42)
    return imdb, rt_std, joined

# -------------------- UI (hell + Sidebar) --------------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style("""
            body{background:#f7f8fb;color:#1a1a1a;}
            .sidebar{background:#ffffff;border-right:1px solid #e8ebf3;}
            .card{background:#fff;border:1px solid #e8ebf3;border-radius:14px;box-shadow:0 8px 18px rgba(16,24,40,.06);}
            .btn{background:#2563eb;color:#fff;border-radius:10px;border:none;}
            h1{margin:0 0 8px 0;font-weight:700;color:#0f172a;}
            .menu a{display:block;padding:10px 12px;border-radius:10px;color:#334155;text-decoration:none;margin-bottom:6px;}
            .menu a.active{background:#e8efff;color:#1e3a8a;}
            .muted{color:#6b7280;font-size:12px;}
        """),
        ui.tags.h2("Movie Ratings", style="margin:6px 0 18px 0;"),
        ui.tags.div(
            ui.tags.div("Navigation", class_="muted"),
            ui.tags.div(
                ui.tags.a("Überblick",       href="#", id="nav_overview",   class_="menu-link"),
                ui.tags.a("IMDb ↔ RT",       href="#", id="nav_compare",    class_="menu-link"),
                ui.tags.a("Genres/Jahrzehnte",href="#", id="nav_trends",     class_="menu-link"),
                ui.tags.a("Google Trends",   href="#", id="nav_gtrends",    class_="menu-link"),
                ui.tags.a("Downloads",       href="#", id="nav_downloads",  class_="menu-link"),
                ui.tags.a("Tabelle",         href="#", id="nav_table",      class_="menu-link"),
                class_="menu"
            ),
        ),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted", style="margin-bottom:6px;"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end",   "Jahr bis", 2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes",  "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "Audience-Score zusätzlich", False),
        ui.input_action_button("reload_rt", "RT (Kagglehub) neu laden"),
        ui.input_file("rt_upload", "oder RT-CSV hochladen", accept=[".csv"], multiple=False),
        open=True
    ),
    ui.layout_column_wrap(
        ui.card(
            ui.card_header(ui.tags.div(id="page_title")),
            ui.output_ui("page_body")
        ),
        fill=False
    ),
    title="Movie Ratings",
)

# -------------------- Server --------------------
def server(input: Inputs, output: Outputs, session: Session):

    # --- Navigation-State ---
    page = reactive.Value("overview")

    @session.on_flushed
    def _init_nav():
        # erstes Item als aktiv markieren (kleines JS)
        session.send_custom_message("set_active", {"id":"nav_overview"})

    @session.register_web_method
    def nav_click(id: str):
        mapping = {
            "nav_overview":"overview",
            "nav_compare":"compare",
            "nav_trends":"trends",
            "nav_gtrends":"gtrends",
            "nav_downloads":"downloads",
            "nav_table":"table",
        }
        if id in mapping:
            page.set(mapping[id])
            # aktive Klasse setzen
            session.send_custom_message("set_active", {"id":id})

    # binde Klicks (einfaches JS)
    session.send_script("""
        const send=(id)=>Shiny.setInputValue('.__web_method__nav_click', { id });
        for (const id of ['nav_overview','nav_compare','nav_trends','nav_gtrends','nav_downloads','nav_table']){
          const el=document.getElementById(id); if(!el) continue;
          el.addEventListener('click', (e)=>{ e.preventDefault(); send(id); });
        }
        Shiny.addCustomMessageHandler('set_active', ({id})=>{
          document.querySelectorAll('.menu-link').forEach(a=>a.classList.remove('active'));
          const el=document.getElementById(id); if(el) el.classList.add('active');
        });
    """)

    # --- Daten-Cache + non-blocking Background Load ---
    data_cache = reactive.Value({"ready": False, "imdb": None, "rt": None, "joined": None, "error": ""})

    def _load_data_background(rt_override: pd.DataFrame | None = None):
        try:
            imdb, rt, joined = build_joined(rt_override)
            data_cache.set({"ready": True, "imdb": imdb, "rt": rt, "joined": joined, "error": ""})
        except Exception as e:
            # Mini-Demo-Datensatz, damit Charts nicht leer sind
            demo = pd.DataFrame({
                "title":["Demo A","Demo B","Demo C","Demo D"],
                "title_norm":["demo a","demo b","demo c","demo d"],
                "year":[1994,1999,2010,2008],
                "averageRating":[9.3,8.8,8.8,9.0],
                "numVotes":[2700000,2200000,2400000,2800000],
                "genres":["Drama","Drama","Action,Sci-Fi","Action,Crime"],
                "rt_tomato":[91,79,87,94],
                "rt_audience":[98,96,91,94],
                "tconst":["tt0111161","tt0137523","tt1375666","tt0468569"],
            })
            data_cache.set({"ready": True, "imdb": demo, "rt": demo[["title_norm","year","rt_tomato","rt_audience"]],
                            "joined": demo, "error": f"{e}"})

    # Start sofort, aber im Hintergrund
    threading.Thread(target=_load_data_background, daemon=True).start()

    # Upload RT → neu laden im Hintergrund
    @reactive.effect
    @reactive.event(input.rt_upload)
    def _on_upload():
        f = input.rt_upload()
        if f:
            try:
                df = pd.read_csv(f[0]["datapath"])
                data_cache.set({"ready": False, "imdb": None, "rt": None, "joined": None, "error": ""})
                threading.Thread(target=_load_data_background, args=(df,), daemon=True).start()
                ui.notification_show("RT-CSV geladen – Daten werden aktualisiert …", type="message", duration=4)
            except Exception as e:
                ui.notification_show(f"Upload-Fehler: {e}", type="warning", duration=6)

    @reactive.effect
    @reactive.event(input.reload_rt)
    def _reload_rt():
        data_cache.set({"ready": False, "imdb": None, "rt": None, "joined": None, "error": ""})
        threading.Thread(target=_load_data_background, daemon=True).start()
        ui.notification_show("Kagglehub wird neu geladen …", type="message", duration=4)

    # Gefilterte Daten
    @reactive.Calc
    def filtered():
        store = data_cache.get()
        if not store["ready"] or store["joined"] is None:
            return pd.DataFrame()
        df = store["joined"].copy()
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2: y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
        df = df[df["numVotes"].fillna(0) >= mv]
        return df

    # ------- Seiten-Rendering -------
    def kpi_block(df: pd.DataFrame):
        imdb_mean = (df["averageRating"].dropna().mean()*10) if "averageRating" in df.columns and df["averageRating"].notna().any() else np.nan
        rt_mean   = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns and df["rt_tomato"].notna().any() else np.nan
        return ui.layout_column_wrap(
            ui.value_box(title="Filme (gefiltert)", value=f"{len(df)}"),
            ui.value_box(title="Ø IMDb (x10)", value=("—" if pd.isna(imdb_mean) else f"{imdb_mean:.1f}")),
            ui.value_box(title="Ø RT Tomatometer", value=("—" if pd.isna(rt_mean) else f"{rt_mean:.1f}")),
            fill=False
        )

    def plot_avg(df):
        vals={}
        if "averageRating" in df.columns and df["averageRating"].notna().any():
            vals["IMDb (x10)"]=df["averageRating"].mean()*10
        if "rt_tomato" in df.columns and df["rt_tomato"].notna().any():
            vals["RT Tomatometer"]=df["rt_tomato"].mean()
        if "rt_audience" in df.columns and df["rt_audience"].notna().any() and input.use_audience():
            vals["RT Audience"]=df["rt_audience"].mean()
        fig,ax=plt.subplots(figsize=(8,4))
        if not vals:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten für Vergleich",ha="center",va="center")
            return fig
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_title("Durchschnittliche Bewertung (0–100)"); ax.set_ylabel("Score"); ax.set_ylim(0,100)
        return fig

    def plot_scatter(df):
        fig,ax=plt.subplots(figsize=(6,6))
        if "rt_tomato" not in df.columns:
            ax.axis("off"); ax.text(0.5,0.5,"RT nicht verfügbar",ha="center",va="center"); return fig
        tmp=df.dropna(subset=["averageRating","rt_tomato"])
        if tmp.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Schnittmenge IMDb × RT",ha="center",va="center"); return fig
        ax.scatter(tmp["averageRating"]*10,tmp["rt_tomato"],alpha=0.5)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer (%)"); ax.set_xlim(0,100); ax.set_ylim(0,100)
        x,y=(tmp["averageRating"]*10).values,tmp["rt_tomato"].values
        if len(x)>=2:
            m,b=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),100); ax.plot(xs,m*xs+b)
        ax.set_title("IMDb vs Rotten Tomatoes")
        return fig

    def plot_dist(df):
        fig,ax=plt.subplots(figsize=(9,4))
        if "rt_tomato" not in df.columns:
            ax.axis("off"); ax.text(0.5,0.5,"RT nicht verfügbar",ha="center",va="center"); return fig
        tmp=df.dropna(subset=["averageRating","rt_tomato"])
        if tmp.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        ax.hist(tmp["averageRating"]*10,bins=30,alpha=0.5,density=True,label="IMDb")
        ax.hist(tmp["rt_tomato"],bins=30,alpha=0.5,density=True,label="RT Tomatometer")
        if "rt_audience" in tmp.columns and tmp["rt_audience"].notna().any() and input.use_audience():
            ax.hist(tmp["rt_audience"],bins=30,alpha=0.3,density=True,label="RT Audience")
        ax.set_title("Bewertungsverteilung"); ax.set_xlabel("Score (0–100)"); ax.legend(); ax.set_xlim(0,100)
        return fig

    def plot_genre(imdb):
        imdb_g = imdb[imdb["numVotes"].fillna(0) >= 50000].dropna(subset=["genres"]).copy()
        fig,ax=plt.subplots(figsize=(9,4))
        if imdb_g.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten",ha="center",va="center"); return fig
        g = imdb_g.assign(genre=imdb_g["genres"].str.split(",")).explode("genre")
        avg = (g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12))*10
        ax.bar(avg.index.tolist(), avg.values); ax.set_xticklabels(avg.index.tolist(), rotation=45, ha="right")
        ax.set_title("IMDb (≥50k Stimmen) — Ø Bewertung nach Genre (Top 12)")
        ax.set_ylabel("Ø (0–100)"); ax.set_ylim(0,100)
        return fig

    def plot_decade(imdb):
        dec = imdb.dropna(subset=["year"]).copy()
        fig,ax=plt.subplots(figsize=(9,4))
        if dec.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        dec["year"]=dec["year"].astype(int); dec["decade"]=(dec["year"]//10)*10
        avg=(dec.groupby("decade")["averageRating"].mean().sort_index())*10
        ax.plot(avg.index.values, avg.values, marker="o")
        ax.set_title("IMDb — Ø Bewertung nach Jahrzehnt"); ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø (0–100)"); ax.set_ylim(0,100)
        return fig

    # Seitenkopf
    @output
    @render.ui
    def page_title():
        titles = {
            "overview": "Überblick",
            "compare": "IMDb ↔ Rotten Tomatoes",
            "trends": "Genres & Jahrzehnte",
            "gtrends": "Google Trends",
            "downloads": "Downloads",
            "table": "Tabelle"
        }
        return ui.tags.h3(titles.get(page.get(),"Überblick"))

    # Seiteninhalt
    @output
    @render.ui
    def page_body():
        store = data_cache.get()
        if not store["ready"]:
            # Loader/Skeleton statt Reload-Overlay
            return ui.div(
                ui.tags.div("Lade Daten …", class_="muted", style="margin-bottom:8px;"),
                ui.progress(id="p1", value=30),
                ui.tags.small("IMDb/RT werden im Hintergrund geladen. Die Seite bleibt nutzbar."),
            )
        df = filtered()
        if page.get()=="overview":
            return ui.div(
                kpi_block(df),
                ui.output_plot("p_avg")
            )
        if page.get()=="compare":
            return ui.div(ui.output_plot("p_scatter"), ui.output_plot("p_dist"))
        if page.get()=="trends":
            return ui.div(ui.output_plot("p_genre"), ui.output_plot("p_decade"))
        if page.get()=="gtrends":
            return ui.div(
                ui.tags.div("Hinweis: Trends kann in Uni/Cloud blockiert sein. Es wird mit Retry & Fallback (ohne Jahr) versucht.", class_="muted"),
                ui.input_text("gt_kw1","Keyword 1","The Dark Knight 2008"),
                ui.input_text("gt_kw2","Keyword 2","Inception 2010"),
                ui.input_text("gt_kw3","Keyword 3","Oppenheimer 2023"),
                ui.input_text("gt_kw4","Keyword 4","Fight Club 1999"),
                ui.input_text("gt_kw5","Keyword 5","Forrest Gump 1994"),
                ui.input_radio_buttons("gt_range","Zeitraum",choices=["today 5-y","today 12-m","today 3-m"],selected="today 5-y", inline=True),
                ui.input_action_button("gt_fetch","Trends abrufen"),
                ui.output_text("gt_status"),
                ui.output_plot("gt_plot"),
                ui.download_button("dl_trends","Google-Trends (CSV)")
            )
        if page.get()=="downloads":
            return ui.div(
                ui.download_button("dl_joined","Ergebnistabelle (IMDb × RT)"),
                ui.download_button("dl_top20","Top-20 nach Stimmen (IMDb)")
            )
        if page.get()=="table":
            return ui.output_data_frame("tbl")
        return ui.div("—")

    # Plots render
    @output @render.plot
    def p_avg(): return plot_avg(filtered())
    @output @render.plot
    def p_scatter(): return plot_scatter(filtered())
    @output @render.plot
    def p_dist(): return plot_dist(filtered())

    @output @render.plot
    def p_genre():
        store = data_cache.get()
        imdb = store["imdb"] if store["ready"] else pd.DataFrame()
        return plot_genre(imdb)

    @output @render.plot
    def p_decade():
        store = data_cache.get()
        imdb = store["imdb"] if store["ready"] else pd.DataFrame()
        return plot_decade(imdb)

    # Tabelle
    @output
    @render.data_frame
    def tbl():
        df = filtered().copy()
        if df.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"])
        cols=["title","year","averageRating","numVotes","rt_tomato","rt_audience","genres","tconst"]
        for c in cols:
            if c not in df.columns: df[c]=pd.NA
        df["IMDb (x10)"] = (df["averageRating"]*10).round(1)
        df["RT Tomatometer"] = df["rt_tomato"].round(1)
        df["RT Audience"] = df["rt_audience"].round(1)
        df = df.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        return df[["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]]

    # Downloads
    @output
    @render.download(filename=lambda: f"joined_movies_{datetime.date.today().isoformat()}.csv")
    def dl_joined():
        store = data_cache.get()
        df = store["joined"] if store["ready"] else pd.DataFrame()
        yield df.to_csv(index=False).encode("utf-8")

    @output
    @render.download(filename=lambda: f"top20_imdb_{datetime.date.today().isoformat()}.csv")
    def dl_top20():
        store = data_cache.get()
        imdb = store["imdb"] if store["ready"] else pd.DataFrame()
        top20 = imdb.sort_values("numVotes", ascending=False).loc[:,["tconst","title","year","averageRating","numVotes"]].head(20)
        yield top20.to_csv(index=False).encode("utf-8")

    # Google Trends
    gt_store = reactive.Value(pd.DataFrame()); gt_cols = reactive.Value([]); gt_status = reactive.Value("Noch keine Abfrage.")

    def _trends_try(keywords, timeframe, max_retries=4, base=2.5):
        try:
            from pytrends.request import TrendReq
        except Exception as e:
            return None, f"Pytrends nicht verfügbar: {e}"
        pt = TrendReq(hl="de-DE", tz=0)
        for k in range(max_retries):
            try:
                pt.build_payload(keywords, timeframe=timeframe)
                d = pt.interest_over_time()
                if d is not None and not d.empty: return d, "OK"
            except Exception:
                time.sleep(base*(2**k)+random.uniform(0,0.8))
        return None, "Rate-Limit/Block?"

    @reactive.effect
    @reactive.event(input.gt_fetch)
    def _gt_fetch():
        gt_status.set("Hole Trends …")
        kws = [v for v in [input.gt_kw1(),input.gt_kw2(),input.gt_kw3(),input.gt_kw4(),input.gt_kw5()] if v and v.strip()]
        if not kws: gt_status.set("Bitte Keywords eingeben."); return
        d, msg = _trends_try(kws, input.gt_range())
        if d is None or d.empty:
            # Fallback ohne Jahr
            k2 = [re.sub(r"\b(19|20)\d{2}\b","", k).strip() for k in kws]
            d, msg2 = _trends_try(k2, input.gt_range())
            if d is None or d.empty:
                gt_status.set(f"Trends fehlgeschlagen. {msg2}"); gt_store.set(pd.DataFrame()); return
            gt_status.set("Trends OK (Fallback ohne Jahr)."); gt_cols.set(k2)
        else:
            gt_status.set("Trends OK."); gt_cols.set(kws)
        gt_store.set(d.reset_index())

    @output
    @render.text
    def gt_status():
        return gt_status.get()

    @output
    @render.plot
    def gt_plot():
        d = gt_store.get()
        fig,ax=plt.subplots(figsize=(9,4))
        if d is None or d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Noch keine Daten",ha="center",va="center"); return fig
        cols=gt_cols.get()
        d = d.set_index("date") if "date" in d.columns else d.set_index(d.columns[0])
        for c in cols:
            if c in d.columns: ax.plot(d.index,d[c],label=c)
        ax.set_title("Google Trends"); ax.set_xlabel("Datum"); ax.set_ylabel("Interesse (0–100)"); ax.legend()
        return fig

    @output
    @render.download(filename=lambda: f"google_trends_{datetime.date.today().isoformat()}.csv")
    def dl_trends():
        d = gt_store.get()
        yield (d if d is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8")


app = App(app_ui, server)
