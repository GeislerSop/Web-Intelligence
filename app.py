# app.py — Movie Ratings (IMDb × RT via Kagglehub) — preload + cache + hübsche Sidebar
from __future__ import annotations
import io, gzip, re, os, time, random, threading, datetime, traceback
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# ---------------- Config ----------------
IMDB_BASE = "https://datasets.imdbws.com"
RT_DATASET_ID = "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
RT_FILE_DEFAULT = os.getenv("RT_DATASET_FILE", "rotten_tomatoes_movies.csv")

REQUEST_TIMEOUT = 45          # sec / HTTP
SAMPLE_MAX      = 80000       # weiches Limit für erste Anzeige
KEEPALIVE_SECS  = 20          # keep alive gegen idle-kick

DATA_DIR        = os.getenv("DATA_DIR", "data")  # Cache-Ordner
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def _cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

def save_csv(df: pd.DataFrame, name: str) -> None:
    try:
        df.to_csv(_cache_path(name), index=False)
    except Exception:
        pass

def try_read_csv(name: str) -> pd.DataFrame | None:
    p = _cache_path(name)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = re.sub(r"[^a-z0-9 ]+"," ", str(t).lower())
    return " ".join(t.split())

def to100(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%","")
    try:
        v = float(s)
        return v*10 if 0 <= v <= 10 else v
    except: return np.nan

def read_imdb(name: str, usecols=None) -> pd.DataFrame:
    r = requests.get(f"{IMDB_BASE}/{name}", timeout=REQUEST_TIMEOUT); r.raise_for_status()
    return pd.read_csv(io.BytesIO(gzip.decompress(r.content)),
                       sep="\t", na_values="\\N", usecols=usecols, low_memory=False)

def std_rt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])
    L={c.lower():c for c in df.columns}
    def pick(*xs):
        for x in xs:
            if x in L: return L[x]
    c_title=pick("movie_title","title","name")
    c_year =pick("original_release_year","year","release_year")
    c_info =pick("movie_info")
    c_tom =pick("tomatometer_rating","tomato_score","tomatometer","rotten_tomatoes","rt","rt_score")
    c_aud =pick("audience_rating","audience_score","audiencescore")
    keep=[c for c in [c_title,c_year,c_info,c_tom,c_aud] if c]
    d=df[keep].copy() if keep else df.iloc[0:0].copy()
    if d.empty:
        return pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])
    ren={}
    if c_title: ren[c_title]="title"
    if c_year:  ren[c_year]="year"
    if c_info:  ren[c_info]="info"
    if c_tom:   ren[c_tom]="rt_tomato_raw"
    if c_aud:   ren[c_aud]="rt_audience_raw"
    d.rename(columns=ren,inplace=True)
    d["title_norm"]=d["title"].map(norm_title) if "title" in d else ""
    d["year"]=pd.to_numeric(d.get("year",pd.Series(index=d.index)),errors="coerce")
    if "info" in d and d["year"].isna().all():
        d["year"]=d["info"].astype(str).str.extract(r"\b(19|20)\d{2}\b",expand=False).astype(float)
    d["rt_tomato"]=d.get("rt_tomato_raw",pd.Series(index=d.index)).map(to100)
    d["rt_audience"]=d.get("rt_audience_raw",pd.Series(index=d.index)).map(to100)
    return d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()

def load_rt_kagglehub(dataset_id=RT_DATASET_ID, file_path=RT_FILE_DEFAULT) -> pd.DataFrame:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_id, file_path)

def build_joined(rt_raw: pd.DataFrame|None=None):
    basics  = read_imdb("title.basics.tsv.gz",  ["tconst","titleType","primaryTitle","startYear","genres"])
    ratings = read_imdb("title.ratings.tsv.gz", ["tconst","averageRating","numVotes"])
    movies  = basics[basics["titleType"]=="movie"].copy()
    movies.rename(columns={"primaryTitle":"title","startYear":"year"},inplace=True)
    movies["year"]=pd.to_numeric(movies["year"],errors="coerce")
    movies["title_norm"]=movies["title"].map(norm_title)
    imdb = movies.merge(ratings,on="tconst",how="left")

    if rt_raw is None or rt_raw.empty:
        try: rt_raw = load_rt_kagglehub()
        except Exception: rt_raw = pd.DataFrame()
    rt_std = std_rt(rt_raw)

    joined = imdb[["tconst","title","title_norm","year","averageRating","numVotes","genres"]].merge(
        rt_std, on=["title_norm","year"], how="left"
    )
    if len(joined) > SAMPLE_MAX:
        joined = joined.sample(SAMPLE_MAX, random_state=42)
    return imdb, rt_std, joined

# ---------------- UI (helle Sidebar, modernes Styling) ----------------
app_ui = ui.page_sidebar(
    # Sidebar (positionsargument)
    ui.sidebar(
        ui.tags.style("""
        :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3;--brand:#2563eb;--brand-weak:#e8efff}
        body{background:var(--bg);color:#0f172a}
        .sidebar{background:var(--card);border-right:1px solid var(--line)}
        .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 18px rgba(16,24,40,.06)}
        .btn{background:var(--brand);color:#fff;border-radius:10px;border:none}
        .muted{color:var(--muted);font-size:12px}
        .side-nav .shiny-input-radiogroup>div>label{display:block;padding:9px 12px;margin:4px 0;border-radius:10px;border:1px solid var(--line);cursor:pointer}
        .side-nav input[type="radio"]{display:none}
        .side-nav .shiny-input-radiogroup>div>label:hover{background:#fafcff}
        .side-nav .shiny-input-radiogroup>div>input:checked+label{background:var(--brand-weak);border-color:#c7d7ff}
        """),
        ui.tags.h2("Movie Ratings", style="margin:6px 0 18px 0;"),
        ui.div(
            ui.tags.div("Navigation", class_="muted", style="margin-bottom:6px;"),
            ui.div(
                ui.input_radio_buttons(
                    "page", None,
                    choices={
                        "overview":"Überblick",
                        "compare":"IMDb ↔ RT",
                        "trends":"Genres/Jahrzehnte",
                        "gtrends":"Google Trends",
                        "downloads":"Downloads",
                        "table":"Tabelle",
                    },
                    selected="overview",
                    inline=False
                ),
                class_="side-nav"
            ),
        ),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted", style="margin-bottom:6px;"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end",   "Jahr bis",  2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes",  "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "Audience-Score zusätzlich", False),
        ui.input_action_button("reload_rt", "RT-Daten neu laden"),
        ui.input_file("rt_upload", "oder RT-CSV hochladen", accept=[".csv"], multiple=False),
        open={"desktop": "open", "mobile": "closed"},
    ),

    # Hauptinhalt (positionsargument)
    ui.layout_column_wrap(
        ui.card(
            ui.card_header(ui.tags.div(id="page_title")),
            ui.output_ui("page_body"),
        ),
        fill=False,
    ),

    title="Movie Ratings",
)

# ---------------- Server ----------------
def server(input: Inputs, output: Outputs, session: Session):

    # Keep alive
    @reactive.effect
    def _keepalive():
        reactive.invalidate_later(KEEPALIVE_SECS)

    # Store + Status
    store = reactive.Value({"ready":False,"imdb":None,"rt":None,"joined":None,"error":""})

    def set_error(prefix: str, e: Exception):
        msg = f"{prefix}: {e.__class__.__name__}: {e}"
        store.set({"ready":False,"imdb":None,"rt":None,"joined":None,"error":msg})
        ui.notification_show(msg, type="warning", duration=8)

    # Start: Versuche Cache zu lesen, sonst Background-Build
    def warm_start():
        try:
            imdb  = try_read_csv("imdb.csv")
            rt    = try_read_csv("rt_std.csv")
            joined= try_read_csv("joined.csv")
            if imdb is not None and rt is not None and joined is not None and not joined.empty:
                store.set({"ready":True,"imdb":imdb,"rt":rt,"joined":joined,"error":""})
                return
        except Exception:
            pass
        # Kein Cache: baue im Hintergrund
        threading.Thread(target=bg_build_and_cache, daemon=True).start()

    def bg_build_and_cache(rt_override: pd.DataFrame|None=None):
        try:
            imdb, rt, joined = build_joined(rt_override)
            # Cache schreiben
            save_csv(imdb,  "imdb.csv")
            save_csv(rt,    "rt_std.csv")
            save_csv(joined,"joined.csv")
            store.set({"ready":True,"imdb":imdb,"rt":rt,"joined":joined,"error":""})
            ui.notification_show("Daten geladen.", type="message", duration=4)
        except Exception as e:
            # Fallback-Demo
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
            save_csv(demo, "joined.csv")
            store.set({"ready":True,"imdb":demo,"rt":demo[["title_norm","year","rt_tomato","rt_audience"]],
                       "joined":demo,"error":str(e)})
            ui.notification_show("Konnte Live-Daten nicht laden – Demo-Daten werden angezeigt.", type="warning", duration=8)

    warm_start()  # kick-off beim Start

    # Upload / Reload (asynchron + Cache)
    @reactive.effect
    @reactive.event(input.rt_upload)
    def _on_upload():
        f=input.rt_upload()
        if f:
            try:
                df=pd.read_csv(f[0]["datapath"])
            except Exception as e:
                set_error("Upload-Fehler beim Einlesen", e); return
            store.set({"ready":False,"imdb":None,"rt":None,"joined":None,"error":""})
            ui.notification_show("RT-CSV geladen – baue Daten zusammen …", type="message", duration=4)
            threading.Thread(target=bg_build_and_cache, args=(df,), daemon=True).start()

    @reactive.effect
    @reactive.event(input.reload_rt)
    def _reload():
        store.set({"ready":False,"imdb":None,"rt":None,"joined":None,"error":""})
        ui.notification_show("Kagglehub wird neu geladen …", type="message", duration=4)
        threading.Thread(target=bg_build_and_cache, daemon=True).start()

    # Filter
    @reactive.Calc
    def df_filtered():
        s=store.get()
        if not s["ready"] or s["joined"] is None: return pd.DataFrame()
        df=s["joined"].copy()
        y1,y2=int(input.year_start()),int(input.year_end())
        if y1>y2: y1,y2=y2,y1
        mv=int(input.min_votes())
        df=df[(df["year"].fillna(0)>=y1)&(df["year"].fillna(0)<=y2)]
        df=df[df["numVotes"].fillna(0)>=mv]
        return df

    # Titel
    @output
    @render.ui
    def page_title():
        t={"overview":"Überblick","compare":"IMDb ↔ Rotten Tomatoes",
           "trends":"Genres & Jahrzehnte","gtrends":"Google Trends",
           "downloads":"Downloads","table":"Tabelle"}
        return ui.tags.h3(t.get(input.page(),"Überblick"))

    # Body
    @output
    @render.ui
    def page_body():
        s = store.get()
        if not s["ready"]:
            return ui.div(
                ui.tags.div("Lade Daten … (oder Cache wird erzeugt)", class_="muted", style="margin-bottom:8px;"),
                ui.tags.progress(max="100", value="25"),
            )
        if s["error"]:
            # zeige Hinweis oberhalb des Inhalts
            warn = ui.tags.div(f"Hinweis: {s['error']}", class_="muted", style="margin-bottom:10px;")
        else:
            warn = ui.tags.div()

        p=input.page()
        if p=="overview":  return ui.div(warn, kpi_ui(df_filtered()), ui.output_plot("p_avg"))
        if p=="compare":   return ui.div(warn, ui.output_plot("p_scatter"), ui.output_plot("p_dist"))
        if p=="trends":    return ui.div(warn, ui.output_plot("p_genre"), ui.output_plot("p_decade"))
        if p=="gtrends":   return ui.div(warn, trends_ui())
        if p=="downloads": return ui.div(warn, dl_ui())
        if p=="table":     return ui.div(warn, ui.output_data_frame("tbl"))
        return ui.div("—")

    # KPIs
    def kpi_ui(df):
        imdb_mean = (df["averageRating"].dropna().mean()*10) if "averageRating" in df.columns and df["averageRating"].notna().any() else np.nan
        rt_mean   = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns and df["rt_tomato"].notna().any() else np.nan
        return ui.layout_column_wrap(
            ui.value_box(title="Filme (gefiltert)", value=f"{len(df)}"),
            ui.value_box(title="Ø IMDb (x10)", value=("—" if pd.isna(imdb_mean) else f"{imdb_mean:.1f}")),
            ui.value_box(title="Ø RT Tomatometer", value=("—" if pd.isna(rt_mean) else f"{rt_mean:.1f}")),
            fill=False
        )

    # Plots
    @output
    @render.plot
    def p_avg():
        df=df_filtered(); vals={}
        if "averageRating" in df: vals["IMDb (x10)"]=df["averageRating"].dropna().mean()*10
        if "rt_tomato" in df:     vals["RT Tomatometer"]=df["rt_tomato"].dropna().mean()
        if "rt_audience" in df and input.use_audience(): vals["RT Audience"]=df["rt_audience"].dropna().mean()
        fig,ax=plt.subplots(figsize=(8,4))
        if not vals: ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        ax.bar(list(vals.keys()), list(vals.values())); ax.set_ylim(0,100)
        ax.set_title("Durchschnittliche Bewertung (0–100)"); ax.set_ylabel("Score")
        return fig

    @output
    @render.plot
    def p_scatter():
        df=df_filtered()
        fig,ax=plt.subplots(figsize=(6,6))
        if "rt_tomato" not in df: ax.axis("off"); ax.text(0.5,0.5,"RT nicht verfügbar",ha="center",va="center"); return fig
        tmp=df.dropna(subset=["averageRating","rt_tomato"])
        if tmp.empty: ax.axis("off"); ax.text(0.5,0.5,"Keine Schnittmenge",ha="center",va="center"); return fig
        ax.scatter(tmp["averageRating"]*10,tmp["rt_tomato"],alpha=.5)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer (%)"); ax.set_xlim(0,100); ax.set_ylim(0,100)
        x,y=(tmp["averageRating"]*10).values,tmp["rt_tomato"].values
        if len(x)>=2:
            m,b=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),100); ax.plot(xs,m*xs+b)
        ax.set_title("IMDb vs Rotten Tomatoes")
        return fig

    @output
    @render.plot
    def p_dist():
        df=df_filtered(); fig,ax=plt.subplots(figsize=(9,4))
        if "rt_tomato" not in df: ax.axis("off"); ax.text(0.5,0.5,"RT nicht verfügbar",ha="center",va="center"); return fig
        tmp=df.dropna(subset=["averageRating","rt_tomato"])
        if tmp.empty: ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        ax.hist(tmp["averageRating"]*10,bins=30,alpha=.5,density=True,label="IMDb")
        ax.hist(tmp["rt_tomato"],bins=30,alpha=.5,density=True,label="RT Tomatometer")
        if "rt_audience" in tmp.columns and input.use_audience():
            ax.hist(tmp["rt_audience"],bins=30,alpha=.3,density=True,label="RT Audience")
        ax.legend(); ax.set_xlim(0,100); ax.set_title("Bewertungsverteilung"); ax.set_xlabel("Score (0–100)")
        return fig

    @output
    @render.plot
    def p_genre():
        s=store.get(); imdb=s["imdb"] if s["ready"] else pd.DataFrame()
        fig,ax=plt.subplots(figsize=(9,4))
        imdb_g=imdb[imdb.get("numVotes",pd.Series()).fillna(0)>=50000].dropna(subset=["genres"]) if not imdb.empty else pd.DataFrame()
        if imdb_g.empty: ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten",ha="center",va="center"); return fig
        g=imdb_g.assign(genre=imdb_g["genres"].str.split(",")).explode("genre")
        avg=(g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12))*10
        ax.bar(avg.index.tolist(), avg.values); ax.set_xticklabels(avg.index.tolist(), rotation=45, ha="right")
        ax.set_ylabel("Ø (0–100)"); ax.set_ylim(0,100); ax.set_title("IMDb (≥50k) — Ø Genre (Top 12)")
        return fig

    @output
    @render.plot
    def p_decade():
        s=store.get(); imdb=s["imdb"] if s["ready"] else pd.DataFrame()
        fig,ax=plt.subplots(figsize=(9,4))
        if imdb.empty: ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        dec=imdb.dropna(subset=["year"]).copy(); dec["year"]=dec["year"].astype(int); dec["decade"]=(dec["year"]//10)*10
        avg=(dec.groupby("decade")["averageRating"].mean().sort_index())*10
        ax.plot(avg.index.values, avg.values, marker="o"); ax.set_ylim(0,100)
        ax.set_title("IMDb — Ø Bewertung nach Jahrzehnt"); ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø (0–100)")
        return fig

    # Tabelle
    @output
    @render.data_frame
    def tbl():
        df=df_filtered().copy()
        if df.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"])
        df["IMDb (x10)"]=(df["averageRating"]*10).round(1)
        df["RT Tomatometer"]=df["rt_tomato"].round(1)
        df["RT Audience"]=df["rt_audience"].round(1)
        df=df.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]
        for c in cols:
            if c not in df.columns: df[c]=pd.NA
        return df[cols].sort_values("Stimmen",ascending=False)

    # Downloads
    @output
    @render.download(filename=lambda: f"joined_{datetime.date.today().isoformat()}.csv")
    def dl_joined():
        s=store.get(); df=s["joined"] if s["ready"] else pd.DataFrame()
        yield df.to_csv(index=False).encode("utf-8")

    @output
    @render.download(filename=lambda: f"top20_{datetime.date.today().isoformat()}.csv")
    def dl_top20():
        s=store.get(); imdb=s["imdb"] if s["ready"] else pd.DataFrame()
        top=imdb.sort_values("numVotes",ascending=False).loc[:,["tconst","title","year","averageRating","numVotes"]].head(20)
        yield top.to_csv(index=False).encode("utf-8")

    # Google Trends (mit Retry / Fallback)
    gt_df = reactive.Value(pd.DataFrame()); gt_cols = reactive.Value([]); gt_msg = reactive.Value("Noch keine Abfrage.")

    def _trends_try(kws, tf, retries=4, base=2.5):
        try:
            from pytrends.request import TrendReq
        except Exception as e:
            return None, f"Pytrends fehlt: {e}"
        pt=TrendReq(hl="de-DE", tz=0)
        for i in range(retries):
            try:
                pt.build_payload(kws, timeframe=tf)
                d=pt.interest_over_time()
                if d is not None and not d.empty: return d, "OK"
            except Exception:
                time.sleep(base*(2**i)+random.uniform(0,0.7))
        return None, "Rate-Limit/Block?"

    def trends_ui():
        return ui.div(
            ui.tags.div("Hinweis: Trends kann im Uni-Netz blockiert sein. Fallback ohne Jahre.", class_="muted"),
            ui.input_text("gt1","Keyword 1","The Dark Knight 2008"),
            ui.input_text("gt2","Keyword 2","Inception 2010"),
            ui.input_text("gt3","Keyword 3","Oppenheimer 2023"),
            ui.input_text("gt4","Keyword 4","Fight Club 1999"),
            ui.input_text("gt5","Keyword 5","Forrest Gump 1994"),
            ui.input_radio_buttons("gt_tf","Zeitraum",choices=["today 5-y","today 12-m","today 3-m"],selected="today 5-y", inline=True),
            ui.input_action_button("gt_go","Trends abrufen"),
            ui.output_text("gt_status"),
            ui.output_plot("gt_plot"),
            ui.download_button("dl_trends","Google-Trends (CSV)")
        )

    def dl_ui():
        return ui.div(
            ui.download_button("dl_joined","Ergebnistabelle (IMDb × RT)"),
            ui.download_button("dl_top20","Top-20 nach Stimmen (IMDb)")
        )

    @reactive.effect
    @reactive.event(input.gt_go)
    def _gt():
        kws=[k for k in [input.gt1(),input.gt2(),input.gt3(),input.gt4(),input.gt5()] if k and k.strip()]
        if not kws: gt_msg.set("Bitte Keywords eingeben."); return
        gt_msg.set("Hole Trends …")
        d,msg=_trends_try(kws, input.gt_tf())
        if d is None or d.empty:
            k2=[re.sub(r"\b(19|20)\d{2}\b","",k).strip() for k in kws]
            d,msg2=_trends_try(k2, input.gt_tf())
            if d is None or d.empty:
                gt_msg.set(f"Trends fehlgeschlagen. {msg2}"); gt_df.set(pd.DataFrame()); return
            gt_cols.set(k2); gt_msg.set("Trends OK (Fallback ohne Jahr)."); gt_df.set(d.reset_index()); return
        gt_cols.set(kws); gt_msg.set("Trends OK."); gt_df.set(d.reset_index())

    @output
    @render.text
    def gt_status():
        return gt_msg.get()

    @output
    @render.plot
    def gt_plot():
        d=gt_df.get(); fig,ax=plt.subplots(figsize=(9,4))
        if d is None or d.empty: ax.axis("off"); ax.text(0.5,0.5,"Noch keine Daten",ha="center",va="center"); return fig
        d=d.set_index("date") if "date" in d.columns else d.set_index(d.columns[0])
        for c in gt_cols.get():
            if c in d.columns: ax.plot(d.index,d[c],label=c)
        ax.legend(); ax.set_title("Google Trends"); ax.set_xlabel("Datum"); ax.set_ylabel("Interesse (0–100)")
        return fig

    @output
    @render.download(filename=lambda: f"google_trends_{datetime.date.today().isoformat()}.csv")
    def dl_trends():
        yield (gt_df.get() if gt_df.get() is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8")

app = App(app_ui, server)
