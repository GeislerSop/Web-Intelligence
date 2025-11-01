# app_shinylive.py — Movie Ratings (Shinylive/Browser)
# Läuft komplett im Browser (Pyodide). Holt NUR eine fertige CSV (joined.csv) per HTTP.
# Keine Threads, kein Kagglehub, keine externen ETL-Schritte in der App.

from __future__ import annotations
import io, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# WICHTIG für Browser: requests → fetch via pyodide_http
try:
    import pyodide_http
    pyodide_http.patch_all()
except Exception:
    pass

SAMPLE_MAX = 80000

# ---------------- Helpers ----------------
def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = re.sub(r"[^a-z0-9 ]+"," ", str(t).lower())
    return " ".join(t.split())

# ---------------- UI ----------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style("""
        :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3;--brand:#2563eb;--brand-weak:#e8efff}
        body{background:var(--bg);color:#0f172a}
        .sidebar{background:var(--card);border-right:1px solid var(--line)}
        .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 18px rgba(16,24,40,.06)}
        .btn{background:var(--brand);color:#fff;border-radius:10px;border:none}
        .muted{color:var(--muted);font-size:12px}
        .side-nav .shiny-input-radiogroup>div>label{display:block;padding:9px 12px;margin:4px 0;border-radius:10px;border:1px solid var(--line);cursor:pointer}
        .side-nav input[type=radio]{display:none}
        .side-nav .shiny-input-radiogroup>div>label:hover{background:#fafcff}
        .side-nav .shiny-input-radiogroup>div>input:checked+label{background:var(--brand-weak);border-color:#c7d7ff}
        """),
        ui.tags.h2("Movie Ratings", style="margin:6px 0 18px 0;"),
        ui.tags.div("CSV-Quelle", class_="muted"),
        ui.input_text("csv_url", None, "https://example.com/joined.csv", width="100%"),
        ui.input_action_button("load", "CSV laden", class_="btn"),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted", style="margin-bottom:6px;"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end",   "Jahr bis",  2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes",  "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "Audience-Score zusätzlich", False),
        open={"desktop": "open", "mobile": "closed"},
    ),

    ui.layout_column_wrap(
        ui.card(
            ui.card_header(ui.tags.div(id="page_title")),
            ui.output_ui("page_body"),
        ),
        fill=False,
    ),

    title="Movie Ratings (Shinylive)",
)

# ---------------- Server ----------------
def server(input: Inputs, output: Outputs, session: Session):
    store = reactive.Value({"ready": False, "df": pd.DataFrame(), "error": "", "src": ""})

    def set_error(msg: str):
        s = {"ready": False, "df": pd.DataFrame(), "error": msg, "src": store.get().get("src", "")}
        store.set(s)
        ui.notification_show(msg, type="warning", duration=6)

    @reactive.effect
    @reactive.event(input.load)
    def _load_csv():
        url = input.csv_url().strip()
        if not url:
            set_error("Bitte CSV-URL angeben."); return
        ui.notification_show("Lade CSV …", duration=None, id="note")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            # Mindestspalten prüfen
            need = {"title","year","averageRating","numVotes"}
            if not need.issubset(set(df.columns)):
                set_error(f"CSV unvollständig. Erwarte mind. Spalten: {sorted(need)}"); return
            if "title_norm" not in df:
                df["title_norm"] = df["title"].map(norm_title)
            if len(df) > SAMPLE_MAX:
                df = df.sample(SAMPLE_MAX, random_state=42)
            store.set({"ready": True, "df": df, "error": "", "src": url})
            ui.notification_show("CSV geladen.", type="message", duration=3)
        except Exception as e:
            set_error(f"CSV-Fehler: {e}")
        finally:
            ui.notification_remove("note")

    # Filter
    @reactive.Calc
    def df_filtered():
        s = store.get()
        if not s["ready"]:
            return pd.DataFrame()
        df = s["df"].copy()
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
        s = store.get()
        title = "Überblick" if not s["ready"] else f"Überblick — Quelle: {s['src']}"
        return ui.tags.h3(title)

    # Body
    @output
    @render.ui
    def page_body():
        s = store.get()
        if not s["ready"]:
            return ui.div(
                ui.tags.div("Noch keine Daten. Bitte CSV-URL eingeben und laden.", class_="muted", style="margin-bottom:8px;"),
                ui.tags.progress(max="100", value="10"),
            )
        if s["error"]:
            warn = ui.tags.div(f"Hinweis: {s['error']}", class_="muted", style="margin-bottom:10px;")
        else:
            warn = ui.tags.div()
        return ui.div(warn, kpi_ui(df_filtered()), ui.output_plot("p_avg"), ui.output_plot("p_scatter"), ui.output_plot("p_dist"), ui.output_data_frame("tbl"))

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
        tmp=df.dropna(subset=["averageRating"]) if not df.empty else df
        if tmp is None or tmp.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        ax.hist((tmp["averageRating"]*10).dropna(),bins=30,alpha=.5,density=True,label="IMDb")
        if "rt_tomato" in tmp.columns:
            ax.hist(tmp["rt_tomato"].dropna(),bins=30,alpha=.5,density=True,label="RT Tomatometer")
        if "rt_audience" in tmp.columns and input.use_audience():
            ax.hist(tmp["rt_audience"].dropna(),bins=30,alpha=.3,density=True,label="RT Audience")
        ax.legend(); ax.set_xlim(0,100); ax.set_title("Bewertungsverteilung"); ax.set_xlabel("Score (0–100)")
        return fig

    # Tabelle
    @output
    @render.data_frame
    def tbl():
        df=df_filtered().copy()
        if df.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"])
        df["IMDb (x10)"]=(df["averageRating"]*10).round(1)
        df["RT Tomatometer"]=df.get("rt_tomato", pd.Series(index=df.index)).round(1)
        df["RT Audience"]=df.get("rt_audience", pd.Series(index=df.index)).round(1)
        df=df.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]
        for c in cols:
            if c not in df.columns: df[c]=pd.NA
        return df[cols].sort_values("Stimmen",ascending=False)

app = App(app_ui, server)
