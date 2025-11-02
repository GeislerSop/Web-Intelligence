# app.py — Movie Ratings Dashboard (Posit Connect)
# Nutzt 3 CSVs:
#   outputs/joined_imdb_rt.csv       (IMDb-Basisdaten, evtl. schon RT-Spalten)
#   outputs/rotten_tomatoes_movies.csv  (oder raw/... als Fallback; RT-Rohdaten)
#   outputs/top20_by_votes_imdb.csv
#   outputs/google_trends_top5.csv

from __future__ import annotations

import os, re, logging, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# --- Noise/Warnings reduzieren (insb. Hist/Divide)
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("movie-app")

# ---------------- Pfade ----------------
BASE = Path(__file__).resolve().parent
def _find(*candidates: Path) -> Path|None:
    for p in candidates:
        if p.exists():
            return p
    return None

CSV_JOINED = _find(BASE/"outputs/joined_imdb_rt.csv", BASE/"joined_imdb_rt.csv")
CSV_RT_RAW = _find(BASE/"outputs/rotten_tomatoes_movies.csv",
                   BASE/"raw/rotten_tomatoes_movies.csv")
CSV_TOP20  = _find(BASE/"outputs/top20_by_votes_imdb.csv", BASE/"top20_by_votes_imdb.csv")
CSV_GTR    = _find(BASE/"outputs/google_trends_top5.csv", BASE/"google_trends_top5.csv")

def _read_csv(p: Path|None) -> pd.DataFrame:
    if p is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        LOG.info(f"Loaded {p} shape={df.shape}")
        return df
    except Exception as e:
        LOG.exception(f"CSV read failed: {p} -> {e}")
        return pd.DataFrame()

joined_raw = _read_csv(CSV_JOINED)     # IMDb-Basis (+ evtl. RT)
rt_raw     = _read_csv(CSV_RT_RAW)     # RT-Rohdaten
top20_raw  = _read_csv(CSV_TOP20)
gtr_raw    = _read_csv(CSV_GTR)

# ---------------- Helpers ----------------
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
    except: 
        return np.nan

def std_rt(df: pd.DataFrame) -> pd.DataFrame:
    """Bringt verschiedene RT-Schemata auf ein gemeinsames Set."""
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
    d.rename(columns=ren, inplace=True)
    d["title_norm"]=d.get("title","").map(norm_title)
    d["year"]=pd.to_numeric(d.get("year",pd.Series(index=d.index)),errors="coerce")
    if "info" in d and d["year"].isna().all():
        d["year"]=d["info"].astype(str).str.extract(r"\b(19|20)\d{2}\b",expand=False).astype(float)
    d["rt_tomato"]=d.get("rt_tomato_raw",pd.Series(index=d.index)).map(to100)
    d["rt_audience"]=d.get("rt_audience_raw",pd.Series(index=d.index)).map(to100)
    out = d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()
    return out

# --- vorbereiten: joined (IMDb) + RT-Standard
if not joined_raw.empty and "title_norm" not in joined_raw.columns and "title" in joined_raw.columns:
    joined_raw["title_norm"] = joined_raw["title"].map(norm_title)
rt_std = std_rt(rt_raw)

# Für Visuals: On-the-fly „sanfter Merge“ (füllt nur, was fehlt)
def joined_for_visuals() -> pd.DataFrame:
    if joined_raw.empty:
        return joined_raw
    df = joined_raw.copy()
    cols = ["title_norm","year"]
    if "year" in df: df["year"]=pd.to_numeric(df["year"], errors="coerce")
    if not rt_std.empty and all(c in df.columns for c in cols):
        df = df.merge(rt_std, on=cols, how="left", suffixes=("","_rtstd"))
        # Priorität: vorhandene RT in joined behalten, sonst RT aus rt_std nehmen
        if "rt_tomato" not in df and "rt_tomato_rtstd" in df:
            df["rt_tomato"] = df["rt_tomato_rtstd"]
        else:
            df["rt_tomato"] = df["rt_tomato"].fillna(df.get("rt_tomato_rtstd"))
        if "rt_audience" not in df and "rt_audience_rtstd" in df:
            df["rt_audience"] = df["rt_audience_rtstd"]
        else:
            df["rt_audience"] = df["rt_audience"].fillna(df.get("rt_audience_rtstd"))
    return df

# ---------------- UI ----------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style("""
        :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3;--brand:#2563eb;--brand-weak:#e8efff}
        html,body{height:100%;background:var(--bg);color:#0f172a}
        .sidebar{background:var(--card);border-right:1px solid var(--line);
                 height:100vh;max-height:100vh;overflow:auto;position:sticky;top:0}
        .muted{color:var(--muted);font-size:12px}
        .side-nav .shiny-input-radiogroup>div>label{
          display:block;padding:9px 12px;margin:4px 0;border-radius:10px;border:1px solid var(--line);cursor:pointer}
        .side-nav input[type=radio]{display:none}
        .side-nav .shiny-input-radiogroup>div>label:hover{background:#fafcff}
        .side-nav .shiny-input-radiogroup>div>input:checked+label{background:var(--brand-weak);border-color:#c7d7ff}
        """),
        ui.tags.h2("🎬 Movie Ratings"),
        ui.tags.div("Navigation", class_="muted"),
        ui.div(
            ui.input_radio_buttons(
                "page", None,
                choices={
                    "overview":"Übersicht",
                    "compare":"IMDb ↔ RT",
                    "coverage":"Abdeckung RT",
                    "trends":"Trends",
                    "top20":"Top 20",
                    "gtrends":"Google Trends",
                    "table":"Tabelle",
                    "rt_only":"RT (nur RT-Daten)"
                },
                selected="overview", inline=False
            ),
            class_="side-nav"
        ),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end",   "Jahr bis",  2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes",  "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "Audience-Score zusätzlich", False),
        ui.tags.hr(),
        ui.tags.div("Status", class_="muted"),
        ui.output_ui("status_files"),
    ),
    ui.layout_column_wrap(
        ui.card(ui.card_header(ui.tags.div(id="page_title")), ui.output_ui("page_body")),
        fill=False,
    ),
    title="Movie Ratings Dashboard",
    # KEIN 'open' dict hier – das hatte den früheren Fehler ausgelöst
)

# ---------------- Server ----------------
def server(input: Inputs, output: Outputs, session: Session):

    # Status
    @output
    @render.ui
    def status_files():
        def li(ok, text): return ui.tags.li((("✅ " if ok else "❌ ") + text))
        rows = [
            li(CSV_JOINED is not None, f"joined: {CSV_JOINED} (shape={tuple(joined_raw.shape)})"),
            li(CSV_RT_RAW is not None, f"rt_raw: {CSV_RT_RAW} (shape={tuple(rt_raw.shape)})"),
            li(CSV_TOP20  is not None, f"top20 : {CSV_TOP20}  (shape={tuple(top20_raw.shape)})"),
            li(CSV_GTR    is not None, f"gtr   : {CSV_GTR}    (shape={tuple(gtr_raw.shape)})"),
        ]
        return ui.tags.small(ui.tags.ul(*rows, style="margin:0;padding-left:18px;"))

    # Gefilterte Datensichten (auf Basis IMDb + „sanft“ ergänzt)
    @reactive.Calc
    def df_joined():
        df = joined_for_visuals().copy()
        if df.empty:
            return df
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2: y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
        if "numVotes" in df:
            df = df[df["numVotes"].fillna(0) >= mv]
        return df

    @reactive.Calc
    def df_with_rt():
        d = df_joined()
        return d[d["rt_tomato"].notna()] if not d.empty and "rt_tomato" in d.columns else d.iloc[0:0]

    @reactive.Calc
    def df_without_rt():
        d = df_joined()
        return d[d["rt_tomato"].isna()] if not d.empty and "rt_tomato" in d.columns else d

    # Titel
    @output
    @render.ui
    def page_title():
        mapping = {
            "overview":"Übersicht",
            "compare":"Vergleich IMDb ↔ Rotten Tomatoes",
            "coverage":"Abdeckung Rotten Tomatoes",
            "trends":"Trends (Genre & Jahrzehnt)",
            "top20":"Top 20 (IMDb-Stimmen)",
            "gtrends":"Google Trends",
            "table":"Tabelle (gefiltert)",
            "rt_only":"Rotten Tomatoes — eigene Sicht"
        }
        return ui.tags.h3(mapping.get(input.page(),"Übersicht"))

    # Routing
    @output
    @render.ui
    def page_body():
        p = input.page()
        if p == "overview":  return ui.div(kpi_ui(), ui.output_plot("p_avg_bars"), ui.output_plot("p_vote_ecdf"))
        if p == "compare":   return ui.div(ui.output_plot("p_scatter_hex"), ui.output_plot("p_diff_box"))
        if p == "coverage":  return ui.div(ui.output_plot("p_coverage_share"), ui.output_data_frame("tbl_missing_rt"))
        if p == "trends":    return ui.div(ui.output_plot("p_genre_avg"), ui.output_plot("p_decade_avg"))
        if p == "top20":     return ui.div(ui.output_plot("p_top20"), ui.output_data_frame("tbl_top20"))
        if p == "gtrends":   return ui.div(ui.output_plot("p_gtrends"), ui.output_data_frame("tbl_gtrends"))
        if p == "rt_only":   return ui.div(ui.output_plot("p_rt_only_avg"), ui.output_plot("p_rt_aud_vs_tomato"))
        if p == "table":     return ui.div(ui.output_data_frame("tbl_all"))
        return ui.div("—")

    # KPIs
    def kpi_ui():
        d_all = df_joined()
        d_rt  = df_with_rt()
        cards = []
        def vb(title, value): return ui.value_box(title=title, value=value)
        cards.append(vb("Filme (gefiltert)", f"{len(d_all):,}".replace(",", ".")))
        if "averageRating" in d_all:
            imdb_mean = d_all["averageRating"].dropna().mean()*10
            cards.append(vb("Ø IMDb (x10)", f"{imdb_mean:.1f}" if not pd.isna(imdb_mean) else "—"))
        if not d_rt.empty:
            rt_mean = d_rt["rt_tomato"].dropna().mean()
            cards.append(vb("Ø RT Tomatometer", f"{rt_mean:.1f}"))
            if input.use_audience() and "rt_audience" in d_rt:
                aud_mean = d_rt["rt_audience"].dropna().mean()
                cards.append(vb("Ø RT Audience", f"{aud_mean:.1f}" if not pd.isna(aud_mean) else "—"))
        share = (len(d_rt)/len(d_all)*100) if len(d_all)>0 else 0
        cards.append(vb("RT-Abdeckung", f"{share:.1f}%"))
        return ui.layout_column_wrap(*cards, fill=False)

    # Übersicht: Balken + ECDF (keine Histogramme)
    @output
    @render.plot
    def p_avg_bars():
        d_all = df_joined()
        d_rt  = df_with_rt()
        vals = {}
        if "averageRating" in d_all and d_all["averageRating"].notna().any():
            vals["IMDb (x10)"] = d_all["averageRating"].mean()*10
        if not d_rt.empty and d_rt["rt_tomato"].notna().any():
            vals["RT Tomatometer"] = d_rt["rt_tomato"].mean()
            if input.use_audience() and "rt_audience" in d_rt and d_rt["rt_audience"].notna().any():
                vals["RT Audience"] = d_rt["rt_audience"].mean()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if not vals:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_ylim(0,100); ax.set_title("Durchschnittliche Bewertung (0–100)"); ax.set_ylabel("Score")
        return fig

    @output
    @render.plot
    def p_vote_ecdf():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "numVotes" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Stimmen-Daten",ha="center",va="center"); return fig
        x = np.log10(d["numVotes"].clip(lower=1)).dropna().to_numpy()
        if x.size == 0:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        xs = np.sort(x); ys = np.arange(1, xs.size+1) / xs.size
        ax.plot(xs, ys)
        ax.set_xlabel("log10(Stimmen)"); ax.set_ylabel("Anteil ≤ x")
        ax.set_title("ECDF: Verteilung der IMDb-Stimmen"); ax.set_ylim(0,1)
        return fig

    # Vergleich: Hexbin + Boxplot
    @output
    @render.plot
    def p_scatter_hex():
        d = df_with_rt().dropna(subset=["averageRating","rt_tomato"])
        fig,ax=plt.subplots(figsize=(7.5,6))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Schnittmenge (IMDb & RT)",ha="center",va="center"); return fig
        x = d["averageRating"]*10; y = d["rt_tomato"]
        hb = ax.hexbin(x, y, gridsize=30, extent=[0,100,0,100], linewidths=0.2)
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer")
        if len(x) >= 2:
            m,b = np.polyfit(x, y, 1); xs = np.linspace(0,100,200); ax.plot(xs, m*xs + b)
            r = np.corrcoef(x, y)[0,1]; ax.set_title(f"IMDb vs RT — Regr. y={m:.2f}x+{b:.1f}, r={r:.2f}")
        else:
            ax.set_title("IMDb vs RT")
        fig.colorbar(hb, ax=ax, label="Dichte")
        return fig

    @output
    @render.plot
    def p_diff_box():
        d = df_with_rt().dropna(subset=["averageRating","rt_tomato"])
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        data = [d["rt_tomato"] - d["averageRating"]*10]; labels = ["RT − IMDb(x10)"]
        if "rt_audience" in d.columns and input.use_audience():
            data.append(d["rt_audience"] - d["averageRating"]*10)
            labels.append("Audience − IMDb(x10)")
        ax.boxplot(data, showmeans=True, vert=True)
        ax.set_xticklabels(labels, rotation=0)
        ax.axhline(0, color="k", linewidth=1)
        ax.set_ylabel("Punkte"); ax.set_title("Differenz zu IMDb (x10)")
        return fig

    # Abdeckung RT
    @output
    @render.plot
    def p_coverage_share():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "year" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        tmp = d.dropna(subset=["year"]).copy()
        tmp["has_rt"] = tmp["rt_tomato"].notna() if "rt_tomato" in tmp.columns else False
        share = (tmp.groupby("year")["has_rt"].mean()*100).sort_index()
        if share.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        ax.plot(share.index, share.values, marker="o")
        ax.set_ylim(0,100); ax.set_xlabel("Jahr"); ax.set_ylabel("RT-Abdeckung (%)")
        ax.set_title("Abdeckung Rotten Tomatoes im Zeitverlauf")
        return fig

    @output
    @render.data_frame
    def tbl_missing_rt():
        d = df_without_rt().copy()
        if d.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","Stimmen","genres","tconst"])
        if "averageRating" in d: d["IMDb (x10)"] = (d["averageRating"]*10).round(1)
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols = ["Titel","Jahr","IMDb (x10)","Stimmen","genres","tconst"]
        for c in cols:
            if c not in d.columns: d[c]=pd.NA
        return d[cols].sort_values("Stimmen",ascending=False).head(200)

    # Trends (IMDb)
    @output
    @render.plot
    def p_genre_avg():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "genres" not in d or "averageRating" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten",ha="center",va="center"); return fig
        g = d[d.get("numVotes",pd.Series()).fillna(0)>=50_000].dropna(subset=["genres"]).assign(
            genre=lambda x: x["genres"].astype(str).str.split(",")
        ).explode("genre")
        if g.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten (≥50k)",ha="center",va="center"); return fig
        avg = (g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12))*10
        ax.bar(avg.index, avg.values); ax.set_ylim(0,100)
        ax.set_xticklabels(avg.index, rotation=45, ha="right")
        ax.set_ylabel("Ø (0–100)"); ax.set_title("Ø IMDb (x10) nach Genre (Top 12, ≥50k Stimmen)")
        return fig

    @output
    @render.plot
    def p_decade_avg():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "year" not in d or "averageRating" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        dec = d.dropna(subset=["year"]).copy()
        dec["decade"] = (dec["year"].astype(int)//10)*10
        avg=(dec.groupby("decade")["averageRating"].mean().sort_index())*10
        if avg.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        ax.plot(avg.index, avg.values, marker="o"); ax.set_ylim(0,100)
        ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø (0–100)")
        ax.set_title("Ø IMDb (x10) nach Jahrzehnt")
        return fig

    # RT-only Seite (ohne IMDb, direkt aus rt_std)
    @output
    @render.plot
    def p_rt_only_avg():
        d = rt_std.copy()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Daten",ha="center",va="center"); return fig
        vals={}
        if d["rt_tomato"].notna().any(): vals["Tomatometer"] = d["rt_tomato"].mean()
        if input.use_audience() and "rt_audience" in d and d["rt_audience"].notna().any():
            vals["Audience"] = d["rt_audience"].mean()
        if not vals:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Werte",ha="center",va="center"); return fig
        ax.bar(list(vals.keys()), list(vals.values())); ax.set_ylim(0,100)
        ax.set_title("RT (nur Rohdaten) — Ø Werte"); ax.set_ylabel("Score")
        return fig

    @output
    @render.plot
    def p_rt_aud_vs_tomato():
        d = rt_std.dropna(subset=["rt_tomato","rt_audience"])
        fig,ax=plt.subplots(figsize=(7.5,6))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Audience/Tomato Schnittmenge",ha="center",va="center"); return fig
        ax.hexbin(d["rt_tomato"], d["rt_audience"], gridsize=30, extent=[0,100,0,100], linewidths=0.2)
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        ax.set_xlabel("Tomatometer"); ax.set_ylabel("Audience")
        ax.set_title("RT Audience vs Tomatometer")
        return fig

    # Top 20
    @output
    @render.plot
    def p_top20():
        d = top20_raw.copy()
        fig,ax=plt.subplots(figsize=(9,5))
        need = {"title","year","numVotes"}
        if d.empty or not need.issubset(set(d.columns)):
            ax.axis("off"); ax.text(0.5,0.5,"Keine Top-20-Daten",ha="center",va="center"); return fig
        d = d.sort_values("numVotes", ascending=True).tail(20)
        labels = d["title"].astype(str) + " (" + d["year"].astype(str) + ")"
        ax.barh(labels, d["numVotes"].values)
        ax.set_xlabel("Stimmen (IMDb)"); ax.set_title("Top 20 nach IMDb-Stimmen")
        return fig

    @output
    @render.data_frame
    def tbl_top20():
        d = top20_raw.copy()
        if d.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","Stimmen","tconst"])
        d = d.sort_values("numVotes", ascending=False).head(20)
        if "averageRating" in d: d["IMDb (x10)"] = (d["averageRating"]*10).round(1)
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols=["Titel","Jahr","IMDb (x10)","Stimmen","tconst"]
        for c in cols:
            if c not in d.columns: d[c]=pd.NA
        return d[cols]

    # Google Trends
    @output
    @render.plot
    def p_gtrends():
        d = gtr_raw.copy()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Trends-Daten",ha="center",va="center"); return fig
        if "date" in d.columns: d = d.set_index("date")
        else: d = d.set_index(d.columns[0])
        for c in d.columns:
            ax.plot(pd.to_datetime(d.index), d[c], label=c)
        ax.legend(); ax.set_title("Google Trends (Top 5)"); ax.set_xlabel("Datum"); ax.set_ylabel("Interesse (0–100)")
        return fig

    @output
    @render.data_frame
    def tbl_gtrends():
        d = gtr_raw.copy()
        return d if not d.empty else pd.DataFrame(columns=["date","kw1","kw2","kw3","kw4","kw5"])

    # Tabelle (gefiltert)
    @output
    @render.data_frame
    def tbl_all():
        d = df_joined().copy()
        if d.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"])
        if "averageRating" in d: d["IMDb (x10)"] = (d["averageRating"]*10).round(1)
        if "rt_tomato" in d: d["RT Tomatometer"] = d["rt_tomato"].round(1)
        if "rt_audience" in d: d["RT Audience"] = d["rt_audience"].round(1)
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]
        for c in cols:
            if c not in d.columns: d[c]=pd.NA
        return d[cols].sort_values("Stimmen", ascending=False)

app = App(app_ui, server)
