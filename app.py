# app.py — Movie Ratings Dashboard (einfach & klar)
# CSVs:
#   outputs/joined_imdb_rt.csv
#   outputs/rotten_tomatoes_movies.csv (oder raw/... als Fallback)
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

# Warnings reduzieren
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("movie-app")

# ---------------- Pfade/Loader ----------------
BASE = Path(__file__).resolve().parent

def _find(*candidates: Path) -> Path|None:
    for p in candidates:
        if p.exists():
            return p
    return None

CSV_JOINED = _find(BASE/"outputs/joined_imdb_rt.csv", BASE/"joined_imdb_rt.csv")
CSV_RT_RAW = _find(BASE/"outputs/rotten_tomatoes_movies.csv", BASE/"raw/rotten_tomatoes_movies.csv")
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
    """Standardisiere RT-Rohdaten → title_norm, year, rt_tomato, rt_audience (nur bis 2020)."""
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
    # Auf 2020 begrenzen
    d = d[d["year"].fillna(0) <= 2020]
    d["rt_tomato"]=d.get("rt_tomato_raw",pd.Series(index=d.index)).map(to100)
    d["rt_audience"]=d.get("rt_audience_raw",pd.Series(index=d.index)).map(to100)
    out = d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()
    return out

# --- vorbereiten: joined (IMDb) + RT-Standard
if not joined_raw.empty and "title_norm" not in joined_raw.columns and "title" in joined_raw.columns:
    joined_raw["title_norm"] = joined_raw["title"].map(norm_title)

rt_std = std_rt(rt_raw)

def joined_for_visuals() -> pd.DataFrame:
    """IMDb-Daten + (sanft) RT-Ergänzung. Einheitliche Spaltennamen für UI."""
    if joined_raw.empty:
        return joined_raw
    df = joined_raw.copy()
    # Einheitliche Namen
    if "year" not in df.columns and "Year" in df.columns:
        df.rename(columns={"Year":"year"}, inplace=True)
    if "numVotes" not in df.columns and "votes" in df.columns:
        df.rename(columns={"votes":"numVotes"}, inplace=True)

    # merge RT (nur bis 2020)
    cols = ["title_norm","year"]
    if "year" in df: df["year"]=pd.to_numeric(df["year"], errors="coerce")
    if not rt_std.empty and all(c in df.columns for c in cols):
        df = df.merge(rt_std, on=cols, how="left", suffixes=("","_rtstd"))
        if "rt_tomato" not in df and "rt_tomato_rtstd" in df:
            df["rt_tomato"] = df["rt_tomato_rtstd"]
        else:
            df["rt_tomato"] = df["rt_tomato"].fillna(df.get("rt_tomato_rtstd"))
        if "rt_audience" not in df and "rt_audience_rtstd" in df:
            df["rt_audience"] = df["rt_audience_rtstd"]
        else:
            df["rt_audience"] = df["rt_audience"].fillna(df.get("rt_audience_rtstd"))

    # Einheitliche, DEUTLICHE Namen für UI/Plots
    if "averageRating" in df: df["IMDb_Score_100"] = (df["averageRating"]*10).clip(0,100)
    if "rt_tomato" in df:     df["RT_Tomatometer"] = df["rt_tomato"].clip(0,100)
    if "rt_audience" in df:   df["RT_Audience"] = df["rt_audience"].clip(0,100)
    if "numVotes" in df:      df["IMDb_Votes"] = df["numVotes"].astype("Int64")
    return df

# ---------------- UI ----------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style("""
        :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3}
        html,body{height:100%;background:var(--bg);color:#0f172a}
        .sidebar{background:var(--card);border-right:1px solid var(--line);
                 height:100vh;max-height:100vh;overflow:auto;position:sticky;top:0}
        .muted{color:var(--muted);font-size:12px}
        """),
        ui.tags.h2("🎬 Movie Ratings"),
        ui.tags.div("Navigation", class_="muted"),
        ui.input_radio_buttons(
            "page", None,
            choices={
                "overview":"Übersicht",
                "compare":"IMDb ↔ RT",
                "coverage":"Abdeckung RT",
                "trends":"Trends",
                "top20":"Top 20",
                "gtrends":"Google Trends",
                "rt_only":"RT (nur RT-Daten)",
                "table":"Tabelle",
            },
            selected="overview", inline=False
        ),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end",   "Jahr bis",  2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes",  "Mind. IMDb-Stimmen (Filter)", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "RT Audience mit anzeigen", True),
        ui.tags.hr(),
        ui.tags.div("Status", class_="muted"),
        ui.output_ui("status_files"),
    ),
    ui.layout_column_wrap(
        ui.card(ui.card_header(ui.tags.div(id="page_title")), ui.output_ui("page_body")),
        fill=False,
    ),
    title="Movie Ratings Dashboard",
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

    # Gefilterte Datensichten
    @reactive.Calc
    def df_joined():
        df = joined_for_visuals().copy()
        if df.empty:
            return df
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2: y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
        if "IMDb_Votes" in df:
            df = df[df["IMDb_Votes"].fillna(0) >= mv]
        return df

    @reactive.Calc
    def df_with_rt():
        d = df_joined()
        col = "RT_Tomatometer"
        return d[d[col].notna()] if not d.empty and col in d.columns else d.iloc[0:0]

    @reactive.Calc
    def df_without_rt():
        d = df_joined()
        col = "RT_Tomatometer"
        return d[d[col].isna()] if not d.empty and col in d.columns else d

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
            "rt_only":"Rotten Tomatoes — eigene Sicht",
            "table":"Tabelle (gefiltert)",
        }
        return ui.tags.h3(mapping.get(input.page(),"Übersicht"))

    # Routing
    @output
    @render.ui
    def page_body():
        p = input.page()
        if p == "overview":  return ui.div(kpi_ui(), ui.output_plot("p_avg_bars"), ui.output_plot("p_vote_ccdf"))
        if p == "compare":   return ui.div(ui.output_plot("p_scatter_simple"), ui.output_plot("p_mean_diff_by_bin"))
        if p == "coverage":  return ui.div(ui.output_plot("p_coverage_share"), ui.output_plot("p_rt_pie"))
        if p == "trends":    return ui.div(ui.output_plot("p_genre_avg"), ui.output_plot("p_decade_avg"))
        if p == "top20":     return ui.div(ui.output_plot("p_top20"), ui.output_data_frame("tbl_top20"))
        if p == "gtrends":   return ui.div(ui.output_plot("p_gtrends"), ui.output_data_frame("tbl_gtrends"))
        if p == "rt_only":   return ui.div(ui.output_plot("p_rt_only_avg"))
        if p == "table":     return ui.div(ui.output_data_frame("tbl_all"))
        return ui.div("—")

    # KPIs
    def kpi_ui():
        d_all = df_joined()
        d_rt  = df_with_rt()
        cards = []
        def vb(title, value): return ui.value_box(title=title, value=value)
        cards.append(vb("Filme (gefiltert)", f"{len(d_all):,}".replace(",", ".")))
        if "IMDb_Score_100" in d_all:
            imdb_mean = d_all["IMDb_Score_100"].dropna().mean()
            cards.append(vb("Ø IMDb (0–100)", f"{imdb_mean:.1f}" if not pd.isna(imdb_mean) else "—"))
        if not d_rt.empty:
            rt_mean = d_rt["RT_Tomatometer"].dropna().mean()
            cards.append(vb("Ø RT Tomatometer", f"{rt_mean:.1f}"))
            if input.use_audience() and "RT_Audience" in d_rt:
                aud_mean = d_rt["RT_Audience"].dropna().mean()
                cards.append(vb("Ø RT Audience", f"{aud_mean:.1f}" if not pd.isna(aud_mean) else "—"))
        share = (len(d_rt)/len(d_all)*100) if len(d_all)>0 else 0
        cards.append(vb("RT-Abdeckung", f"{share:.1f}%"))
        return ui.layout_column_wrap(*cards, fill=False)

    # Übersicht: Ø-Balken + CCDF (einfach)
    @output
    @render.plot
    def p_avg_bars():
        d_all = df_joined()
        d_rt  = df_with_rt()
        fig, ax = plt.subplots(figsize=(9,3.8))
        labels, values = [], []
        if "IMDb_Score_100" in d_all and d_all["IMDb_Score_100"].notna().any():
            labels.append(f"IMDb\nN={d_all['IMDb_Score_100'].notna().sum():,}".replace(",", "."))
            values.append(d_all["IMDb_Score_100"].mean())
        if not d_rt.empty and d_rt["RT_Tomatometer"].notna().any():
            labels.append(f"RT Tomatometer\nN={d_rt['RT_Tomatometer'].notna().sum():,}".replace(",", "."))
            values.append(d_rt["RT_Tomatometer"].mean())
        if input.use_audience() and "RT_Audience" in d_rt and d_rt["RT_Audience"].notna().any():
            labels.append(f"RT Audience\nN={d_rt['RT_Audience'].notna().sum():,}".replace(",", "."))
            values.append(d_rt["RT_Audience"].mean())
        if not values:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        colors = ["#2563EB","#F59E0B","#10B981"][:len(values)]  # kräftig
        ax.bar(labels, values, color=colors)
        for i,v in enumerate(values):
            ax.text(i, v+1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0,100)
        ax.set_title("Durchschnittliche Bewertungen (0–100)")
        ax.set_ylabel("Punkte")
        return fig

    @output
    @render.plot
    def p_vote_ccdf():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9,3.8))
        if d.empty or "IMDb_Votes" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Stimmen-Daten",ha="center",va="center"); return fig
        x = np.log10(d["IMDb_Votes"].clip(lower=1)).dropna().to_numpy()
        if x.size == 0:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        xs = np.sort(x); ys = np.arange(1, xs.size+1) / xs.size
        ax.plot(xs, 1-ys, color="#2563EB", linewidth=2)  # kräftig
        ax.set_xlabel("log10(Stimmen)  (3=1.000, 4=10.000, 5=100.000)")
        ax.set_ylabel("Anteil der Filme ≥ X")
        ax.set_title(f"Wie viele Stimmen haben die Filme? (N={x.size:,})".replace(",", "."))
        ax.set_ylim(0,1)
        return fig

    # Vergleich: einfacher Scatter + Mittelwert-Differenz pro IMDb-Bin (Linie)
    @output
    @render.plot
    def p_scatter_simple():
        d = df_with_rt().dropna(subset=["IMDb_Score_100","RT_Tomatometer"])
        fig, ax = plt.subplots(figsize=(9,5))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Schnittmenge (IMDb & RT)",ha="center",va="center"); return fig
        ax.scatter(d["IMDb_Score_100"], d["RT_Tomatometer"], s=10, alpha=0.4, color="#4F46E5")
        ax.plot([0,100],[0,100], linestyle="--", linewidth=1.5, color="#F59E0B", alpha=0.9)
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        ax.set_xlabel("IMDb (0–100)")
        ax.set_ylabel("RT Tomatometer (0–100)")
        ax.set_title(f"IMDb vs. Rotten Tomatoes (N={len(d):,})".replace(",", "."))
        return fig

    @output
    @render.plot
    def p_mean_diff_by_bin():
        d = df_with_rt().dropna(subset=["IMDb_Score_100","RT_Tomatometer"])
        fig, ax = plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        # IMDb in 10-Punkte-Bins und Mittelwert der Differenz pro Bin
        d["bin"] = (d["IMDb_Score_100"]//10*10).astype(int).clip(0,90)
        s = d.groupby("bin").apply(lambda x: (x["RT_Tomatometer"] - x["IMDb_Score_100"]).mean())
        ax.plot(s.index, s.values, marker="o", color="#EF4444", linewidth=2)  # kräftig
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xticks(list(range(0,101,10)))
        ax.set_xlabel("IMDb (gerundet, 0–100)")
        ax.set_ylabel("Ø [RT − IMDb] (Punkte)")
        ax.set_title("Wo weichen RT und IMDb ab? (Ø Differenz je IMDb-Bin)")
        return fig

    # Abdeckung + RT-Kreisdiagramm
    @output
    @render.plot
    def p_coverage_share():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9,3.8))
        if d.empty or "year" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        tmp = d.dropna(subset=["year"]).copy()
        tmp["has_rt"] = tmp["RT_Tomatometer"].notna() if "RT_Tomatometer" in tmp.columns else False
        share = (tmp.groupby("year")["has_rt"].mean()*100).sort_index()
        if share.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        ax.plot(share.index, share.values, marker="o", color="#2563EB", linewidth=2)
        ax.set_ylim(0,100); ax.set_xlabel("Jahr"); ax.set_ylabel("RT-Abdeckung (%)")
        ax.set_title("Wie oft gibt es RT-Bewertungen? (Anteil pro Jahr)")
        return fig

    @output
    @render.plot
    def p_rt_pie():
        d = df_with_rt()
        fig, ax = plt.subplots(figsize=(6,6))
        if d.empty or "RT_Tomatometer" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Daten",ha="center",va="center"); return fig
        cats = pd.Series(np.where(d["RT_Tomatometer"]>=60, "Fresh", "Rotten"))
        counts = cats.value_counts().reindex(["Fresh","Rotten"]).fillna(0)
        if counts.sum() == 0:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Werte",ha="center",va="center"); return fig
        colors = ["#10B981","#EF4444"]  # grün/rot
        ax.pie(counts.values, labels=[f"{k} ({int(v)})" for k,v in counts.items()], autopct="%1.0f%%",
               startangle=90, colors=colors, textprops={"fontsize":11})
        ax.set_title("RT-Verteilung (Fresh ≥ 60)")
        ax.axis("equal")
        return fig

    # Trends (IMDb)
    @output
    @render.plot
    def p_genre_avg():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "genres" not in d or "IMDb_Score_100" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten",ha="center",va="center"); return fig
        g = d[d.get("IMDb_Votes",pd.Series(dtype=float)).fillna(0)>=50_000].dropna(subset=["genres"]).assign(
            genre=lambda x: x["genres"].astype(str).str.split(",")
        ).explode("genre")
        if g.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten (≥50k)",ha="center",va="center"); return fig
        s = g.groupby("genre").agg(avg=("IMDb_Score_100","mean"), n=("IMDb_Score_100","size"))
        s = s.sort_values("avg", ascending=False).head(12)
        ax.bar(s.index, s["avg"], color="#2563EB")
        for i,(v,n) in enumerate(zip(s["avg"], s["n"])):
            ax.text(i, v+1, f"{v:.1f}\nN={n}", ha="center", fontsize=8)
        ax.set_ylim(0,100); ax.set_xticklabels(s.index, rotation=45, ha="right")
        ax.set_ylabel("Ø IMDb (0–100)")
        ax.set_title("Welche Genres schneiden gut ab? (Filme mit ≥50k Stimmen)")
        return fig

    @output
    @render.plot
    def p_decade_avg():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9,3.8))
        if d.empty or "year" not in d or "IMDb_Score_100" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        dec = d.dropna(subset=["year"]).copy()
        dec["decade"] = (dec["year"].astype(int)//10)*10
        s = dec.groupby("decade").agg(avg=("IMDb_Score_100","mean"), n=("IMDb_Score_100","size")).sort_index()
        ax.plot(s.index, s["avg"], marker="o", color="#F59E0B", linewidth=2)
        for x,y,n in zip(s.index, s["avg"], s["n"]):
            ax.text(x, y+1, f"{y:.1f}\nN={n}", ha="center", fontsize=8)
        ax.set_ylim(0,100)
        ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø IMDb (0–100)")
        ax.set_title("Wie haben sich Ø-Bewertungen je Jahrzehnt entwickelt?")
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
        labels = d["title"].astype(str) + " (" + d["year"].astype(int).astype(str) + ")"
        ax.barh(labels, d["numVotes"].values, color="#2563EB")
        ax.set_xlabel("Stimmen (IMDb)"); ax.set_title("Top 20 nach IMDb-Stimmen")
        return fig

    @output
    @render.data_frame
    def tbl_top20():
        d = top20_raw.copy()
        if d.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb_Score_100","IMDb_Votes","tconst"])
        d = d.sort_values("numVotes", ascending=False).head(20)
        if "averageRating" in d: d["IMDb_Score_100"] = (d["averageRating"]*10).round(1)
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"IMDb_Votes"})
        cols=["Titel","Jahr","IMDb_Score_100","IMDb_Votes","tconst"]
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
        for i,c in enumerate(d.columns):
            ax.plot(pd.to_datetime(d.index), d[c], label=c, linewidth=2)
        ax.legend(fontsize=8); ax.set_title("Google Trends (Top 5)")
        ax.set_xlabel("Datum"); ax.set_ylabel("Interesse (0–100)")
        return fig

    @output
    @render.data_frame
    def tbl_gtrends():
        d = gtr_raw.copy()
        return d if not d.empty else pd.DataFrame(columns=["date","kw1","kw2","kw3","kw4","kw5"])

    # Tabelle (gefiltert) – eindeutige Spalten
    @output
    @render.data_frame
    def tbl_all():
        d = df_joined().copy()
        if d.empty:
            return pd.DataFrame(columns=[
                "Titel","Jahr","IMDb_Score_100","RT_Tomatometer","RT_Audience","IMDb_Votes","genres","tconst"
            ])
        d = d.rename(columns={"title":"Titel","year":"Jahr"})
        cols=["Titel","Jahr","IMDb_Score_100","RT_Tomatometer","RT_Audience","IMDb_Votes","genres","tconst"]
        for c in cols:
            if c not in d.columns: d[c]=pd.NA
        return d[cols].sort_values("IMDb_Votes", ascending=False)

    # RT-only Seite (nur RT-Rohdaten, bis 2020)
    @output
    @render.plot
    def p_rt_only_avg():
        d = rt_std.copy()
        fig, ax = plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Daten",ha="center",va="center"); return fig
        labels, values, colors = [], [], []
        if d["rt_tomato"].notna().any():
            labels.append(f"Tomatometer\nN={d['rt_tomato'].notna().sum():,}".replace(",", "."))
            values.append(d["rt_tomato"].mean()); colors.append("#F59E0B")
        if input.use_audience() and d["rt_audience"].notna().any():
            labels.append(f"Audience\nN={d['rt_audience'].notna().sum():,}".replace(",", "."))
            values.append(d["rt_audience"].mean()); colors.append("#10B981")
        if not values:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Werte",ha="center",va="center"); return fig
        ax.bar(labels, values, color=colors)
        for i,v in enumerate(values):
            ax.text(i, v+1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0,100); ax.set_title("Rotten Tomatoes (nur RT-Daten, bis 2020) — Ø Werte"); ax.set_ylabel("Punkte")
        return fig

app = App(app_ui, server)
