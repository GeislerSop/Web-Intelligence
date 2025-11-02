# app.py — Movie Ratings Dashboard (Posit Connect)
# nutzt 3 CSVs aus ./outputs:
#   joined_imdb_rt.csv | top20_by_votes_imdb.csv | google_trends_top5.csv

from __future__ import annotations
import os, sys, re, logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("movie-app")

# ---------------- Pfade + Loader ----------------
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [BASE_DIR / "outputs", Path.cwd() / "outputs", BASE_DIR]
FILES = {
    "joined": "joined_imdb_rt.csv",
    "top20":  "top20_by_votes_imdb.csv",
    "gtr":    "google_trends_top5.csv",
}

def _find(key: str) -> Path:
    fname = FILES[key]
    for root in CANDIDATES:
        p = root / fname
        if p.exists():
            LOG.info(f"Found {key}: {p}")
            return p
    LOG.warning(f"{key} not found in {CANDIDATES}; fallback to outputs/{fname}")
    return Path("outputs") / fname

CSV_JOINED = _find("joined")
CSV_TOP20  = _find("top20")
CSV_GT     = _find("gtr")

def read_csv_local(path: Path | str) -> pd.DataFrame:
    try:
        p = str(path) if isinstance(path, str) else str(path)
        # falls Deine CSVs Semikolon-getrennt sind: sep=";"
        df = pd.read_csv(p)
        LOG.info(f"Loaded {p} → shape={df.shape}")
        return df
    except Exception as e:
        LOG.exception(f"Failed to read {path}: {e}")
        return pd.DataFrame()

# ---------------- Daten laden ----------------
joined_raw = read_csv_local(CSV_JOINED)
top20_raw  = read_csv_local(CSV_TOP20)
gtr_raw    = read_csv_local(CSV_GT)

SAMPLE_MAX = 80_000

def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = re.sub(r"[^a-z0-9 ]+"," ", str(t).lower())
    return " ".join(t.split())

if not joined_raw.empty and "title_norm" not in joined_raw.columns and "title" in joined_raw.columns:
    joined_raw["title_norm"] = joined_raw["title"].map(norm_title)
if not joined_raw.empty and len(joined_raw) > SAMPLE_MAX:
    joined_raw = joined_raw.sample(SAMPLE_MAX, random_state=42)

# ---------------- kleine Plot-Helper ----------------
def safe_hist(ax, data, *, bins=30, label=None, alpha=0.7, density=False, xlim=None, title=None, xlabel=None, ylabel=None):
    """Histogram ohne RuntimeWarning, wenn Daten leer/konstant sind."""
    s = pd.Series(data).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
        return
    # density nur, wenn Summe > 0
    use_density = bool(density) and (s.size > 0) and (s.sum() != 0)
    ax.hist(s.values, bins=bins, alpha=alpha, label=label, density=use_density)
    if label: ax.legend()
    if xlim: ax.set_xlim(*xlim)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

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
        @media (max-width: 991px){ .sidebar{position:fixed;width:78%;} }
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
        ui.card(
            ui.card_header(ui.tags.div(id="page_title")),
            ui.output_ui("page_body"),
        ),
        fill=False,
    ),
    title="Movie Ratings Dashboard",
    open={"desktop":"open","mobile":"closed"}
)

# ---------------- Server ----------------
def server(input: Inputs, output: Outputs, session: Session):

    # -------- Status ----------
    @output
    @render.ui
    def status_files():
        def li(ok, text): return ui.tags.li((("✅ " if ok else "❌ ") + text))
        rows = [
            li(Path(CSV_JOINED).exists(), f"joined: {CSV_JOINED} (shape={tuple(joined_raw.shape)})"),
            li(Path(CSV_TOP20).exists(),  f"top20 : {CSV_TOP20}  (shape={tuple(top20_raw.shape)})"),
            li(Path(CSV_GT).exists(),     f"gtr   : {CSV_GT}     (shape={tuple(gtr_raw.shape)})"),
        ]
        return ui.tags.small(ui.tags.ul(*rows, style="margin:0;padding-left:18px;"))

    # -------- Filter ----------
    @reactive.Calc
    def df_joined():
        df = joined_raw.copy()
        if df.empty:
            return df
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2: y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0)>=y1) & (df["year"].fillna(0)<=y2)]
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

    # -------- Titel ----------
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
        }
        return ui.tags.h3(mapping.get(input.page(),"Übersicht"))

    # -------- Routing ----------
    @output
    @render.ui
    def page_body():
        p = input.page()
        if p == "overview":  return ui.div(kpi_ui(), ui.output_plot("p_avg_bars"), ui.output_plot("p_vote_hist"))
        if p == "compare":   return ui.div(ui.output_plot("p_scatter_hex"), ui.output_plot("p_diff_hist"))
        if p == "coverage":  return ui.div(ui.output_plot("p_coverage_share"), ui.output_data_frame("tbl_missing_rt"))
        if p == "trends":    return ui.div(ui.output_plot("p_genre_avg"), ui.output_plot("p_decade_avg"))
        if p == "top20":     return ui.div(ui.output_plot("p_top20"), ui.output_data_frame("tbl_top20"))
        if p == "gtrends":   return ui.div(ui.output_plot("p_gtrends"), ui.output_data_frame("tbl_gtrends"))
        if p == "table":     return ui.div(ui.output_data_frame("tbl_all"))
        return ui.div("—")

    # -------- KPIs ----------
    def kpi_ui():
        d_all = df_joined()
        d_rt  = df_with_rt()
        cards = []
        def vb(title, value): return ui.value_box(title=title, value=value)
        cards.append(vb("Filme (gefiltert)", f"{len(d_all):,}".replace(",", ".")))
        imdb_mean = d_all["averageRating"].dropna().mean()*10 if "averageRating" in d_all else np.nan
        cards.append(vb("Ø IMDb (x10)", "—" if pd.isna(imdb_mean) else f"{imdb_mean:.1f}"))
        if not d_rt.empty:
            rt_mean = d_rt["rt_tomato"].dropna().mean()
            cards.append(vb("Ø RT Tomatometer", f"{rt_mean:.1f}"))
            if input.use_audience() and "rt_audience" in d_rt:
                aud_mean = d_rt["rt_audience"].dropna().mean()
                cards.append(vb("Ø RT Audience", "—" if pd.isna(aud_mean) else f"{aud_mean:.1f}"))
        share = (len(d_rt)/len(d_all)*100) if len(d_all)>0 else 0
        cards.append(vb("RT-Abdeckung", f"{share:.1f}%"))
        return ui.layout_column_wrap(*cards, fill=False)

    # -------- Übersicht: Ø-Balken + Stimmen-Hist --------
    @output
    @render.plot
    def p_avg_bars():
        d_all = df_joined()
        d_rt  = df_with_rt()
        vals = {}
        if "averageRating" in d_all: vals["IMDb (x10)"] = d_all["averageRating"].dropna().mean()*10
        if not d_rt.empty:
            vals["RT Tomatometer"] = d_rt["rt_tomato"].dropna().mean()
            if input.use_audience() and "rt_audience" in d_rt:
                vals["RT Audience"] = d_rt["rt_audience"].dropna().mean()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if not vals:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        ax.bar(list(vals.keys()), list(vals.values()))
        ax.set_ylim(0,100); ax.set_title("Durchschnittliche Bewertung (0–100)"); ax.set_ylabel("Score")
        return fig

    @output
    @render.plot
    def p_vote_hist():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "numVotes" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Stimmen-Daten",ha="center",va="center"); return fig
        safe_hist(ax, np.log10(d["numVotes"].clip(lower=1)), bins=30,
                  title="Verteilung der IMDb-Stimmen (log10)", xlabel="log10(Stimmen)")
        return fig

    # -------- Vergleich: Hexbin + Differenz-Hist --------
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
        # Regression + r
        if len(x) >= 2:
            m,b = np.polyfit(x, y, 1)
            xs = np.linspace(0,100,200); ax.plot(xs, m*xs + b)
            r = np.corrcoef(x, y)[0,1]
            ax.set_title(f"IMDb vs RT — Regr. y={m:.2f}x+{b:.1f},  r={r:.2f}")
        else:
            ax.set_title("IMDb vs RT")
        fig.colorbar(hb, ax=ax, label="Dichte")
        return fig

    @output
    @render.plot
    def p_diff_hist():
        d = df_with_rt().dropna(subset=["averageRating","rt_tomato"])
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        diff = d["rt_tomato"] - d["averageRating"]*10
        safe_hist(ax, diff, bins=40, density=False, title="Differenz RT − IMDb(x10)",
                  xlabel="Punkte", ylabel="Häufigkeit")
        ax.axvline(0, color="k", linewidth=1)
        return fig

    # -------- Abdeckung RT ----------
    @output
    @render.plot
    def p_coverage_share():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Daten",ha="center",va="center"); return fig
        tmp = d.dropna(subset=["year"]).copy()
        tmp["has_rt"] = tmp["rt_tomato"].notna() if "rt_tomato" in tmp.columns else False
        if tmp.empty or "has_rt" not in tmp:
            ax.axis("off"); ax.text(0.5,0.5,"Keine RT-Infos",ha="center",va="center"); return fig
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
        d["IMDb (x10)"] = (d["averageRating"]*10).round(1) if "averageRating" in d.columns else pd.NA
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols = ["Titel","Jahr","IMDb (x10)","Stimmen","genres","tconst"]
        for c in cols:
            if c not in d.columns: d[c] = pd.NA
        return d[cols].sort_values("Stimmen",ascending=False).head(200)

    # -------- Trends ----------
    @output
    @render.plot
    def p_genre_avg():
        d = df_joined()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty or "genres" not in d or "averageRating" not in d:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Genre-Daten",ha="center",va="center"); return fig
        g = d[d["numVotes"].fillna(0)>=50_000].dropna(subset=["genres"]).assign(
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
        if dec.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Jahresdaten",ha="center",va="center"); return fig
        dec["decade"] = (dec["year"].astype(int)//10)*10
        avg=(dec.groupby("decade")["averageRating"].mean().sort_index())*10
        ax.plot(avg.index, avg.values, marker="o"); ax.set_ylim(0,100)
        ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø (0–100)")
        ax.set_title("Ø IMDb (x10) nach Jahrzehnt")
        return fig

    # -------- Top 20 ----------
    @output
    @render.plot
    def p_top20():
        d = top20_raw.copy()
        fig,ax=plt.subplots(figsize=(9,5))
        if d.empty or "numVotes" not in d or "title" not in d or "year" not in d:
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
        if "averageRating" in d:
            d["IMDb (x10)"] = (d["averageRating"]*10).round(1)
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols=["Titel","Jahr","IMDb (x10)","Stimmen","tconst"]
        for c in cols:
            if c not in d.columns: d[c]=pd.NA
        return d[cols]

    # -------- Google Trends ----------
    @output
    @render.plot
    def p_gtrends():
        d = gtr_raw.copy()
        fig,ax=plt.subplots(figsize=(9,3.8))
        if d.empty:
            ax.axis("off"); ax.text(0.5,0.5,"Keine Trends-Daten",ha="center",va="center"); return fig
        if "date" in d.columns:
            d = d.set_index("date")
        else:
            d = d.set_index(d.columns[0])
        for c in d.columns:
            ax.plot(pd.to_datetime(d.index), d[c], label=c)
        ax.legend(); ax.set_title("Google Trends (Top 5)"); ax.set_xlabel("Datum"); ax.set_ylabel("Interesse (0–100)")
        return fig

    @output
    @render.data_frame
    def tbl_gtrends():
        d = gtr_raw.copy()
        return d if not d.empty else pd.DataFrame(columns=["date","kw1","kw2","kw3","kw4","kw5"])

    # -------- Tabelle (gefiltert) ----------
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
