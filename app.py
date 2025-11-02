# app.py — Movie Ratings Dashboard (Posit Connect-ready)
# Liest CSV-Dateien lokal aus ./outputs und zeigt IMDb, RT und Trends-Analysen.

from __future__ import annotations
import io, re, os, sys, logging
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# --- Logging für Posit Logs ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("app")

# --- Runtime-Erkennung ---
IN_BROWSER = ("__pyodide__" in sys.modules) or os.environ.get("PYODIDE") == "1"

# --- Pfade robust bestimmen ---
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    BASE_DIR / "outputs",
    Path.cwd() / "outputs",
    BASE_DIR,
]

FILENAMES = {
    "joined": "joined_imdb_rt.csv",
    "top20": "top20_by_votes_imdb.csv",
    "gtr": "google_trends_top5.csv",
}

def find_file(name_key: str) -> Path:
    fname = FILENAMES[name_key]
    for root in CANDIDATES:
        p = root / fname
        if p.exists():
            LOG.info(f"Found {name_key} at: {p}")
            return p
    p = Path("outputs") / fname
    LOG.warning(f"{name_key} not found in {CANDIDATES}; using relative path {p}")
    return p

CSV_JOINED = find_file("joined")
CSV_TOP20 = find_file("top20")
CSV_GT = find_file("gtr")

def read_csv_portable(path: Path | str) -> pd.DataFrame:
    """Server: liest CSV direkt, Browser: würde per HTTP fetchen."""
    try:
        p = str(path) if isinstance(path, str) else str(path)
        df = pd.read_csv(p)
        LOG.info(f"Loaded CSV {path} → shape={df.shape}")
        return df
    except Exception as e:
        LOG.exception(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()

# --- Daten laden ---
joined = read_csv_portable(CSV_JOINED)
top20 = read_csv_portable(CSV_TOP20)
gtr = read_csv_portable(CSV_GT)

SAMPLE_MAX = 80000

def norm_title(t: str) -> str:
    if pd.isna(t): return ""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(t).lower())
    return " ".join(t.split())

# --- UI ---
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style("""
        :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3;--brand:#2563eb;--brand-weak:#e8efff}
        body{background:var(--bg);color:#0f172a}
        .sidebar{background:var(--card);border-right:1px solid var(--line);overflow-y:auto}
        .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 18px rgba(16,24,40,.06)}
        .btn{background:var(--brand);color:#fff;border-radius:10px;border:none}
        .muted{color:var(--muted);font-size:12px}
        """),
        ui.tags.h2("🎬 Movie Ratings Dashboard", style="margin:6px 0 18px 0;"),
        ui.tags.div("Filter", class_="muted", style="margin-bottom:6px;"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end", "Jahr bis", 2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes", "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "Audience-Score zusätzlich", False),
        ui.tags.hr(),
        ui.tags.div("Status", class_="muted"),
        ui.output_ui("status_files"),
        open={"desktop": "open", "mobile": "closed"},
    ),
    ui.layout_column_wrap(
        ui.card(
            ui.card_header(ui.tags.div(id="page_title")),
            ui.output_ui("page_body"),
        ),
        fill=False,
    ),
    title="Movie Ratings Dashboard",
)

# --- Server ---
def server(input: Inputs, output: Outputs, session: Session):

    store = reactive.Value({"df": joined.copy(), "ready": not joined.empty})

    @output
    @render.ui
    def status_files():
        rows = []
        for name, path in {"joined": CSV_JOINED, "top20": CSV_TOP20, "gtr": CSV_GT}.items():
            ok = "✅" if Path(path).exists() else "❌"
            rows.append(f"{ok} {name}: {path}")
        return ui.tags.small(ui.tags.ul(*[ui.tags.li(r) for r in rows],
                                        style="margin:0;padding-left:18px;"))

    # --- Filter ---
    @reactive.Calc
    def df_filtered():
        df = store.get()["df"].copy()
        if df.empty:
            return pd.DataFrame()
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2: y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0) >= y1) & (df["year"].fillna(0) <= y2)]
        df = df[df["numVotes"].fillna(0) >= mv]
        return df

    # --- Titel ---
    @output
    @render.ui
    def page_title():
        return ui.tags.h3("Überblick")

    # --- Body ---
    @output
    @render.ui
    def page_body():
        df = df_filtered()
        if df.empty:
            return ui.div(
                ui.tags.div("Keine Daten geladen oder Filter zu eng.", class_="muted", style="margin-bottom:8px;"),
                ui.tags.progress(max="100", value="10"),
            )
        return ui.div(
            kpi_ui(df),
            ui.output_plot("p_avg"),
            ui.output_plot("p_scatter"),
            ui.output_plot("p_dist"),
            ui.output_data_frame("tbl")
        )

    # --- KPIs ---
    def kpi_ui(df):
        imdb_mean = (df["averageRating"].dropna().mean() * 10) if "averageRating" in df.columns else np.nan
        rt_mean = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns else np.nan
        return ui.layout_column_wrap(
            ui.value_box(title="Filme (gefiltert)", value=f"{len(df)}"),
            ui.value_box(title="Ø IMDb (x10)", value=("—" if pd.isna(imdb_mean) else f"{imdb_mean:.1f}")),
            ui.value_box(title="Ø RT Tomatometer", value=("—" if pd.isna(rt_mean) else f"{rt_mean:.1f}")),
            fill=False
        )

    # --- Plots ---
    @output
    @render.plot
    def p_avg():
        df = df_filtered(); vals = {}
        if "averageRating" in df: vals["IMDb (x10)"] = df["averageRating"].dropna().mean() * 10
        if "rt_tomato" in df: vals["RT Tomatometer"] = df["rt_tomato"].dropna().mean()
        if "rt_audience" in df and input.use_audience(): vals["RT Audience"] = df["rt_audience"].dropna().mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        if not vals: ax.axis("off"); ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center"); return fig
        ax.bar(list(vals.keys()), list(vals.values())); ax.set_ylim(0, 100)
        ax.set_title("Durchschnittliche Bewertung (0–100)"); ax.set_ylabel("Score")
        return fig

    @output
    @render.plot
    def p_scatter():
        df = df_filtered()
        fig, ax = plt.subplots(figsize=(6, 6))
        if "rt_tomato" not in df: ax.axis("off"); ax.text(0.5, 0.5, "RT nicht verfügbar", ha="center", va="center"); return fig
        tmp = df.dropna(subset=["averageRating", "rt_tomato"])
        if tmp.empty: ax.axis("off"); ax.text(0.5, 0.5, "Keine Schnittmenge", ha="center", va="center"); return fig
        ax.scatter(tmp["averageRating"] * 10, tmp["rt_tomato"], alpha=.5)
        ax.set_xlabel("IMDb (x10)"); ax.set_ylabel("RT Tomatometer (%)"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        x, y = (tmp["averageRating"] * 10).values, tmp["rt_tomato"].values
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 100); ax.plot(xs, m * xs + b)
        ax.set_title("IMDb vs Rotten Tomatoes")
        return fig

    @output
    @render.plot
    def p_dist():
        df = df_filtered(); fig, ax = plt.subplots(figsize=(9, 4))
        tmp = df.dropna(subset=["averageRating"]) if not df.empty else df
        if tmp is None or tmp.empty:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center"); return fig
        ax.hist((tmp["averageRating"] * 10).dropna(), bins=30, alpha=.5, density=True, label="IMDb")
        if "rt_tomato" in tmp.columns:
            ax.hist(tmp["rt_tomato"].dropna(), bins=30, alpha=.5, density=True, label="RT Tomatometer")
        if "rt_audience" in tmp.columns and input.use_audience():
            ax.hist(tmp["rt_audience"].dropna(), bins=30, alpha=.3, density=True, label="RT Audience")
        ax.legend(); ax.set_xlim(0, 100); ax.set_title("Bewertungsverteilung"); ax.set_xlabel("Score (0–100)")
        return fig

    # --- Tabelle ---
    @output
    @render.data_frame
    def tbl():
        df = df_filtered().copy()
        if df.empty:
            return pd.DataFrame(columns=["Titel", "Jahr", "IMDb (x10)", "RT Tomatometer", "RT Audience", "Stimmen", "genres", "tconst"])
        df["IMDb (x10)"] = (df["averageRating"] * 10).round(1)
        df["RT Tomatometer"] = df.get("rt_tomato", pd.Series(index=df.index)).round(1)
        df["RT Audience"] = df.get("rt_audience", pd.Series(index=df.index)).round(1)
        df = df.rename(columns={"title": "Titel", "year": "Jahr", "numVotes": "Stimmen"})
        cols = ["Titel", "Jahr", "IMDb (x10)", "RT Tomatometer", "RT Audience", "Stimmen", "genres", "tconst"]
        for c in cols:
            if c not in df.columns: df[c] = pd.NA
        return df[cols].sort_values("Stimmen", ascending=False)

app = App(app_ui, server)
