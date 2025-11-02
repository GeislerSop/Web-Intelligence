from __future__ import annotations
import io, re, os, sys, logging
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# Logging für Posit-Logs
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("app")

# Runtime-Erkennung + optionales Pyodide-Patching (unschädlich auf Server)
IN_BROWSER = ("__pyodide__" in sys.modules) or os.environ.get("PYODIDE") == "1"
try:
    import pyodide_http  # type: ignore
    if IN_BROWSER:
        pyodide_http.patch_all()
except Exception:
    pass

# ---- Pfade robust bestimmen
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    BASE_DIR / "outputs",
    Path.cwd() / "outputs",          # falls Working-Dir differiert
    BASE_DIR,                         # fallback (selbes Verzeichnis)
]

FILENAMES = {
    "joined": "joined_imdb_rt.csv",
    "top20":  "top20_by_votes_imdb.csv",
    "gtr":    "google_trends_top5.csv",
}

def find_file(name_key: str) -> Path:
    fname = FILENAMES[name_key]
    for root in CANDIDATES:
        p = root / fname
        if p.exists():
            LOG.info(f"Found {name_key} at: {p}")
            return p
    # Letzter Versuch: relative Zeichenkette für Browser-Load
    p = (Path("outputs") / fname)
    LOG.warning(f"{name_key} not found in {CANDIDATES}; using relative path {p}")
    return p

CSV_JOINED = find_file("joined")
CSV_TOP20  = find_file("top20")
CSV_GT     = find_file("gtr")

def read_csv_portable(path: Path | str) -> pd.DataFrame:
    """Shinylive: HTTP-fetch; Server: direkt von Platte."""
    try:
        if IN_BROWSER:
            url = path.as_posix() if isinstance(path, Path) else str(path)
            r = requests.get(url, timeout=45); r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
        else:
            p = str(path) if isinstance(path, str) else str(path)
            # CSVs sind idR. UTF-8; wenn nötig, hier encoding="utf-8-sig" setzen
            df = pd.read_csv(p)
        LOG.info(f"Loaded CSV {path} → shape={df.shape}")
        return df
    except Exception as e:
        LOG.exception(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()


# ---------------- Config ----------------
SAMPLE_MAX = 80_000  # weiches Limit für Interaktivität

# ---------------- Helpers ----------------
def norm_title(t: str) -> str:
    if pd.isna(t):
        return ""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(t).lower())
    return " ".join(t.split())

# ---------------- UI ----------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        # Styling + Sidebar-Scroll-Fix (immer öffnbar, eigener Scrollbereich)
        ui.tags.style(
            """
            :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3;--brand:#2563eb;--brand-weak:#e8efff}
            html,body{height:100%;background:var(--bg);color:#0f172a}
            .sidebar{
              background:var(--card);border-right:1px solid var(--line);
              height:100vh;max-height:100vh;overflow:auto;position:sticky;top:0
            }
            .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 18px rgba(16,24,40,.06)}
            .btn{background:var(--brand);color:#fff;border-radius:10px;border:none}
            .muted{color:var(--muted);font-size:12px}
            .side-nav .shiny-input-radiogroup>div>label{
              display:block;padding:9px 12px;margin:4px 0;border-radius:10px;border:1px solid var(--line);cursor:pointer
            }
            .side-nav input[type=radio]{display:none}
            .side-nav .shiny-input-radiogroup>div>label:hover{background:#fafcff}
            .side-nav .shiny-input-radiogroup>div>input:checked+label{background:var(--brand-weak);border-color:#c7d7ff}
            @media (max-width: 991px){ .sidebar{position:fixed; width:78%;} }
            """
        ),
        ui.tags.h2("Movie Ratings", style="margin:6px 0 18px 0;"),
        ui.tags.div("Quelle: gebündelte CSVs im Projekt (./outputs)", class_="muted"),
        ui.tags.hr(),
        ui.tags.div("Navigation", class_="muted", style="margin-bottom:6px;"),
        ui.div(
            ui.input_radio_buttons(
                "page",
                None,
                choices={
                    "overview": "Überblick",
                    "compare":  "IMDb ↔ RT",
                    "trends":   "Genres/Jahrzehnte",
                    "top20":    "Top 20 (IMDb)",
                    "gtrends":  "Google Trends",
                    "table":    "Tabelle",
                },
                selected="overview",
                inline=False,
            ),
            class_="side-nav",
        ),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted", style="margin-bottom:6px;"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end",   "Jahr bis",  2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes",  "Min. IMDb-Stimmen", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "Audience-Score zusätzlich", False),
    ),
    ui.layout_column_wrap(
        ui.card(
            ui.card_header(ui.tags.div(id="page_title")),
            ui.output_ui("page_body"),
        ),
        fill=False,
    ),
    title="Movie Ratings",
    open={"desktop": "open", "mobile": "closed"},
)

# ---------------- Server ----------------
def server(input: Inputs, output: Outputs, session: Session):
    store = reactive.Value({
        "ready": False,
        "joined": pd.DataFrame(),  # Hauptdaten
        "top20":  pd.DataFrame(),  # Top-20 IMDb
        "gtr":    pd.DataFrame(),  # Google Trends
        "error":  "",
        "src":    "./outputs",
    })
    _boot_done = reactive.Value(False)

    def set_error(msg: str):
        s = store.get().copy()
        s.update({"ready": False, "error": msg})
        store.set(s)
        ui.notification_show(msg, type="warning", duration=6)

    # Autoload aller CSVs einmal beim Start
    @reactive.effect
    def _auto_load_once():
        if _boot_done.get():
            return
        _boot_done.set(True)
        ui.notification_show("Lade CSVs …", duration=None, id="note")
        try:
            joined = read_csv_portable(CSV_JOINED)
            top20  = read_csv_portable(CSV_TOP20)
            gtr    = read_csv_portable(CSV_GT)

            need = {"title", "year", "averageRating", "numVotes"}
            if joined.empty or not need.issubset(set(joined.columns)):
                raise RuntimeError("joined_imdb_rt.csv fehlt oder Mindestspalten fehlen")

            if "title_norm" not in joined.columns:
                joined["title_norm"] = joined["title"].map(norm_title)
            if len(joined) > SAMPLE_MAX:
                joined = joined.sample(SAMPLE_MAX, random_state=42)

            store.set({
                "ready": True,
                "joined": joined,
                "top20":  top20 if not top20.empty else pd.DataFrame(),
                "gtr":    gtr if not gtr.empty else pd.DataFrame(),
                "error":  "",
                "src":    "./outputs",
            })
            ui.notification_show("CSVs geladen.", type="message", duration=3)
        except Exception as e:
            set_error(f"CSV-Fehler: {e}")
        finally:
            ui.notification_remove("note")

    # Filter auf Haupttabelle
    @reactive.Calc
    def df_filtered():
        s = store.get()
        if not s["ready"]:
            return pd.DataFrame()
        df = s["joined"].copy()
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2:
            y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0) >= y1) & (df["year"].fillna(0) <= y2)]
        df = df[df["numVotes"].fillna(0) >= mv]
        return df

    # Titel
    @output
    @render.ui
    def page_title():
        s = store.get()
        mapping = {
            "overview": "Überblick",
            "compare":  "IMDb ↔ Rotten Tomatoes",
            "trends":   "Genres & Jahrzehnte",
            "top20":    "Top 20 (IMDb)",
            "gtrends":  "Google Trends",
            "table":    "Tabelle",
        }
        title = mapping.get(input.page(), "Überblick")
        return ui.tags.h3(f"{title} — Quelle: {s['src']}")

    # Seiteninhalt
    @output
    @render.ui
    def page_body():
        s = store.get()
        if not s["ready"]:
            return ui.div(
                ui.tags.div("Noch keine Daten. CSVs werden automatisch geladen …",
                            class_="muted", style="margin-bottom:8px;"),
                ui.tags.progress(max="100", value="25"),
            )
        warn = ui.tags.div(f"Hinweis: {s['error']}", class_="muted",
                           style="margin-bottom:10px;") if s["error"] else ui.tags.div()
        p = input.page()
        if p == "overview":
            return ui.div(warn, kpi_ui(df_filtered()), ui.output_plot("p_avg"))
        if p == "compare":
            return ui.div(warn, ui.output_plot("p_scatter"), ui.output_plot("p_dist"))
        if p == "trends":
            return ui.div(warn, ui.output_plot("p_genre"), ui.output_plot("p_decade"))
        if p == "top20":
            return ui.div(warn, ui.output_plot("p_top20"), ui.output_data_frame("tbl_top20"))
        if p == "gtrends":
            return ui.div(warn, ui.output_plot("p_gtrend"), ui.output_data_frame("tbl_gtrend"))
        if p == "table":
            return ui.div(warn, ui.output_data_frame("tbl"))
        return ui.div("—")

    # KPIs
    def kpi_ui(df: pd.DataFrame):
        imdb_mean = (df["averageRating"].dropna().mean() * 10) if "averageRating" in df.columns and df["averageRating"].notna().any() else np.nan
        rt_mean   = df["rt_tomato"].dropna().mean() if "rt_tomato" in df.columns and df["rt_tomato"].notna().any() else np.nan
        return ui.layout_column_wrap(
            ui.value_box(title="Filme (gefiltert)", value=f"{len(df)}"),
            ui.value_box(title="Ø IMDb (x10)", value=("—" if pd.isna(imdb_mean) else f"{imdb_mean:.1f}")),
            ui.value_box(title="Ø RT Tomatometer", value=("—" if pd.isna(rt_mean) else f"{rt_mean:.1f}")),
            fill=False,
        )

    # —— Plots (Hauptdaten) ——
    @output
    @render.plot
    def p_avg():
        df = df_filtered(); vals = {}
        if "averageRating" in df:
            vals["IMDb (x10)"] = df["averageRating"].dropna().mean() * 10
        if "rt_tomato" in df:
            vals["RT Tomatometer"] = df["rt_tomato"].dropna().mean()
        if "rt_audience" in df and input.use_audience():
            vals["RT Audience"] = df["rt_audience"].dropna().mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        if not vals:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center"); return fig
        ax.bar(list(vals.keys()), list(vals.values())); ax.set_ylim(0, 100)
        ax.set_title("Durchschnittliche Bewertung (0–100)"); ax.set_ylabel("Score")
        return fig

    @output
    @render.plot
    def p_scatter():
        df = df_filtered(); fig, ax = plt.subplots(figsize=(6, 6))
        if "rt_tomato" not in df:
            ax.axis("off"); ax.text(0.5, 0.5, "RT nicht verfügbar", ha="center", va="center"); return fig
        tmp = df.dropna(subset=["averageRating", "rt_tomato"])
        if tmp.empty:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Schnittmenge", ha="center", va="center"); return fig
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

    @output
    @render.plot
    def p_genre():
        s = store.get(); df = s["joined"] if s["ready"] else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(9, 4))
        imdb_g = df[df.get("numVotes", pd.Series()).fillna(0) >= 50000].dropna(subset=["genres"]) if not df.empty else pd.DataFrame()
        if imdb_g.empty:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Genre-Daten", ha="center", va="center"); return fig
        g = imdb_g.assign(genre=imdb_g["genres"].astype(str).str.split(",")).explode("genre")
        avg = (g.groupby("genre")["averageRating"].mean().sort_values(ascending=False).head(12)) * 10
        ax.bar(avg.index.tolist(), avg.values); ax.set_xticklabels(avg.index.tolist(), rotation=45, ha="right")
        ax.set_ylabel("Ø (0–100)"); ax.set_ylim(0, 100); ax.set_title("IMDb (≥50k) — Ø Genre (Top 12)")
        return fig

    @output
    @render.plot
    def p_decade():
        s = store.get(); df = s["joined"] if s["ready"] else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(9, 4))
        if df.empty:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Jahresdaten", ha="center", va="center"); return fig
        dec = df.dropna(subset=["year"]).copy(); dec["year"] = dec["year"].astype(int); dec["decade"] = (dec["year"] // 10) * 10
        avg = (dec.groupby("decade")["averageRating"].mean().sort_index()) * 10
        ax.plot(avg.index.values, avg.values, marker="o"); ax.set_ylim(0, 100)
        ax.set_title("IMDb — Ø Bewertung nach Jahrzehnt"); ax.set_xlabel("Jahrzehnt"); ax.set_ylabel("Ø (0–100)")
        return fig

    # —— Top-20 Seite ——
    @output
    @render.plot
    def p_top20():
        s = store.get(); df = s["top20"] if s["ready"] else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(9, 5))
        if df.empty:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Top-20-Daten", ha="center", va="center"); return fig
        d = df.copy().sort_values("numVotes", ascending=True).tail(20)  # 20 größte
        labels = d["title"].astype(str) + " (" + d["year"].astype(str) + ")"
        ax.barh(labels, d["numVotes"].values)
        ax.set_xlabel("Stimmen (IMDb)"); ax.set_title("Top 20 nach Stimmen (IMDb)")
        return fig

    @output
    @render.data_frame
    def tbl_top20():
        s = store.get(); df = s["top20"] if s["ready"] else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=["tconst","title","year","averageRating","numVotes"])
        d = df.copy().sort_values("numVotes", ascending=False).head(20)
        d["IMDb (x10)"] = (d.get("averageRating", pd.Series(index=d.index)) * 10).round(1)
        d = d.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols = ["Titel","Jahr","IMDb (x10)","Stimmen","tconst"]
        for c in cols:
            if c not in d.columns: d[c] = pd.NA
        return d[cols]

    # —— Google Trends Seite ——
    @output
    @render.plot
    def p_gtrend():
        s = store.get(); d = s["gtr"] if s["ready"] else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(9, 4))
        if d.empty:
            ax.axis("off"); ax.text(0.5, 0.5, "Keine Trends-Daten", ha="center", va="center"); return fig
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
    def tbl_gtrend():
        s = store.get(); d = s["gtr"] if s["ready"] else pd.DataFrame()
        return d if not d.empty else pd.DataFrame(columns=["date","kw1","kw2","kw3","kw4","kw5"])

    # —— Tabelle (Hauptdaten) ——
    @output
    @render.data_frame
    def tbl():
        df = df_filtered().copy()
        if df.empty:
            return pd.DataFrame(columns=["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"])
        df["IMDb (x10)"] = (df["averageRating"] * 10).round(1)
        df["RT Tomatometer"] = df.get("rt_tomato", pd.Series(index=df.index)).round(1)
        df["RT Audience"]    = df.get("rt_audience", pd.Series(index=df.index)).round(1)
        df = df.rename(columns={"title":"Titel","year":"Jahr","numVotes":"Stimmen"})
        cols = ["Titel","Jahr","IMDb (x10)","RT Tomatometer","RT Audience","Stimmen","genres","tconst"]
        for c in cols:
            if c not in df.columns: df[c] = pd.NA
        return df[cols].sort_values("Stimmen", ascending=False)

app = App(app_ui, server)
