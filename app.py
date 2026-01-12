# app.py — Movie Ratings Dashboard (+ RT-Text, Graph & OpenAI-Analyse)
from __future__ import annotations

import os, re, json, logging, warnings, textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# optional: nicer graph plot (wenn installiert)
try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    nx = None
    HAS_NX = False

# optional: OpenAI API Call via requests
try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except Exception:
    requests = None
    HAS_REQUESTS = False

# Warnings leiser
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("movie-app")

# ---------------- Pfade/Loader ----------------
BASE = Path(__file__).resolve().parent


def _find(*candidates: Path) -> Path | None:
    for p in candidates:
        if p and p.exists():
            return p
    return None


CSV_JOINED = _find(BASE / "outputs/joined_imdb_rt.csv", BASE / "joined_imdb_rt.csv")
CSV_RT_METR = _find(BASE / "outputs/rt_metrics.csv")
CSV_RT_RAW = _find(BASE / "outputs/rotten_tomatoes_movies.csv", BASE / "raw/rotten_tomatoes_movies.csv")
CSV_TOP20 = _find(BASE / "outputs/top20_by_votes_imdb.csv", BASE / "top20_by_votes_imdb.csv")
CSV_GTR = _find(BASE / "outputs/google_trends_top5.csv", BASE / "google_trends_top5.csv")


def _read_csv(p: Path | None) -> pd.DataFrame:
    if p is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        LOG.info(f"Loaded {p} shape={df.shape}")
        return df
    except Exception as e:
        LOG.exception(f"CSV read failed: {p} -> {e}")
        return pd.DataFrame()


joined_raw = _read_csv(CSV_JOINED)
rt_metrics = _read_csv(CSV_RT_METR)
rt_raw = _read_csv(CSV_RT_RAW)
top20_raw = _read_csv(CSV_TOP20)
gtr_raw = _read_csv(CSV_GTR)

# ---------------- Helpers ----------------
def norm_title(t: str) -> str:
    if pd.isna(t):
        return ""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(t).lower())
    return " ".join(t.split())


def to100(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "")
    try:
        v = float(s)
        return v * 10 if 0 <= v <= 10 else v
    except Exception:
        return np.nan


def split_list(x) -> list[str]:
    """Komma-separierte Listen robust splitten (Genres/Actors/Directors)."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    # kleine Normalisierung (Doppelleerzeichen etc.)
    parts = [" ".join(p.split()) for p in parts]
    return parts


# ---------------- RT Scores (wie vorher) ----------------
def std_rt_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: RT-Rohdatei in Standardform bringen."""
    if df.empty:
        return pd.DataFrame(columns=["title_norm", "year", "rt_tomato", "rt_audience"])
    L = {c.lower(): c for c in df.columns}

    def pick(*xs):
        for x in xs:
            if x in L:
                return L[x]

    c_title = pick("movie_title", "title", "name")
    c_year = pick("original_release_year", "year", "release_year")
    c_info = pick("movie_info")
    c_tom = pick("tomatometer_rating", "tomato_score", "tomatometer", "rt", "rt_score")
    c_aud = pick("audience_rating", "audience_score", "audiencescore")
    keep = [c for c in [c_title, c_year, c_info, c_tom, c_aud] if c]
    d = df[keep].copy() if keep else df.iloc[0:0].copy()
    if d.empty:
        return pd.DataFrame(columns=["title_norm", "year", "rt_tomato", "rt_audience"])

    ren = {}
    if c_title:
        ren[c_title] = "title"
    if c_year:
        ren[c_year] = "year"
    if c_info:
        ren[c_info] = "info"
    if c_tom:
        ren[c_tom] = "rt_tomato_raw"
    if c_aud:
        ren[c_aud] = "rt_audience_raw"
    d.rename(columns=ren, inplace=True)

    d["title_norm"] = d.get("title", "").map(norm_title)
    d["year"] = pd.to_numeric(d.get("year", pd.Series(index=d.index)), errors="coerce")
    if "info" in d and d["year"].isna().all():
        d["year"] = (
            d["info"]
            .astype(str)
            .str.extract(r"\b(19|20)\d{2}\b", expand=False)
            .astype(float)
        )

    # RT bis 2020 (für Score-Join – Text/Graph darf trotzdem alle Jahre haben)
    d = d[d["year"].fillna(0) <= 2020]

    d["rt_tomato"] = d.get("rt_tomato_raw", pd.Series(index=d.index)).map(to100)
    d["rt_audience"] = d.get("rt_audience_raw", pd.Series(index=d.index)).map(to100)
    return (
        d[["title_norm", "year", "rt_tomato", "rt_audience"]]
        .dropna(subset=["title_norm"])
        .drop_duplicates()
    )


def std_rt_from_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Bevorzugt: aus rt_metrics.csv lesen und vereinheitlichen."""
    need = {"title", "title_norm", "year", "rt_tomato", "rt_audience"}
    if df.empty or not need.issubset(set(df.columns)):
        return pd.DataFrame()
    d = df.copy()
    d["title_norm"] = d["title_norm"].astype(str).map(norm_title)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[d["year"].fillna(0) <= 2020]  # RT bis 2020
    d["rt_tomato"] = pd.to_numeric(d["rt_tomato"], errors="coerce").clip(0, 100)
    d["rt_audience"] = pd.to_numeric(d["rt_audience"], errors="coerce").clip(0, 100)
    return (
        d[["title_norm", "year", "rt_tomato", "rt_audience"]]
        .dropna(subset=["title_norm"])
        .drop_duplicates()
    )


# --- IMDb vorbereiten & RT-Quelle bestimmen
if not joined_raw.empty and "title_norm" not in joined_raw.columns and "title" in joined_raw.columns:
    joined_raw["title_norm"] = joined_raw["title"].map(norm_title)

rt_std = std_rt_from_metrics(rt_metrics)
RT_SOURCE = "rt_metrics.csv"
if rt_std.empty:
    rt_std = std_rt_from_raw(rt_raw)
    RT_SOURCE = "rotten_tomatoes_movies.csv (raw)"


def joined_for_visuals() -> pd.DataFrame:
    """IMDb + RT zusammenführen; UI-Spaltennamen setzen."""
    if joined_raw.empty:
        return joined_raw
    df = joined_raw.copy()

    # einheitliche Namen
    if "year" not in df.columns and "Year" in df.columns:
        df.rename(columns={"Year": "year"}, inplace=True)
    if "numVotes" not in df.columns and "votes" in df.columns:
        df.rename(columns={"votes": "numVotes"}, inplace=True)

    # merge RT (nur bis 2020, das macht rt_std bereits)
    cols = ["title_norm", "year"]
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if not rt_std.empty and all(c in df.columns for c in cols):
        df = df.merge(rt_std, on=cols, how="left", suffixes=("", "_rtstd"))
        df["rt_tomato"] = df.get("rt_tomato", pd.Series(index=df.index)).fillna(df.get("rt_tomato_rtstd"))
        df["rt_audience"] = df.get("rt_audience", pd.Series(index=df.index)).fillna(df.get("rt_audience_rtstd"))

    # UI-Spalten
    if "averageRating" in df:
        df["IMDb_Score_100"] = (df["averageRating"] * 10).clip(0, 100)
    if "rt_tomato" in df:
        df["RT_Tomatometer"] = df["rt_tomato"].clip(0, 100)
    if "rt_audience" in df:
        df["RT_Audience"] = df["rt_audience"].clip(0, 100)
    if "numVotes" in df:
        df["IMDb_Votes"] = pd.to_numeric(df["numVotes"], errors="coerce").astype("Int64")
    return df


# ---------------- RT Textdaten (statt Wikidata) ----------------
def std_rt_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Textquelle + Metadaten aus rotten_tomatoes_movies.csv:
    - movie_info (Plot/Beschreibung)
    - critics_consensus (Kurzfazit)
    - genres, actors, directors
    """
    if df.empty:
        return pd.DataFrame(columns=["title", "title_norm", "year", "movie_info", "critics_consensus", "genres", "actors"])

    L = {c.lower(): c for c in df.columns}

    def pick(*xs):
        for x in xs:
            if x in L:
                return L[x]
        return None

    c_title = pick("movie_title", "title", "name")
    c_year = pick("original_release_year", "year", "release_year")
    c_info = pick("movie_info", "info", "plot", "description", "synopsis")
    c_cons = pick("critics_consensus", "consensus", "summary")
    c_gen = pick("genres", "genre")
    c_act = pick("actors", "cast")

    keep = [c for c in [c_title, c_year, c_info, c_cons, c_gen, c_act] if c]
    d = df[keep].copy() if keep else df.copy()

    ren = {}
    if c_title:
        ren[c_title] = "title"
    if c_year:
        ren[c_year] = "year"
    if c_info:
        ren[c_info] = "movie_info"
    if c_cons:
        ren[c_cons] = "critics_consensus"
    if c_gen:
        ren[c_gen] = "genres"
    if c_act:
        ren[c_act] = "actors"
    d.rename(columns=ren, inplace=True)

    for col in ["title", "movie_info", "critics_consensus", "genres", "actors"]:
        if col not in d.columns:
            d[col] = ""

    d["title"] = d["title"].astype(str)
    d["title_norm"] = d["title"].map(norm_title)

    d["year"] = pd.to_numeric(d.get("year", pd.Series(index=d.index)), errors="coerce")

    d["movie_info"] = d["movie_info"].astype(str).fillna("").replace({"nan": ""})
    d["critics_consensus"] = d["critics_consensus"].astype(str).fillna("").replace({"nan": ""})
    d["genres"] = d["genres"].astype(str).fillna("").replace({"nan": ""})
    d["actors"] = d["actors"].astype(str).fillna("").replace({"nan": ""})

    d = d.dropna(subset=["title_norm"]).drop_duplicates(subset=["title_norm", "year"])
    return d[["title", "title_norm", "year", "movie_info", "critics_consensus", "genres", "actors"]]


rt_text = std_rt_text(rt_raw)


# ---------------- Graph aus RT bauen (Movie–Genre, Movie–Actor, Actor–Genre) ----------------
def build_edges_from_rt(rt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt typed edges:
    source,target,relation,source_type,target_type,source_norm,target_norm
    """
    cols = ["source", "target", "relation", "source_type", "target_type", "source_norm", "target_norm"]
    if rt_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, r in rt_df.iterrows():
        title = str(r.get("title", "")).strip()
        tn = str(r.get("title_norm", "")).strip()
        if not title or not tn:
            continue

        genres = split_list(r.get("genres", ""))
        actors = split_list(r.get("actors", ""))

        # Movie -> Genre
        for g in genres:
            gn = norm_title(g)
            if not gn:
                continue
            rows.append((title, g, "has_genre", "movie", "genre", tn, gn))

        # Movie -> Actor
        for a in actors:
            an = norm_title(a)
            if not an:
                continue
            rows.append((title, a, "has_actor", "movie", "actor", tn, an))

        # Actor -> Genre (aus demselben Film abgeleitet)
        for a in actors:
            an = norm_title(a)
            if not an:
                continue
            for g in genres:
                gn = norm_title(g)
                if not gn:
                    continue
                rows.append((a, g, "actor_in_genre", "actor", "genre", an, gn))

    e = pd.DataFrame(rows, columns=cols)
    e = e.drop_duplicates()
    return e


edges = build_edges_from_rt(rt_text)

# ---------------- OpenAI: Textanalyse Funktion ----------------
def openai_analyze_text(text: str) -> dict:
    """
    Nutzt Responses API (empfohlen) über HTTP.
    Erwartet OPENAI_API_KEY als Environment Variable.
    Gibt dict mit summary/sentiment/topics zurück.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return {"error": "OPENAI_API_KEY ist nicht gesetzt (Environment Variable fehlt)."}
    if not HAS_REQUESTS:
        return {"error": "Python-Paket 'requests' ist nicht verfügbar."}
    if not text.strip():
        return {"error": "Kein Text vorhanden."}

    # kurze, stabile Ausgabe als JSON erzwingen
    prompt = (
        "Analysiere den folgenden Filmtext (Beschreibung oder Critics Consensus) und gib NUR JSON zurück "
        "mit den Keys: summary (max 2 Sätze), sentiment (positive|neutral|negative), topics (Liste mit 5 Stichworten).\n\n"
        f"TEXT:\n{text.strip()}"
    )

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.2",
                "input": prompt,
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            return {"error": f"OpenAI API Error {resp.status_code}: {resp.text[:400]}"}

        data = resp.json()

        # Responses API: convenience: output_text (wenn vorhanden), sonst output[] zusammensetzen
        out_text = data.get("output_text", "")
        if not out_text:
            # fallback: best-effort
            try:
                chunks = []
                for item in data.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            chunks.append(c.get("text", ""))
                out_text = "\n".join(chunks).strip()
            except Exception:
                out_text = ""

        out_text = out_text.strip()
        if not out_text:
            return {"error": "Keine Ausgabe von der API erhalten."}

        # JSON parsen (falls Modell trotz Bitte noch Text drumherum packt)
        m = re.search(r"\{.*\}", out_text, flags=re.S)
        candidate = m.group(0) if m else out_text
        try:
            obj = json.loads(candidate)
            # normalize
            return {
                "summary": str(obj.get("summary", "")).strip(),
                "sentiment": str(obj.get("sentiment", "")).strip(),
                "topics": obj.get("topics", []),
            }
        except Exception:
            return {"error": "Antwort war kein gültiges JSON.", "raw": out_text[:600]}

    except Exception as e:
        return {"error": f"Request fehlgeschlagen: {e}"}


# ---------------- UI ----------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style(
            """
        :root{--bg:#f7f8fb;--card:#ffffff;--muted:#6b7280;--line:#e8ebf3}
        html,body{height:100%;background:var(--bg);color:#0f172a}
        .sidebar{background:var(--card);border-right:1px solid var(--line);
                 height:100vh;max-height:100vh;overflow:auto;position:sticky;top:0}
        .muted{color:var(--muted);font-size:12px}
        """
        ),
        ui.tags.h2("🎬 Movie Ratings"),
        ui.tags.div("Navigation", class_="muted"),
        ui.input_radio_buttons(
            "page",
            None,
            choices={
                "overview": "Übersicht",
                "compare": "IMDb ↔ RT",
                "coverage": "Abdeckung RT",
                "trends": "Trends",
                "top20": "Top 20",
                "gtrends": "Google Trends",
                "rt_only": "RT (nur RT-Daten)",
                "rt_text_ai": "RT Text + KI",
                "graph": "Graph (RT: Movie/Actor/Genre)",
                "table": "Tabelle",
            },
            selected="overview",
            inline=False,
        ),
        ui.tags.hr(),
        ui.tags.div("Filter", class_="muted"),
        ui.input_numeric("year_start", "Jahr von", 1980, min=1920, max=2025, step=1),
        ui.input_numeric("year_end", "Jahr bis", 2025, min=1920, max=2025, step=1),
        ui.input_numeric("min_votes", "Mind. IMDb-Stimmen (Filter)", value=50000, min=0, step=1000),
        ui.input_checkbox("use_audience", "RT Audience mit anzeigen", True),
        ui.tags.hr(),
        ui.tags.div("RT Text + KI", class_="muted"),
        ui.input_selectize("rt_title", "Film auswählen", choices=[], selected=None, multiple=False),
        ui.input_radio_buttons(
            "rt_text_source",
            "Textquelle",
            choices={"movie_info": "Beschreibung (movie_info)", "critics_consensus": "Critics Consensus"},
            selected="movie_info",
            inline=False,
        ),
        ui.input_action_button("run_ai", "KI-Analyse starten"),
        ui.tags.div(
            "Hinweis: OPENAI_API_KEY muss als Environment Variable gesetzt sein.",
            class_="muted",
            style="margin-top:6px;",
        ),
        ui.tags.hr(),
        ui.tags.div("Graph", class_="muted"),
        ui.input_selectize("graph_node", "Knoten auswählen", choices=[], selected=None, multiple=False),
        ui.input_numeric("graph_hops", "Hops", value=1, min=1, max=2, step=1),
        ui.input_numeric("graph_max_nodes", "Max. Knoten (Layout)", value=80, min=20, max=200, step=10),
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
    dropdowns_inited = reactive.Value(False)

    # Dropdowns (einmal initialisieren – ohne session.user_data)
    @reactive.Effect
    def _init_dropdowns_once():
        if dropdowns_inited.get():
            return

        # RT Titel Dropdown
        if rt_text.empty:
            ui.update_selectize("rt_title", choices={})
        else:
            tmp = rt_text.copy()
            tmp["label"] = tmp.apply(
                lambda r: f"{r['title']} ({int(r['year'])})" if pd.notna(r["year"]) else str(r["title"]),
                axis=1,
            )
            tmp = tmp.sort_values(["title"]).head(6000)
            choices = {r["title_norm"]: r["label"] for _, r in tmp.iterrows()}
            ui.update_selectize("rt_title", choices=choices)
            if input.rt_title() is None and choices:
                ui.update_selectize("rt_title", selected=next(iter(choices.keys())))

        # Graph Nodes Dropdown
        if edges.empty:
            ui.update_selectize("graph_node", choices={})
        else:
            nodes_src = edges[["source", "source_norm", "source_type"]].rename(
                columns={"source": "label", "source_norm": "norm", "source_type": "type"}
            )
            nodes_tgt = edges[["target", "target_norm", "target_type"]].rename(
                columns={"target": "label", "target_norm": "norm", "target_type": "type"}
            )
            nodes = pd.concat([nodes_src, nodes_tgt], ignore_index=True).drop_duplicates(subset=["norm", "type"])
            nodes = nodes.sort_values(["type", "label"]).head(9000)
            gchoices = {f"{r['type']}::{r['norm']}": f"{r['type']}: {r['label']}" for _, r in nodes.iterrows()}
            ui.update_selectize("graph_node", choices=gchoices)
            if input.graph_node() is None and gchoices:
                ui.update_selectize("graph_node", selected=next(iter(gchoices.keys())))

        dropdowns_inited.set(True)

    # Status / Quelle anzeigen
    @output
    @render.ui
    def status_files():
        def li(ok, text):
            return ui.tags.li((("✅ " if ok else "❌ ") + text))

        rows = [
            li(CSV_JOINED is not None, f"joined: {CSV_JOINED} (shape={tuple(joined_raw.shape)})"),
            li(CSV_RT_METR is not None, f"rt_metrics: {CSV_RT_METR} (shape={tuple(rt_metrics.shape)})"),
            li(CSV_RT_RAW is not None, f"rt_raw: {CSV_RT_RAW} (shape={tuple(rt_raw.shape)})"),
            li(CSV_TOP20 is not None, f"top20 : {CSV_TOP20}  (shape={tuple(top20_raw.shape)})"),
            li(CSV_GTR is not None, f"gtr   : {CSV_GTR}    (shape={tuple(gtr_raw.shape)})"),
            ui.tags.li(f"RT-Quelle aktiv (Scores): {RT_SOURCE}"),
            ui.tags.li(f"RT-Scores Datensätze (≤2020): {len(rt_std):,}".replace(",", ".")),
            ui.tags.li("—"),
            ui.tags.li(f"RT Textdatensätze: {len(rt_text):,}".replace(",", ".")),
            ui.tags.li(f"Edges (aus RT erzeugt): {len(edges):,}".replace(",", ".")),
            ui.tags.li(f"networkx verfügbar: {'ja' if HAS_NX else 'nein (Fallback)'}"),
            ui.tags.li(f"requests verfügbar: {'ja' if HAS_REQUESTS else 'nein (OpenAI Call deaktiviert)'}"),
        ]
        return ui.tags.small(ui.tags.ul(*rows, style="margin:0;padding-left:18px;"))

    # Daten-Sichten
    @reactive.Calc
    def df_joined():
        df = joined_for_visuals().copy()
        if df.empty:
            return df
        y1, y2 = int(input.year_start()), int(input.year_end())
        if y1 > y2:
            y1, y2 = y2, y1
        mv = int(input.min_votes())
        df = df[(df["year"].fillna(0) >= y1) & (df["year"].fillna(0) <= y2)]
        if "IMDb_Votes" in df:
            df = df[df["IMDb_Votes"].fillna(0) >= mv]
        return df

    @reactive.Calc
    def df_with_rt():
        d = df_joined()
        col = "RT_Tomatometer"
        return d[d[col].notna()] if not d.empty and col in d.columns else d.iloc[0:0]

    # ---------------- Titel ----------------
    @output
    @render.ui
    def page_title():
        mapping = {
            "overview": "Übersicht",
            "compare": "Vergleich IMDb ↔ Rotten Tomatoes",
            "coverage": "Abdeckung Rotten Tomatoes",
            "trends": "Trends (Genre & दशकहnt)",
            "top20": "Top 20 (IMDb-Stimmen)",
            "gtrends": "Google Trends",
            "rt_only": "Rotten Tomatoes — eigene Sicht",
            "rt_text_ai": "Rotten Tomatoes — Textdaten + KI",
            "graph": "Graph — Movie / Actor / Genre (aus RT gebaut)",
            "table": "Tabelle (gefiltert)",
        }
        return ui.tags.h3(mapping.get(input.page(), "Übersicht"))

    # ---------------- Routing ----------------
    @output
    @render.ui
    def page_body():
        p = input.page()
        if p == "overview":
            return ui.div(kpi_ui(), ui.output_plot("p_avg_bars"), ui.output_plot("p_vote_ccdf"))
        if p == "compare":
            return ui.div(ui.output_plot("p_scatter_simple"), ui.output_plot("p_mean_diff_by_bin"))
        if p == "coverage":
            return ui.div(ui.output_plot("p_coverage_share"), ui.output_plot("p_rt_pie"))
        if p == "trends":
            return ui.div(ui.output_plot("p_genre_avg"), ui.output_plot("p_decade_avg"))
        if p == "top20":
            return ui.div(ui.output_plot("p_top20"), ui.output_data_frame("tbl_top20"))
        if p == "gtrends":
            return ui.div(ui.output_plot("p_gtrends"), ui.output_data_frame("tbl_gtrends"))
        if p == "rt_only":
            return ui.div(ui.output_plot("p_rt_only_avg"))
        if p == "rt_text_ai":
            return ui.div(
                ui.output_ui("rt_text_block"),
                ui.layout_column_wrap(
                    ui.card(ui.card_header("Top-Wörter (einfach)"), ui.output_data_frame("tbl_rt_terms")),
                    ui.card(ui.card_header("KI-Ergebnis (OpenAI)"), ui.output_ui("ai_result_block")),
                    fill=False,
                ),
            )
        if p == "graph":
            return ui.div(
                ui.card(ui.card_header("Netzplan (Subgraph)"), ui.output_plot("p_graph_nice")),
                ui.card(ui.card_header("Edge-Liste (Subgraph)"), ui.output_data_frame("tbl_graph_edges")),
            )
        if p == "table":
            return ui.div(ui.output_data_frame("tbl_all"))
        return ui.div("—")

    # ---------------- KPIs ----------------
    def kpi_ui():
        d_all = df_joined()
        d_rt = df_with_rt()
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
        share = (len(d_rt) / len(d_all) * 100) if len(d_all) > 0 else 0
        cards.append(vb("RT-Abdeckung", f"{share:.1f}%"))
        return ui.layout_column_wrap(*cards, fill=False)

    # ---------------- Übersicht: Balken + CCDF ----------------
    @output
    @render.plot
    def p_avg_bars():
        d_all = df_joined()
        d_rt = df_with_rt()
        fig, ax = plt.subplots(figsize=(9, 3.8))
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
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
            return fig

        colors = ["#2563EB", "#F59E0B", "#10B981"][: len(values)]
        ax.bar(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Durchschnittliche Bewertungen (0–100)")
        ax.set_ylabel("Punkte")
        return fig

    @output
    @render.plot
    def p_vote_ccdf():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty or "IMDb_Votes" not in d:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Stimmen-Daten", ha="center", va="center")
            return fig
        x = np.log10(d["IMDb_Votes"].clip(lower=1)).dropna().to_numpy()
        if x.size == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
            return fig
        xs = np.sort(x)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, 1 - ys, color="#2563EB", linewidth=2)
        ax.set_xlabel("log10(Stimmen)  (3=1.000, 4=10.000, 5=100.000)")
        ax.set_ylabel("Anteil der Filme ≥ X")
        ax.set_title(f"Wie viele Stimmen haben die Filme? (N={x.size:,})".replace(",", "."))
        ax.set_ylim(0, 1)
        return fig

    # ---------------- Vergleich: Scatter + Ø-Differenz je Bin ----------------
    @output
    @render.plot
    def p_scatter_simple():
        d = df_with_rt().dropna(subset=["IMDb_Score_100", "RT_Tomatometer"])
        fig, ax = plt.subplots(figsize=(9, 5))
        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Schnittmenge (IMDb & RT)", ha="center", va="center")
            return fig
        ax.scatter(d["IMDb_Score_100"], d["RT_Tomatometer"], s=10, alpha=0.4, color="#4F46E5")
        ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.5, color="#F59E0B", alpha=0.9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("IMDb (0–100)")
        ax.set_ylabel("RT Tomatometer (0–100)")
        ax.set_title(f"IMDb vs. Rotten Tomatoes (N={len(d):,})".replace(",", "."))
        return fig

    @output
    @render.plot
    def p_mean_diff_by_bin():
        d = df_with_rt().dropna(subset=["IMDb_Score_100", "RT_Tomatometer"])
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
            return fig
        d = d.copy()
        d["bin"] = (d["IMDb_Score_100"] // 10 * 10).astype(int).clip(0, 90)
        s = d.groupby("bin").apply(lambda x: (x["RT_Tomatometer"] - x["IMDb_Score_100"]).mean())
        ax.plot(s.index, s.values, marker="o", color="#EF4444", linewidth=2)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xticks(list(range(0, 101, 10)))
        ax.set_xlabel("IMDb (gerundet, 0–100)")
        ax.set_ylabel("Ø [RT − IMDb] (Punkte)")
        ax.set_title("Wo weichen RT und IMDb ab? (Ø Differenz je IMDb-Bin)")
        return fig

    # ---------------- Abdeckung + RT-Kreisdiagramm ----------------
    @output
    @render.plot
    def p_coverage_share():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty or "year" not in d:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
            return fig
        tmp = d.dropna(subset=["year"]).copy()
        tmp["has_rt"] = tmp["RT_Tomatometer"].notna() if "RT_Tomatometer" in tmp.columns else False
        share = (tmp.groupby("year")["has_rt"].mean() * 100).sort_index()
        if share.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Jahresdaten", ha="center", va="center")
            return fig
        ax.plot(share.index, share.values, marker="o", color="#2563EB", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Jahr")
        ax.set_ylabel("RT-Abdeckung (%)")
        ax.set_title("Wie oft gibt es RT-Bewertungen? (Anteil pro Jahr)")
        return fig

    @output
    @render.plot
    def p_rt_pie():
        d = df_with_rt()
        fig, ax = plt.subplots(figsize=(6, 6))
        if d.empty or "RT_Tomatometer" not in d:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine RT-Daten", ha="center", va="center")
            return fig
        cats = pd.Series(np.where(d["RT_Tomatometer"] >= 60, "Fresh", "Rotten"))
        counts = cats.value_counts().reindex(["Fresh", "Rotten"]).fillna(0)
        if counts.sum() == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine RT-Werte", ha="center", va="center")
            return fig
        colors = ["#10B981", "#EF4444"]
        ax.pie(
            counts.values,
            labels=[f"{k} ({int(v)})" for k, v in counts.items()],
            autopct="%1.0f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 11},
        )
        ax.set_title("RT-Verteilung (Fresh ≥ 60)")
        ax.axis("equal")
        return fig

    # ---------------- Trends (IMDb) ----------------
    @output
    @render.plot
    def p_genre_avg():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty or "genres" not in d or "IMDb_Score_100" not in d:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Genre-Daten", ha="center", va="center")
            return fig
        g = (
            d[d.get("IMDb_Votes", pd.Series(dtype=float)).fillna(0) >= 50_000]
            .dropna(subset=["genres"])
            .assign(genre=lambda x: x["genres"].astype(str).str.split(","))
            .explode("genre")
        )
        if g.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Genre-Daten (≥50k)", ha="center", va="center")
            return fig
        s = g.groupby("genre").agg(avg=("IMDb_Score_100", "mean"), n=("IMDb_Score_100", "size"))
        s = s.sort_values("avg", ascending=False).head(12)
        ax.bar(s.index, s["avg"], color="#2563EB")
        for i, (v, n) in enumerate(zip(s["avg"], s["n"])):
            ax.text(i, v + 1, f"{v:.1f}\nN={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xticklabels(s.index, rotation=45, ha="right")
        ax.set_ylabel("Ø IMDb (0–100)")
        ax.set_title("Welche Genres schneiden gut ab? (Filme mit ≥50k Stimmen)")
        return fig

    @output
    @render.plot
    def p_decade_avg():
        d = df_joined()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty or "year" not in d or "IMDb_Score_100" not in d:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Jahresdaten", ha="center", va="center")
            return fig
        dec = d.dropna(subset=["year"]).copy()
        dec["decade"] = (dec["year"].astype(int) // 10) * 10
        s = dec.groupby("decade").agg(avg=("IMDb_Score_100", "mean"), n=("IMDb_Score_100", "size")).sort_index()
        ax.plot(s.index, s["avg"], marker="o", color="#F59E0B", linewidth=2)
        for x, y, n in zip(s.index, s["avg"], s["n"]):
            ax.text(x, y + 1, f"{y:.1f}\nN={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Jahrzehnt")
        ax.set_ylabel("Ø IMDb (0–100)")
        ax.set_title("Wie haben sich Ø-Bewertungen je Jahrzehnt entwickelt?")
        return fig

    # ---------------- Top 20 ----------------
    @output
    @render.plot
    def p_top20():
        d = top20_raw.copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        need = {"title", "year", "numVotes"}
        if d.empty or not need.issubset(set(d.columns)):
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Top-20-Daten", ha="center", va="center")
            return fig
        d = d.sort_values("numVotes", ascending=True).tail(20)
        labels = d["title"].astype(str) + " (" + d["year"].astype(int).astype(str) + ")"
        ax.barh(labels, d["numVotes"].values, color="#2563EB")
        ax.set_xlabel("Stimmen (IMDb)")
        ax.set_title("Top 20 nach IMDb-Stimmen")
        return fig

    @output
    @render.data_frame
    def tbl_top20():
        d = top20_raw.copy()
        if d.empty:
            return pd.DataFrame(columns=["Titel", "Jahr", "IMDb_Score_100", "IMDb_Votes", "tconst"])
        d = d.sort_values("numVotes", ascending=False).head(20)
        if "averageRating" in d:
            d["IMDb_Score_100"] = (d["averageRating"] * 10).round(1)
        d = d.rename(columns={"title": "Titel", "year": "Jahr", "numVotes": "IMDb_Votes"})
        cols = ["Titel", "Jahr", "IMDb_Score_100", "IMDb_Votes", "tconst"]
        for c in cols:
            if c not in d.columns:
                d[c] = pd.NA
        return d[cols]

    # ---------------- Google Trends ----------------
    @output
    @render.plot
    def p_gtrends():
        d = gtr_raw.copy()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Trends-Daten", ha="center", va="center")
            return fig
        if "date" in d.columns:
            d = d.set_index("date")
        else:
            d = d.set_index(d.columns[0])
        for c in d.columns:
            ax.plot(pd.to_datetime(d.index), d[c], label=c, linewidth=2)
        ax.legend(fontsize=8)
        ax.set_title("Google Trends (Top 5)")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Interesse (0–100)")
        return fig

    @output
    @render.data_frame
    def tbl_gtrends():
        d = gtr_raw.copy()
        return d if not d.empty else pd.DataFrame(columns=["date", "kw1", "kw2", "kw3", "kw4", "kw5"])

    # ---------------- Tabelle (gefiltert) ----------------
    @output
    @render.data_frame
    def tbl_all():
        d = df_joined().copy()
        if d.empty:
            return pd.DataFrame(
                columns=["Titel", "Jahr", "IMDb_Score_100", "RT_Tomatometer", "RT_Audience", "IMDb_Votes", "genres", "tconst"]
            )
        d = d.rename(columns={"title": "Titel", "year": "Jahr"})
        cols = ["Titel", "Jahr", "IMDb_Score_100", "RT_Tomatometer", "RT_Audience", "IMDb_Votes", "genres", "tconst"]
        for c in cols:
            if c not in d.columns:
                d[c] = pd.NA
        return d[cols].sort_values("IMDb_Votes", ascending=False)

    # ---------------- RT-only Seite (nur RT, bis 2020) ----------------
    @output
    @render.plot
    def p_rt_only_avg():
        d = rt_std.copy()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine RT-Daten", ha="center", va="center")
            return fig
        labels, values, colors = [], [], []
        if d["rt_tomato"].notna().any():
            labels.append(f"Tomatometer\nN={d['rt_tomato'].notna().sum():,}".replace(",", "."))
            values.append(d["rt_tomato"].mean())
            colors.append("#F59E0B")
        if input.use_audience() and d["rt_audience"].notna().any():
            labels.append(f"Audience\nN={d['rt_audience'].notna().sum():,}".replace(",", "."))
            values.append(d["rt_audience"].mean())
            colors.append("#10B981")
        if not values:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine RT-Werte", ha="center", va="center")
            return fig
        ax.bar(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Rotten Tomatoes (nur RT-Daten, bis 2020) — Ø Werte")
        ax.set_ylabel("Punkte")
        return fig

    # ---------------- RT Text + KI ----------------
    @reactive.Calc
    def rt_selected_row() -> pd.Series:
        key = input.rt_title()
        if rt_text.empty or key is None:
            return pd.Series({"title": "", "title_norm": "", "year": pd.NA, "movie_info": "", "critics_consensus": "", "genres": "", "actors": ""})
        hit = rt_text[rt_text["title_norm"] == str(key)]
        if hit.empty:
            return pd.Series({"title": "", "title_norm": str(key), "year": pd.NA, "movie_info": "", "critics_consensus": "", "genres": "", "actors": ""})
        # falls mehrere Jahre: nimm das erste (oder man könnte hier später per Jahr auswählen)
        return hit.iloc[0]

    @output
    @render.ui
    def rt_text_block():
        r = rt_selected_row()
        title = str(r.get("title", "")) or "—"
        year = r.get("year", pd.NA)
        ytxt = f"({int(year)})" if pd.notna(year) else ""
        src = input.rt_text_source()
        txt = str(r.get(src, "") or "")
        meta = []
        meta.append(f"Quelle: {src}")
        meta.append(f"Textlänge: {len(txt):,} Zeichen".replace(",", "."))
        meta.append(f"Genres: {', '.join(split_list(r.get('genres',''))[:8])}".strip())
        meta.append(f"Actors: {', '.join(split_list(r.get('actors',''))[:6])}".strip())
        return ui.card(
            ui.card_header(f"📄 {title} {ytxt}"),
            ui.tags.div(" · ".join([m for m in meta if m]), class_="muted", style="margin-bottom:8px;"),
            ui.tags.pre(
                textwrap.fill(txt, width=110) if txt else "Kein Text in rotten_tomatoes_movies.csv gefunden.",
                style="white-space:pre-wrap;margin:0;",
            ),
        )

    @output
    @render.data_frame
    def tbl_rt_terms():
        r = rt_selected_row()
        src = input.rt_text_source()
        txt = str(r.get(src, "") or "")
        if not txt.strip():
            return pd.DataFrame(columns=["Wort", "Häufigkeit"])

        stop = set(
            """
        the a an and or of to in on for with as is are was were be been being at by from this that it its
        der die das ein eine und oder zu im in auf für mit als ist sind war waren sein
        """.split()
        )
        words = re.findall(r"[A-Za-zÄÖÜäöüß']{3,}", txt.lower())
        words = [w for w in words if w not in stop]
        if not words:
            return pd.DataFrame(columns=["Wort", "Häufigkeit"])
        s = pd.Series(words).value_counts().head(20)
        return pd.DataFrame({"Wort": s.index, "Häufigkeit": s.values})

    # KI Trigger + Result Cache
    ai_state = reactive.Value({"note": "Noch keine Analyse gestartet."})

    @reactive.Effect
    @reactive.event(input.run_ai)
    def _run_ai():
        r = rt_selected_row()
        src = input.rt_text_source()
        txt = str(r.get(src, "") or "")
        res = openai_analyze_text(txt)
        ai_state.set(res)

    @output
    @render.ui
    def ai_result_block():
        res = ai_state.get()
        if not isinstance(res, dict):
            return ui.tags.div("—")

        if "error" in res:
            raw = res.get("raw", "")
            return ui.card(
                ui.card_header("⚠️ KI-Analyse"),
                ui.tags.div(res["error"], style="margin-bottom:8px;"),
                ui.tags.pre(raw, style="white-space:pre-wrap;") if raw else ui.tags.div(),
            )

        topics = res.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]
        if isinstance(topics, list):
            topics_txt = ", ".join([str(x) for x in topics if str(x).strip()])
        else:
            topics_txt = ""

        return ui.card(
            ui.card_header("🤖 KI-Analyse (OpenAI)"),
            ui.tags.div(f"Sentiment: {res.get('sentiment','—')}", style="margin-bottom:6px;"),
            ui.tags.div("Themen: " + (topics_txt or "—"), style="margin-bottom:10px;"),
            ui.tags.pre(
                textwrap.fill(str(res.get("summary", "") or "—"), width=110),
                style="white-space:pre-wrap;margin:0;",
            ),
        )

    # ---------------- Graph (Subgraph + schöner Netzplan) ----------------
    @reactive.Calc
    def graph_selected() -> tuple[str, str]:
        k = input.graph_node()
        if not k or "::" not in str(k):
            return ("", "")
        t, n = str(k).split("::", 1)
        return (t.strip(), n.strip())

    @reactive.Calc
    def graph_sub_edges() -> pd.DataFrame:
        if edges.empty:
            return edges.iloc[0:0]

        node_type, node_norm = graph_selected()
        if not node_type or not node_norm:
            return edges.iloc[0:0]

        hops = int(input.graph_hops())
        hops = 1 if hops < 1 else (2 if hops > 2 else hops)

        sub = edges[
            ((edges["source_type"] == node_type) & (edges["source_norm"] == node_norm))
            | ((edges["target_type"] == node_type) & (edges["target_norm"] == node_norm))
        ].copy()

        if hops == 1:
            return sub.head(800)

        # 2-hop
        neigh: set[tuple[str, str]] = set()
        for _, r in sub.iterrows():
            if r["source_type"] == node_type and r["source_norm"] == node_norm:
                neigh.add((r["target_type"], r["target_norm"]))
            if r["target_type"] == node_type and r["target_norm"] == node_norm:
                neigh.add((r["source_type"], r["source_norm"]))

        mask = pd.Series(False, index=edges.index)
        for (t, n) in neigh:
            mask = mask | (
                ((edges["source_type"] == t) & (edges["source_norm"] == n))
                | ((edges["target_type"] == t) & (edges["target_norm"] == n))
            )

        sub2 = edges[mask].copy()
        out = pd.concat([sub, sub2], ignore_index=True).drop_duplicates()
        return out.head(1200)

    @output
    @render.data_frame
    def tbl_graph_edges():
        d = graph_sub_edges()
        if d.empty:
            return pd.DataFrame(columns=["Quelle", "Relation", "Ziel", "Quelle_Typ", "Ziel_Typ"])
        show = d[["source", "relation", "target", "source_type", "target_type"]].copy()
        show.columns = ["Quelle", "Relation", "Ziel", "Quelle_Typ", "Ziel_Typ"]
        return show.head(400)

    @output
    @render.plot
    def p_graph_nice():
        fig, ax = plt.subplots(figsize=(10, 6))
        d = graph_sub_edges()

        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Graphdaten für die aktuelle Auswahl.", ha="center", va="center")
            return fig

        node_type, node_norm = graph_selected()
        max_nodes = int(input.graph_max_nodes())

        if not HAS_NX:
            ax.axis("off")
            lines = []
            for _, r in d.head(25).iterrows():
                lines.append(f"{r['source_type']}:{r['source']} --{r['relation']}--> {r['target_type']}:{r['target']}")
            ax.text(0.01, 0.99, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)
            ax.set_title("Graph-Fallback (erste 25 Kanten)")
            return fig

        # NetworkX Graph bauen
        G = nx.Graph()
        for _, r in d.iterrows():
            s_id = f"{r['source_type']}::{r['source_norm']}"
            t_id = f"{r['target_type']}::{r['target_norm']}"
            # label separat speichern (Originalname)
            G.add_node(s_id, label=str(r["source"]), ntype=str(r["source_type"]))
            G.add_node(t_id, label=str(r["target"]), ntype=str(r["target_type"]))
            G.add_edge(s_id, t_id, relation=str(r["relation"]))

        # Fokusnode
        focus = f"{node_type}::{node_norm}"
        if focus not in G:
            focus = list(G.nodes)[0]

        # wenn zu groß: auf ego-graph + degree-cut reduzieren
        if G.number_of_nodes() > max_nodes:
            # erst ego um Fokus
            ego = nx.ego_graph(G, focus, radius=int(input.graph_hops()))
            # dann nach degree (wichtige Knoten behalten)
            deg = dict(ego.degree())
            keep = sorted(deg.keys(), key=lambda k: deg[k], reverse=True)[:max_nodes]
            G = ego.subgraph(keep).copy()

        # Layout
        try:
            pos = nx.spring_layout(G, seed=42, k=0.9, iterations=80)
        except Exception:
            pos = nx.random_layout(G, seed=42)

        # Styling: Farben pro Typ + Node size nach Degree
        type_color = {"movie": "#2563EB", "actor": "#10B981", "genre": "#F59E0B"}
        degrees = dict(G.degree())
        sizes = [220 + 70 * min(degrees.get(n, 1), 10) for n in G.nodes]
        node_colors = [type_color.get(G.nodes[n].get("ntype", ""), "#94A3B8") for n in G.nodes]

        # Edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=1.2)

        # Nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors, linewidths=0.8, edgecolors="#0f172a")

        # Labels: Fokus + wichtigste Nachbarn + Top-degree
        labels = {}
        if focus in G:
            labels[focus] = G.nodes[focus].get("label", "focus")
            # Nachbarn des Fokus
            for nb in list(G.neighbors(focus))[:18]:
                labels[nb] = G.nodes[nb].get("label", "")
        # Top-degree ergänzen
        top_nodes = sorted(G.nodes, key=lambda n: degrees.get(n, 0), reverse=True)[:10]
        for n in top_nodes:
            labels.setdefault(n, G.nodes[n].get("label", ""))

        # Labels zeichnen
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

        # Legende manuell (klein)
        ax.text(0.01, 0.01, "movie / actor / genre", transform=ax.transAxes, fontsize=9, alpha=0.7)

        ax.set_title(f"Netzplan (aus RT gebaut) — Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}  | Fokus: {labels.get(focus,'')}")
        ax.axis("off")
        return fig


app = App(app_ui, server)
