# app.py — Movie Ratings Dashboard
# - Ratings: nutzt rt_metrics.csv bevorzugt (Fallback: rotten_tomatoes_movies.csv raw)
# - Text + Graph: nutzt rotten_tomatoes_movies.csv (movie_info / critics_consensus + actor/genre/etc.)
from __future__ import annotations

import re, logging, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import textwrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# optional: networkx (wenn installiert -> schöneres Layout)
try:
    import networkx as nx  # type: ignore

    HAS_NX = True
except Exception:
    nx = None
    HAS_NX = False

# Warnings leiser
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
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
CSV_RT_RAW = _find(
    BASE / "outputs/rotten_tomatoes_movies.csv", BASE / "raw/rotten_tomatoes_movies.csv"
)
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


def _pick_col(df: pd.DataFrame, *names: str) -> str | None:
    L = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in L:
            return L[n.lower()]
    return None


def _split_list_field(x: str) -> List[str]:
    if pd.isna(x):
        return []
    parts = [p.strip() for p in str(x).split(",")]
    return [p for p in parts if p]


def _short_label(s: str, maxlen: int = 18) -> str:
    s = str(s)
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 1] + "…"


# ---------------- Ratings: RT standardisieren (für Merge mit joined) ----------------
def std_rt_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: RT-Rohdatei in Standardform bringen (ratings)."""
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
            d["info"].astype(str).str.extract(r"\b(19|20)\d{2}\b", expand=False).astype(float)
        )
    d = d[d["year"].fillna(0) <= 2020]  # RT bis 2020

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
    d = d[d["year"].fillna(0) <= 2020]
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

    if "year" not in df.columns and "Year" in df.columns:
        df.rename(columns={"Year": "year"}, inplace=True)
    if "numVotes" not in df.columns and "votes" in df.columns:
        df.rename(columns={"votes": "numVotes"}, inplace=True)

    cols = ["title_norm", "year"]
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if not rt_std.empty and all(c in df.columns for c in cols):
        df = df.merge(rt_std, on=cols, how="left", suffixes=("", "_rtstd"))
        # vorhandene RT behalten, fehlende füllen
        if "rt_tomato" not in df and "rt_tomato_rtstd" in df:
            df["rt_tomato"] = df["rt_tomato_rtstd"]
        else:
            df["rt_tomato"] = df["rt_tomato"].fillna(df.get("rt_tomato_rtstd"))
        if "rt_audience" not in df and "rt_audience_rtstd" in df:
            df["rt_audience"] = df["rt_audience_rtstd"]
        else:
            df["rt_audience"] = df["rt_audience"].fillna(df.get("rt_audience_rtstd"))

    if "averageRating" in df:
        df["IMDb_Score_100"] = (df["averageRating"] * 10).clip(0, 100)
    if "rt_tomato" in df:
        df["RT_Tomatometer"] = df["rt_tomato"].clip(0, 100)
    if "rt_audience" in df:
        df["RT_Audience"] = df["rt_audience"].clip(0, 100)
    if "numVotes" in df:
        df["IMDb_Votes"] = pd.to_numeric(df["numVotes"], errors="coerce").astype("Int64")
    return df


# =========================
# Text + Graph: aus rotten_tomatoes_movies.csv
# =========================
def std_rt_text_graph(df: pd.DataFrame) -> pd.DataFrame:
    """Normiert RT raw für Text & Graph (Plot/Consensus + Genres + Actors ...)."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "title",
                "title_norm",
                "year",
                "movie_info",
                "critics_consensus",
                "genres",
                "actors",
                "directors",
                "authors",
                "rt_tomato",
                "rt_audience",
            ]
        )

    c_title = _pick_col(df, "movie_title", "title", "name")
    c_year = _pick_col(df, "original_release_year", "year")
    c_date = _pick_col(df, "original_release_date")
    c_info = _pick_col(df, "movie_info", "info", "plot", "synopsis")
    c_cons = _pick_col(df, "critics_consensus", "consensus")
    c_gen = _pick_col(df, "genres", "genre")
    c_act = _pick_col(df, "actors", "cast")
    c_dir = _pick_col(df, "directors", "director")
    c_auth = _pick_col(df, "authors", "writer", "writers")
    c_tom = _pick_col(df, "tomatometer_rating", "tomato_score", "rt_tomato")
    c_aud = _pick_col(df, "audience_rating", "audience_score", "rt_audience")

    keep = [c for c in [c_title, c_year, c_date, c_info, c_cons, c_gen, c_act, c_dir, c_auth, c_tom, c_aud] if c]
    d = df[keep].copy() if keep else df.copy()

    if c_title:
        d.rename(columns={c_title: "title"}, inplace=True)
    if c_info:
        d.rename(columns={c_info: "movie_info"}, inplace=True)
    if c_cons:
        d.rename(columns={c_cons: "critics_consensus"}, inplace=True)
    if c_gen:
        d.rename(columns={c_gen: "genres"}, inplace=True)
    if c_act:
        d.rename(columns={c_act: "actors"}, inplace=True)
    if c_dir:
        d.rename(columns={c_dir: "directors"}, inplace=True)
    if c_auth:
        d.rename(columns={c_auth: "authors"}, inplace=True)
    if c_tom:
        d.rename(columns={c_tom: "rt_tomato"}, inplace=True)
    if c_aud:
        d.rename(columns={c_aud: "rt_audience"}, inplace=True)

    if "year" not in d.columns:
        d["year"] = pd.NA

    if c_year:
        d["year"] = pd.to_numeric(d.get("year"), errors="coerce")
    if (d["year"].isna().all() or "year" not in d.columns) and c_date:
        d["year"] = pd.to_datetime(df[c_date], errors="coerce").dt.year

    d["title"] = d.get("title", "").astype(str)
    d["title_norm"] = d["title"].map(norm_title)

    for col in ["movie_info", "critics_consensus", "genres", "actors", "directors", "authors"]:
        if col not in d.columns:
            d[col] = ""
        d[col] = d[col].astype(str).fillna("")

    d["rt_tomato"] = pd.to_numeric(d.get("rt_tomato", pd.NA), errors="coerce").clip(0, 100)
    d["rt_audience"] = pd.to_numeric(d.get("rt_audience", pd.NA), errors="coerce").clip(0, 100)

    d = d.dropna(subset=["title_norm"]).drop_duplicates(subset=["title_norm"])
    return d[
        [
            "title",
            "title_norm",
            "year",
            "movie_info",
            "critics_consensus",
            "genres",
            "actors",
            "directors",
            "authors",
            "rt_tomato",
            "rt_audience",
        ]
    ]


rt_tg = std_rt_text_graph(rt_raw)


def build_edges_from_rt(df: pd.DataFrame) -> pd.DataFrame:
    """Edges aus RT raw erzeugen (movie->genre/actor/director/author + actor->genre abgeleitet)."""
    cols = ["source", "target", "relation", "source_type", "target_type", "source_norm", "target_norm"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict] = []
    for _, r in df.iterrows():
        title = str(r.get("title", "")).strip()
        if not title:
            continue

        m_norm = norm_title(title)
        genres = _split_list_field(r.get("genres", ""))
        actors = _split_list_field(r.get("actors", ""))
        directors = _split_list_field(r.get("directors", ""))
        authors = _split_list_field(r.get("authors", ""))

        for g in genres:
            rows.append(
                dict(
                    source=title,
                    target=g,
                    relation="has_genre",
                    source_type="movie",
                    target_type="genre",
                    source_norm=m_norm,
                    target_norm=norm_title(g),
                )
            )
        for a in actors:
            rows.append(
                dict(
                    source=title,
                    target=a,
                    relation="has_actor",
                    source_type="movie",
                    target_type="actor",
                    source_norm=m_norm,
                    target_norm=norm_title(a),
                )
            )
        for d in directors:
            rows.append(
                dict(
                    source=title,
                    target=d,
                    relation="directed_by",
                    source_type="movie",
                    target_type="director",
                    source_norm=m_norm,
                    target_norm=norm_title(d),
                )
            )
        for au in authors:
            rows.append(
                dict(
                    source=title,
                    target=au,
                    relation="written_by",
                    source_type="movie",
                    target_type="author",
                    source_norm=m_norm,
                    target_norm=norm_title(au),
                )
            )

        # actor -> genre (abgeleitet)
        for a in actors:
            a_norm = norm_title(a)
            for g in genres:
                rows.append(
                    dict(
                        source=a,
                        target=g,
                        relation="acts_in_genre",
                        source_type="actor",
                        target_type="genre",
                        source_norm=a_norm,
                        target_norm=norm_title(g),
                    )
                )

    e = pd.DataFrame(rows, columns=cols)
    if e.empty:
        return pd.DataFrame(columns=cols)

    for c in cols:
        e[c] = e[c].astype(str)

    e["source_norm"] = e["source_norm"].map(norm_title)
    e["target_norm"] = e["target_norm"].map(norm_title)
    e = e[(e["source_norm"] != "") & (e["target_norm"] != "")]
    e = e.drop_duplicates(subset=["source", "target", "relation", "source_type", "target_type"])
    return e[cols]


edges = build_edges_from_rt(rt_tg)

# =========================
# Fallback: hübsches Force-Layout ohne networkx
# =========================
def spring_layout_numpy(nodes: List[str], edges_pairs: List[Tuple[str, str]], seed: int = 42, iters: int = 250) -> Dict[str, Tuple[float, float]]:
    """
    Sehr simples Force-Directed Layout (Fruchterman-Reingold inspiriert).
    Reicht für einen "Netzplan", wenn networkx nicht verfügbar ist.
    """
    rng = np.random.default_rng(seed)
    n = len(nodes)
    if n == 0:
        return {}

    idx = {nodes[i]: i for i in range(n)}
    # Initial positions
    pos = rng.normal(0, 1, size=(n, 2)).astype(float)

    # Build adjacency list from edges
    E = []
    for u, v in edges_pairs:
        if u in idx and v in idx and u != v:
            E.append((idx[u], idx[v]))

    # constants
    area = 4.0
    k = np.sqrt(area / max(n, 1))
    temperature = 0.5

    def cool(t, step):
        return t * (1.0 - step / max(iters, 1))

    for step in range(iters):
        disp = np.zeros((n, 2), dtype=float)

        # Repulsive
        for i in range(n):
            delta = pos[i] - pos  # (n,2)
            dist2 = (delta[:, 0] ** 2 + delta[:, 1] ** 2) + 1e-6
            inv = (k * k) / dist2
            # weighted sum ignoring self
            inv[i] = 0.0
            disp[i] += (delta * inv[:, None]).sum(axis=0)

        # Attractive
        for (a, b) in E:
            delta = pos[a] - pos[b]
            dist = np.sqrt(delta[0] ** 2 + delta[1] ** 2) + 1e-6
            force = (dist * dist) / k
            vec = delta / dist
            disp[a] -= vec * force
            disp[b] += vec * force

        # Update positions with limited step (temperature)
        lengths = np.sqrt((disp[:, 0] ** 2 + disp[:, 1] ** 2)) + 1e-9
        step_vec = disp / lengths[:, None] * np.minimum(lengths, temperature)[:, None]
        pos += step_vec

        # Normalize into a reasonable window
        pos -= pos.mean(axis=0)
        mx = np.max(np.abs(pos)) + 1e-6
        pos /= mx

        temperature = cool(temperature, step)

    return {nodes[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


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
                "rt_text": "RT Textdaten",
                "graph": "Graph (RT Netzwerk)",
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
        ui.tags.div("RT Textdaten", class_="muted"),
        ui.input_selectize("rt_title", "Film auswählen", choices=[], selected=None, multiple=False),
        ui.input_radio_buttons(
            "rt_text_field",
            "Textquelle",
            choices={"movie_info": "Plot / movie_info", "critics_consensus": "Critics Consensus", "both": "Beides"},
            selected="movie_info",
            inline=False,
        ),
        ui.tags.hr(),
        ui.tags.div("Graph (RT Netzwerk)", class_="muted"),
        ui.input_selectize("graph_node", "Knoten auswählen", choices=[], selected=None, multiple=False),
        ui.input_numeric("graph_hops", "Hops", value=1, min=1, max=2, step=1),
        ui.input_numeric("graph_max_edges", "Max. Edges im Plot", value=250, min=50, max=800, step=50),
        ui.input_checkbox("graph_show_labels", "Labels anzeigen (nur Fokus + Nachbarn)", True),
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

    # ---- Dropdowns nur einmal initialisieren (ohne session.user_data) ----
    dropdowns_inited = reactive.Value(False)

    @reactive.Effect
    def _init_dropdowns_once():
        if dropdowns_inited():
            return

        # RT titles (für Texttab)
        if not rt_tg.empty:
            tmp = rt_tg[["title", "title_norm"]].drop_duplicates().sort_values("title").head(9000)
            rt_choices = {r["title_norm"]: r["title"] for _, r in tmp.iterrows()}
            ui.update_selectize("rt_title", choices=rt_choices)
            if rt_choices:
                ui.update_selectize("rt_title", selected=next(iter(rt_choices.keys())))
        else:
            ui.update_selectize("rt_title", choices={})

        # Graph nodes (aus edges generiert)
        if not edges.empty:
            nodes_src = edges[["source", "source_norm", "source_type"]].rename(
                columns={"source": "label", "source_norm": "norm", "source_type": "type"}
            )
            nodes_tgt = edges[["target", "target_norm", "target_type"]].rename(
                columns={"target": "label", "target_norm": "norm", "target_type": "type"}
            )
            nodes = pd.concat([nodes_src, nodes_tgt], ignore_index=True).drop_duplicates(subset=["norm", "type"])
            nodes = nodes.sort_values(["type", "label"]).head(14000)
            g_choices = {f"{r['type']}::{r['norm']}": f"{r['type']}: {r['label']}" for _, r in nodes.iterrows()}
            ui.update_selectize("graph_node", choices=g_choices)
            if g_choices:
                ui.update_selectize("graph_node", selected=next(iter(g_choices.keys())))
        else:
            ui.update_selectize("graph_node", choices={})

        dropdowns_inited.set(True)

    # ---------------- Status ----------------
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
            ui.tags.li(f"RT-Quelle (Ratings): {RT_SOURCE}"),
            ui.tags.li(f"RT Ratings (≤2020): {len(rt_std):,}".replace(",", ".")),
            ui.tags.li("—"),
            ui.tags.li(f"RT Text/Graph Titles: {len(rt_tg):,}".replace(",", ".")),
            ui.tags.li(f"Edges (aus RT erzeugt): {len(edges):,}".replace(",", ".")),
            ui.tags.li(f"networkx verfügbar: {'ja' if HAS_NX else 'nein (Fallback spring layout in numpy)'}"),
        ]
        return ui.tags.small(ui.tags.ul(*rows, style="margin:0;padding-left:18px;"))

    # ---------------- Data Views (Ratings) ----------------
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

    @reactive.Calc
    def df_without_rt():
        d = df_joined()
        col = "RT_Tomatometer"
        return d[d[col].isna()] if not d.empty and col in d.columns else d

    # ---------------- Texttab: ausgewählter Film ----------------
    @reactive.Calc
    def rt_selected_row() -> pd.Series:
        key = input.rt_title()
        if rt_tg.empty or key is None:
            return pd.Series({"title": "", "title_norm": "", "movie_info": "", "critics_consensus": ""})
        hit = rt_tg[rt_tg["title_norm"] == str(key)]
        if hit.empty:
            return pd.Series({"title": "", "title_norm": str(key), "movie_info": "", "critics_consensus": ""})
        return hit.iloc[0]

    def rt_text_for_analysis(row: pd.Series) -> str:
        mode = str(input.rt_text_field())
        info = str(row.get("movie_info", "") or "")
        cons = str(row.get("critics_consensus", "") or "")
        if mode == "movie_info":
            return info
        if mode == "critics_consensus":
            return cons
        return (info.strip() + "\n\n" + cons.strip()).strip()

    # ---------------- Graphtab: Auswahl + Subgraph ----------------
    @reactive.Calc
    def graph_selected() -> Tuple[str, str]:
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
            return sub.head(int(input.graph_max_edges()))

        neigh: set[Tuple[str, str]] = set()
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
        return out.head(int(input.graph_max_edges()))

    # ---------------- Titel ----------------
    @output
    @render.ui
    def page_title():
        mapping = {
            "overview": "Übersicht",
            "compare": "Vergleich IMDb ↔ Rotten Tomatoes",
            "coverage": "Abdeckung Rotten Tomatoes",
            "trends": "Trends (IMDb)",
            "top20": "Top 20 (IMDb-Stimmen)",
            "gtrends": "Google Trends",
            "rt_only": "Rotten Tomatoes — eigene Sicht",
            "rt_text": "Rotten Tomatoes — Textdaten (Plot/Consensus)",
            "graph": "Graph — Netzplan (Movie / Actor / Genre ...)",
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
        if p == "rt_text":
            return ui.div(ui.output_ui("rt_text_block"), ui.output_data_frame("tbl_rt_terms"))
        if p == "graph":
            return ui.div(ui.output_plot("p_graph_edges"), ui.output_data_frame("tbl_graph_edges"))
        if p == "table":
            return ui.div(ui.output_data_frame("tbl_all"))
        return ui.div("—")

    # ---------------- KPIs ----------------
    def kpi_ui():
        d_all = df_joined()
        d_rt = df_with_rt()
        cards = []
        cards.append(ui.value_box(title="Filme (gefiltert)", value=f"{len(d_all):,}".replace(",", ".")))
        if "IMDb_Score_100" in d_all and d_all["IMDb_Score_100"].notna().any():
            cards.append(ui.value_box(title="Ø IMDb (0–100)", value=f"{d_all['IMDb_Score_100'].mean():.1f}"))
        if not d_rt.empty and d_rt["RT_Tomatometer"].notna().any():
            cards.append(ui.value_box(title="Ø RT Tomatometer", value=f"{d_rt['RT_Tomatometer'].mean():.1f}"))
        if input.use_audience() and not d_rt.empty and "RT_Audience" in d_rt and d_rt["RT_Audience"].notna().any():
            cards.append(ui.value_box(title="Ø RT Audience", value=f"{d_rt['RT_Audience'].mean():.1f}"))
        share = (len(d_rt) / len(d_all) * 100) if len(d_all) > 0 else 0
        cards.append(ui.value_box(title="RT-Abdeckung", value=f"{share:.1f}%"))
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
        ax.bar(labels, values)
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
        ax.plot(xs, 1 - ys, linewidth=2)
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
        ax.scatter(d["IMDb_Score_100"], d["RT_Tomatometer"], s=10, alpha=0.4)
        ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.5, alpha=0.9)
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
        ax.plot(s.index, s.values, marker="o", linewidth=2)
        ax.axhline(0, linewidth=1)
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
        ax.plot(share.index, share.values, marker="o", linewidth=2)
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
        ax.pie(
            counts.values,
            labels=[f"{k} ({int(v)})" for k, v in counts.items()],
            autopct="%1.0f%%",
            startangle=90,
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
        ax.bar(s.index, s["avg"])
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
        ax.plot(s.index, s["avg"], marker="o", linewidth=2)
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
        ax.barh(labels, d["numVotes"].values)
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
        labels, values = [], []
        if d["rt_tomato"].notna().any():
            labels.append(f"Tomatometer\nN={d['rt_tomato'].notna().sum():,}".replace(",", "."))
            values.append(d["rt_tomato"].mean())
        if input.use_audience() and d["rt_audience"].notna().any():
            labels.append(f"Audience\nN={d['rt_audience'].notna().sum():,}".replace(",", "."))
            values.append(d["rt_audience"].mean())
        if not values:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine RT-Werte", ha="center", va="center")
            return fig
        ax.bar(labels, values)
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Rotten Tomatoes (nur RT-Daten, bis 2020) — Ø Werte")
        ax.set_ylabel("Punkte")
        return fig

    # ---------------- RT Text Outputs ----------------
    @output
    @render.ui
    def rt_text_block():
        row = rt_selected_row()
        title = str(row.get("title", "")) or "—"
        txt = rt_text_for_analysis(row)
        meta = []
        g = str(row.get("genres", "") or "")
        if g.strip():
            meta.append(f"Genres: {g}")
        return ui.card(
            ui.card_header(f"📄 Textdaten: {title}"),
            ui.tags.div(" · ".join(meta), class_="muted", style="margin-bottom:8px;"),
            ui.tags.pre(
                textwrap.fill(txt, width=110) if txt else "Kein Text in RT vorhanden.",
                style="white-space:pre-wrap;margin:0;",
            ),
        )

    @output
    @render.data_frame
    def tbl_rt_terms():
        row = rt_selected_row()
        txt = rt_text_for_analysis(row)
        if not txt.strip():
            return pd.DataFrame(columns=["Wort", "Häufigkeit"])
        stop = set(
            """
            the a an and or of to in on for with as is are was were be been being at by from this that it its
            der die das ein eine und oder zu im in auf für mit als ist sind war waren sein
            """
            .split()
        )
        words = re.findall(r"[A-Za-zÄÖÜäöüß']{3,}", txt.lower())
        words = [w for w in words if w not in stop]
        if not words:
            return pd.DataFrame(columns=["Wort", "Häufigkeit"])
        s = pd.Series(words).value_counts().head(25)
        return pd.DataFrame({"Wort": s.index, "Häufigkeit": s.values})

    # ---------------- Graph Outputs ----------------
    @output
    @render.data_frame
    def tbl_graph_edges():
        d = graph_sub_edges()
        if d.empty:
            return pd.DataFrame(columns=["Quelle", "Relation", "Ziel", "Quelle_Typ", "Ziel_Typ"])
        show = d[["source", "relation", "target", "source_type", "target_type"]].copy()
        show.columns = ["Quelle", "Relation", "Ziel", "Quelle_Typ", "Ziel_Typ"]
        return show.head(int(input.graph_max_edges()))

    @output
    @render.plot
    def p_graph_edges():
        fig, ax = plt.subplots(figsize=(9, 5))
        d = graph_sub_edges()
        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Edges für die aktuelle Auswahl.", ha="center", va="center")
            return fig

        node_type, node_norm = graph_selected()

        # Build node keys
        def node_key(t: str, label: str) -> str:
            return f"{t}:{label}"

        # Limit for readability
        d = d.head(int(input.graph_max_edges())).copy()

        # Build edge list (for plotting)
        edge_pairs: List[Tuple[str, str]] = []
        node_types: Dict[str, str] = {}

        for _, r in d.iterrows():
            s = node_key(r["source_type"], r["source"])
            t = node_key(r["target_type"], r["target"])
            edge_pairs.append((s, t))
            node_types[s] = str(r["source_type"])
            node_types[t] = str(r["target_type"])

        nodes = sorted(set([u for u, _ in edge_pairs] + [v for _, v in edge_pairs]))

        # Find focus key (best effort)
        focus = None
        for _, r in d.iterrows():
            if r["source_type"] == node_type and norm_title(r["source"]) == node_norm:
                focus = node_key(r["source_type"], r["source"])
                break
            if r["target_type"] == node_type and norm_title(r["target"]) == node_norm:
                focus = node_key(r["target_type"], r["target"])
                break

        # Degree (for sizing)
        deg = {n: 0 for n in nodes}
        for u, v in edge_pairs:
            if u in deg:
                deg[u] += 1
            if v in deg:
                deg[v] += 1

        # Colors per type (legend)
        type_palette = {
            "movie": "#2563EB",
            "actor": "#10B981",
            "genre": "#F59E0B",
            "director": "#8B5CF6",
            "author": "#EF4444",
        }

        def color_for(n: str) -> str:
            t = node_types.get(n, "")
            return type_palette.get(t, "#64748B")

        # Layout
        if HAS_NX:
            G = nx.Graph()
            for u, v in edge_pairs:
                G.add_edge(u, v)
            try:
                pos = nx.spring_layout(G, seed=42, k=0.9)
            except Exception:
                pos = nx.random_layout(G, seed=42)
            pos_dict = {n: (float(pos[n][0]), float(pos[n][1])) for n in G.nodes}
        else:
            pos_dict = spring_layout_numpy(nodes, edge_pairs, seed=42, iters=260)

        # Draw edges
        for u, v in edge_pairs:
            if u not in pos_dict or v not in pos_dict:
                continue
            x1, y1 = pos_dict[u]
            x2, y2 = pos_dict[v]
            ax.plot([x1, x2], [y1, y2], alpha=0.25, linewidth=1.1, color="#111827")

        # Draw nodes by type (so we can show a clean legend)
        nodes_by_type: Dict[str, List[str]] = {}
        for n in nodes:
            t = node_types.get(n, "other")
            nodes_by_type.setdefault(t, []).append(n)

        for t, ns in nodes_by_type.items():
            xs = [pos_dict[n][0] for n in ns if n in pos_dict]
            ys = [pos_dict[n][1] for n in ns if n in pos_dict]
            sizes = [60 + min(deg.get(n, 0), 25) * 9 for n in ns if n in pos_dict]
            ax.scatter(xs, ys, s=sizes, alpha=0.92, c=type_palette.get(t, "#64748B"), label=t)

        # Highlight focus
        if focus and focus in pos_dict:
            fx, fy = pos_dict[focus]
            ax.scatter([fx], [fy], s=420, alpha=0.15, c="#0F172A")
            ax.scatter([fx], [fy], s=160, alpha=1.0, c="#0F172A")

        # Labels (nur Fokus + Nachbarn)
        show_labels = bool(input.graph_show_labels())
        if show_labels and focus and focus in pos_dict:
            # neighbors
            neigh = set()
            for u, v in edge_pairs:
                if u == focus:
                    neigh.add(v)
                elif v == focus:
                    neigh.add(u)

            label_nodes = [focus] + list(neigh)[:18]
            for n in label_nodes:
                if n not in pos_dict:
                    continue
                x, y = pos_dict[n]
                # label = nur rechter Teil nach "type:"
                if ":" in n:
                    t, raw = n.split(":", 1)
                    lab = f"{t}: {_short_label(raw, 18)}"
                else:
                    lab = _short_label(n, 20)
                ax.text(x, y, lab, fontsize=8, ha="center", va="center", color="#0F172A")

        ax.set_title(
            f"Netzplan (Hops={int(input.graph_hops())}) — Nodes={len(nodes)}  Edges={len(edge_pairs)}"
            + ("  [networkx]" if HAS_NX else "  [numpy-layout]")
        )
        ax.axis("off")
        ax.set_aspect("equal", adjustable="datalim")

        # Legend (nur wenn nicht zu viele Kategorien)
        if len(nodes_by_type) <= 6:
            ax.legend(loc="upper right", fontsize=8, frameon=True)

        return fig


app = App(app_ui, server)
