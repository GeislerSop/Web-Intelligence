# app.py — Movie Ratings Dashboard (nutzt rt_metrics.csv bevorzugt)
from __future__ import annotations

import re, logging, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# optional: Graph-Plot (wenn installiert)
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
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOG = logging.getLogger("movie-app")

# ---------------- Pfade/Loader ----------------
BASE = Path(__file__).resolve().parent

def _find(*candidates: Path) -> Path|None:
    for p in candidates:
        if p and p.exists():
            return p
    return None

CSV_JOINED   = _find(BASE/"outputs/joined_imdb_rt.csv", BASE/"joined_imdb_rt.csv")
CSV_RT_METR  = _find(BASE/"outputs/rt_metrics.csv")
CSV_RT_RAW   = _find(BASE/"outputs/rotten_tomatoes_movies.csv", BASE/"raw/rotten_tomatoes_movies.csv")
CSV_TOP20    = _find(BASE/"outputs/top20_by_votes_imdb.csv", BASE/"top20_by_votes_imdb.csv")
CSV_GTR      = _find(BASE/"outputs/google_trends_top5.csv", BASE/"google_trends_top5.csv")

# --- Textdaten (Wikidata Cache)
CSV_WD_TEXT  = _find(BASE/"outputs/text_wikidata_cache.csv", BASE/"text_wikidata_cache.csv")

# --- NEU: Graphdaten (Edges) im gewünschten Format:
# source,target,relation,source_type,target_type,source_norm,target_norm
CSV_EDGES    = _find(BASE/"outputs/edges.csv", BASE/"edges.csv")

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

joined_raw = _read_csv(CSV_JOINED)
rt_metrics = _read_csv(CSV_RT_METR)
rt_raw     = _read_csv(CSV_RT_RAW)
top20_raw  = _read_csv(CSV_TOP20)
gtr_raw    = _read_csv(CSV_GTR)

wd_text_raw = _read_csv(CSV_WD_TEXT)
edges_raw   = _read_csv(CSV_EDGES)

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

def std_rt_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: RT-Rohdatei in Standardform bringen."""
    if df.empty:
        return pd.DataFrame(columns=["title_norm","year","rt_tomato","rt_audience"])
    L={c.lower():c for c in df.columns}
    def pick(*xs):
        for x in xs:
            if x in L: return L[x]
    c_title=pick("movie_title","title","name")
    c_year =pick("original_release_year","year","release_year")
    c_info =pick("movie_info")
    c_tom =pick("tomatometer_rating","tomato_score","tomatometer","rt","rt_score")
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
    d = d[d["year"].fillna(0) <= 2020]  # RT bis 2020
    d["rt_tomato"]=d.get("rt_tomato_raw",pd.Series(index=d.index)).map(to100)
    d["rt_audience"]=d.get("rt_audience_raw",pd.Series(index=d.index)).map(to100)
    return d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()

def std_rt_from_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Bevorzugt: aus rt_metrics.csv lesen und vereinheitlichen."""
    need = {"title","title_norm","year","rt_tomato","rt_audience"}
    if df.empty or not need.issubset(set(df.columns)):
        return pd.DataFrame()
    d = df.copy()
    d["title_norm"] = d["title_norm"].astype(str).map(norm_title)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[d["year"].fillna(0) <= 2020]  # RT bis 2020
    d["rt_tomato"]   = pd.to_numeric(d["rt_tomato"], errors="coerce").clip(0,100)
    d["rt_audience"] = pd.to_numeric(d["rt_audience"], errors="coerce").clip(0,100)
    return d[["title_norm","year","rt_tomato","rt_audience"]].dropna(subset=["title_norm"]).drop_duplicates()

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
        df.rename(columns={"Year":"year"}, inplace=True)
    if "numVotes" not in df.columns and "votes" in df.columns:
        df.rename(columns={"votes":"numVotes"}, inplace=True)

    # merge RT (nur bis 2020, das macht rt_std bereits)
    cols = ["title_norm","year"]
    if "year" in df: df["year"]=pd.to_numeric(df["year"], errors="coerce")
    if not rt_std.empty and all(c in df.columns for c in cols):
        df = df.merge(rt_std, on=cols, how="left", suffixes=("","_rtstd"))
        # vorhandene RT behalten, fehlende füllen
        if "rt_tomato" not in df and "rt_tomato_rtstd" in df:
            df["rt_tomato"] = df["rt_tomato_rtstd"]
        else:
            df["rt_tomato"] = df["rt_tomato"].fillna(df.get("rt_tomato_rtstd"))
        if "rt_audience" not in df and "rt_audience_rtstd" in df:
            df["rt_audience"] = df["rt_audience_rtstd"]
        else:
            df["rt_audience"] = df["rt_audience"].fillna(df.get("rt_audience_rtstd"))

    # UI-Spalten
    if "averageRating" in df: df["IMDb_Score_100"] = (df["averageRating"]*10).clip(0,100)
    if "rt_tomato" in df:     df["RT_Tomatometer"] = df["rt_tomato"].clip(0,100)
    if "rt_audience" in df:   df["RT_Audience"] = df["rt_audience"].clip(0,100)
    if "numVotes" in df:      df["IMDb_Votes"] = pd.to_numeric(df["numVotes"], errors="coerce").astype("Int64")
    return df

# ---------------- Wikidata Text standardisieren ----------------
def std_wd_text(df: pd.DataFrame) -> pd.DataFrame:
    """Wikidata-Textcache robust vereinheitlichen -> title, title_norm, description, qid(optional)."""
    if df.empty:
        return pd.DataFrame(columns=["title","title_norm","description","qid"])
    L = {c.lower(): c for c in df.columns}
    def pick(*xs):
        for x in xs:
            if x in L: return L[x]
        return None

    c_title = pick("title", "label", "movie", "film", "name")
    c_desc  = pick("description", "desc", "text", "plot", "abstract", "summary")
    c_qid   = pick("qid", "id", "wikidata_id", "entity", "item")

    keep = [c for c in [c_title, c_desc, c_qid] if c]
    d = df[keep].copy() if keep else df.copy()
    if c_title and c_title != "title":
        d.rename(columns={c_title: "title"}, inplace=True)
    if c_desc and c_desc != "description":
        d.rename(columns={c_desc: "description"}, inplace=True)
    if c_qid and c_qid != "qid":
        d.rename(columns={c_qid: "qid"}, inplace=True)

    if "title" not in d.columns:
        d["title"] = df.iloc[:, 0].astype(str)
    if "description" not in d.columns:
        d["description"] = ""

    d["title"] = d["title"].astype(str)
    d["title_norm"] = d["title"].map(norm_title)
    if "qid" not in d.columns:
        d["qid"] = pd.NA
    d["description"] = d["description"].astype(str).fillna("")
    d = d.dropna(subset=["title_norm"]).drop_duplicates(subset=["title_norm"])
    return d[["title","title_norm","description","qid"]]

wd_text = std_wd_text(wd_text_raw)

# ---------------- NEU: Edges (typed) standardisieren – passend zu deinem Format ----------------
def std_edges_typed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartetes Format:
    source,target,relation,source_type,target_type,source_norm,target_norm
    """
    cols = ["source","target","relation","source_type","target_type","source_norm","target_norm"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    need = set(cols)
    if not need.issubset(df.columns):
        # Wenn du mal eine andere Edges-Datei reinwirfst, lieber leer statt crash
        missing = sorted(list(need - set(df.columns)))
        LOG.warning(f"edges.csv missing columns: {missing} -> graph tab disabled")
        return pd.DataFrame(columns=cols)

    d = df.copy()
    for c in cols:
        d[c] = d[c].astype(str)

    d["source_norm"] = d["source_norm"].map(norm_title)
    d["target_norm"] = d["target_norm"].map(norm_title)

    d = d[(d["source_norm"] != "") & (d["target_norm"] != "")]
    d = d.drop_duplicates(subset=["source","target","relation","source_type","target_type"])
    return d[cols]

edges = std_edges_typed(edges_raw)

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
                "wikidata":"Wikidata Analyse (Text)",
                "graph":"Graph (Edges: Movie/Actor/Genre)",   # <-- NEU
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
        ui.tags.div("Wikidata (Text)", class_="muted"),
        ui.input_selectize("wd_title", "Film auswählen", choices=[], selected=None, multiple=False),

        ui.tags.hr(),
        ui.tags.div("Graph (Edges)", class_="muted"),
        ui.input_selectize("graph_node", "Knoten auswählen", choices=[], selected=None, multiple=False),
        ui.input_numeric("graph_hops", "Hops (1=Nachbarn, 2=Nachbarn der Nachbarn)", value=1, min=1, max=2, step=1),

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

    # -------- Wikidata Dropdown choices --------
    @reactive.Effect
    def _init_wd_choices():
        if wd_text.empty:
            choices = {}
        else:
            tmp = wd_text.copy()
            tmp["label"] = tmp["title"]
            if "qid" in tmp.columns and tmp["qid"].notna().any():
                tmp["label"] = tmp.apply(lambda r: f"{r['title']}  ·  {r['qid']}" if pd.notna(r["qid"]) else r["title"], axis=1)
            tmp = tmp.sort_values("title").head(5000)
            choices = {r["title_norm"]: r["label"] for _, r in tmp.iterrows()}

        ui.update_selectize("wd_title", choices=choices)
        if input.wd_title() is None and choices:
            ui.update_selectize("wd_title", selected=next(iter(choices.keys())))

    # -------- Graph Dropdown choices (aus edges.csv) --------
    @reactive.Effect
    def _init_graph_choices():
        if edges.empty:
            ui.update_selectize("graph_node", choices={})
            return

        nodes_src = edges[["source","source_norm","source_type"]].rename(
            columns={"source":"label","source_norm":"norm","source_type":"type"}
        )
        nodes_tgt = edges[["target","target_norm","target_type"]].rename(
            columns={"target":"label","target_norm":"norm","target_type":"type"}
        )
        nodes = pd.concat([nodes_src, nodes_tgt], ignore_index=True).drop_duplicates(subset=["norm","type"])

        nodes = nodes.sort_values(["type","label"]).head(8000)
        choices = {f"{r['type']}::{r['norm']}": f"{r['type']}: {r['label']}" for _, r in nodes.iterrows()}

        ui.update_selectize("graph_node", choices=choices)
        if input.graph_node() is None and choices:
            ui.update_selectize("graph_node", selected=next(iter(choices.keys())))

    # -------- Status / Quelle anzeigen --------
    @output
    @render.ui
    def status_files():
        def li(ok, text): return ui.tags.li((("✅ " if ok else "❌ ") + text))
        rows = [
            li(CSV_JOINED is not None, f"joined: {CSV_JOINED} (shape={tuple(joined_raw.shape)})"),
            li(CSV_RT_METR is not None, f"rt_metrics: {CSV_RT_METR} (shape={tuple(rt_metrics.shape)})"),
            li(CSV_RT_RAW is not None, f"rt_raw: {CSV_RT_RAW} (shape={tuple(rt_raw.shape)})"),
            li(CSV_TOP20 is not None, f"top20 : {CSV_TOP20}  (shape={tuple(top20_raw.shape)})"),
            li(CSV_GTR is not None, f"gtr   : {CSV_GTR}    (shape={tuple(gtr_raw.shape)})"),
            ui.tags.li(f"RT-Quelle aktiv: {RT_SOURCE}"),
            ui.tags.li(f"RT-Datensätze (≤2020): {len(rt_std):,}".replace(",", ".")),
            ui.tags.li("—"),
            li(CSV_WD_TEXT is not None, f"wikidata_text: {CSV_WD_TEXT} (shape={tuple(wd_text_raw.shape)})"),
            li(CSV_EDGES is not None, f"edges: {CSV_EDGES} (shape={tuple(edges_raw.shape)})"),
            ui.tags.li(f"Wikidata-Texte (unique titles): {len(wd_text):,}".replace(",", ".")),
            ui.tags.li(f"Edges (typed): {len(edges):,}".replace(",", ".")),
            ui.tags.li(f"networkx verfügbar: {'ja' if HAS_NX else 'nein (Fallback)'}"),
        ]
        return ui.tags.small(ui.tags.ul(*rows, style="margin:0;padding-left:18px;"))

    # -------- Daten-Sichten --------
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

    # -------- Wikidata selected --------
    @reactive.Calc
    def wd_selected_row() -> pd.Series:
        key = input.wd_title()
        if wd_text.empty or key is None:
            return pd.Series({"title":"", "title_norm":"", "description":"", "qid":pd.NA})
        hit = wd_text[wd_text["title_norm"] == key]
        if hit.empty:
            return pd.Series({"title":"", "title_norm":str(key), "description":"", "qid":pd.NA})
        return hit.iloc[0]

    # -------- Graph selected + Subgraph (1–2 hops) --------
    @reactive.Calc
    def graph_selected() -> tuple[str,str]:
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
            ((edges["source_type"] == node_type) & (edges["source_norm"] == node_norm)) |
            ((edges["target_type"] == node_type) & (edges["target_norm"] == node_norm))
        ].copy()

        if hops == 1:
            return sub.head(400)

        # 2-hop: Nachbarn sammeln
        neigh: set[tuple[str,str]] = set()
        for _, r in sub.iterrows():
            if r["source_type"] == node_type and r["source_norm"] == node_norm:
                neigh.add((r["target_type"], r["target_norm"]))
            if r["target_type"] == node_type and r["target_norm"] == node_norm:
                neigh.add((r["source_type"], r["source_norm"]))

        mask = pd.Series(False, index=edges.index)
        for (t, n) in neigh:
            mask = mask | (
                ((edges["source_type"] == t) & (edges["source_norm"] == n)) |
                ((edges["target_type"] == t) & (edges["target_norm"] == n))
            )
        sub2 = edges[mask].copy()

        out = pd.concat([sub, sub2], ignore_index=True).drop_duplicates()
        return out.head(600)

    # ---------------- Titel ----------------
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
            "wikidata":"Wikidata Analyse — Text",
            "graph":"Graph — Movie / Actor / Genre",
            "table":"Tabelle (gefiltert)",
        }
        return ui.tags.h3(mapping.get(input.page(),"Übersicht"))

    # ---------------- Routing ----------------
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
        if p == "wikidata":  return ui.div(
            ui.output_ui("wd_text_block"),
            ui.card(ui.card_header("Top-Wörter (Beschreibung)"), ui.output_data_frame("tbl_wd_terms")),
        )
        if p == "graph":     return ui.div(
            ui.card(ui.card_header("Graph-Ausschnitt"), ui.output_plot("p_graph_edges")),
            ui.card(ui.card_header("Edge-Liste (Subgraph)"), ui.output_data_frame("tbl_graph_edges")),
        )
        if p == "table":     return ui.div(ui.output_data_frame("tbl_all"))
        return ui.div("—")

    # ---------------- KPIs ----------------
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

    # ---------------- Übersicht: Balken + CCDF ----------------
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
        colors = ["#2563EB","#F59E0B","#10B981"][:len(values)]
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
        ax.plot(xs, 1-ys, color="#2563EB", linewidth=2)
        ax.set_xlabel("log10(Stimmen)  (3=1.000, 4=10.000, 5=100.000)")
        ax.set_ylabel("Anteil der Filme ≥ X")
        ax.set_title(f"Wie viele Stimmen haben die Filme? (N={x.size:,})".replace(",", "."))
        ax.set_ylim(0,1)
        return fig

    # ---------------- Vergleich: Scatter + Ø-Differenz je Bin ----------------
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
        d = d.copy()
        d["bin"] = (d["IMDb_Score_100"]//10*10).astype(int).clip(0,90)
        s = d.groupby("bin").apply(lambda x: (x["RT_Tomatometer"] - x["IMDb_Score_100"]).mean())
        ax.plot(s.index, s.values, marker="o", color="#EF4444", linewidth=2)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xticks(list(range(0,101,10)))
        ax.set_xlabel("IMDb (gerundet, 0–100)")
        ax.set_ylabel("Ø [RT − IMDb] (Punkte)")
        ax.set_title("Wo weichen RT und IMDb ab? (Ø Differenz je IMDb-Bin)")
        return fig

    # ---------------- Abdeckung + RT-Kreisdiagramm ----------------
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
        colors = ["#10B981","#EF4444"]
        ax.pie(counts.values, labels=[f"{k} ({int(v)})" for k,v in counts.items()],
               autopct="%1.0f%%", startangle=90, colors=colors, textprops={"fontsize":11})
        ax.set_title("RT-Verteilung (Fresh ≥ 60)"); ax.axis("equal")
        return fig

    # ---------------- Trends (IMDb) ----------------
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

    # ---------------- Top 20 ----------------
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

    # ---------------- Google Trends ----------------
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
            ax.plot(pd.to_datetime(d.index), d[c], label=c, linewidth=2)
        ax.legend(fontsize=8); ax.set_title("Google Trends (Top 5)")
        ax.set_xlabel("Datum"); ax.set_ylabel("Interesse (0–100)")
        return fig

    @output
    @render.data_frame
    def tbl_gtrends():
        d = gtr_raw.copy()
        return d if not d.empty else pd.DataFrame(columns=["date","kw1","kw2","kw3","kw4","kw5"])

    # ---------------- Tabelle (gefiltert) ----------------
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

    # ---------------- RT-only Seite (nur RT, bis 2020) ----------------
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

    # ---------------- Wikidata Outputs ----------------
    @output
    @render.ui
    def wd_text_block():
        row = wd_selected_row()
        title = str(row.get("title","")) or "—"
        desc = str(row.get("description","")) or ""
        qid = row.get("qid", pd.NA)
        meta = []
        if pd.notna(qid):
            meta.append(f"QID: {qid}")
        meta.append(f"Beschreibungslänge: {len(desc):,} Zeichen".replace(",", "."))
        return ui.card(
            ui.card_header(f"📄 Beschreibung: {title}"),
            ui.tags.div(" · ".join(meta), class_="muted", style="margin-bottom:8px;"),
            ui.tags.pre(textwrap.fill(desc, width=110) if desc else "Keine Beschreibung im Cache gefunden.",
                       style="white-space:pre-wrap;margin:0;")
        )

    @output
    @render.data_frame
    def tbl_wd_terms():
        row = wd_selected_row()
        txt = str(row.get("description","") or "")
        if not txt.strip():
            return pd.DataFrame(columns=["Wort","Häufigkeit"])
        stop = set("""
        the a an and or of to in on for with as is are was were be been being at by from this that it its
        der die das ein eine und oder zu im in auf für mit als ist sind war waren sein
        """.split())
        words = re.findall(r"[A-Za-zÄÖÜäöüß']{3,}", txt.lower())
        words = [w for w in words if w not in stop]
        if not words:
            return pd.DataFrame(columns=["Wort","Häufigkeit"])
        s = pd.Series(words).value_counts().head(20)
        return pd.DataFrame({"Wort": s.index, "Häufigkeit": s.values})

    # ---------------- Graph Outputs ----------------
    @output
    @render.data_frame
    def tbl_graph_edges():
        d = graph_sub_edges()
        if d.empty:
            return pd.DataFrame(columns=["Quelle","Relation","Ziel","Quelle_Typ","Ziel_Typ"])
        show = d[["source","relation","target","source_type","target_type"]].copy()
        show.columns = ["Quelle","Relation","Ziel","Quelle_Typ","Ziel_Typ"]
        return show

    @output
    @render.plot
    def p_graph_edges():
        fig, ax = plt.subplots(figsize=(9,5))
        d = graph_sub_edges()
        if d.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "Keine Edges für die aktuelle Auswahl.", ha="center", va="center")
            return fig

        node_type, node_norm = graph_selected()

        if HAS_NX:
            G = nx.Graph()
            for _, r in d.iterrows():
                s = f"{r['source_type']}:{r['source']}"
                t = f"{r['target_type']}:{r['target']}"
                G.add_edge(s, t, relation=r["relation"])

            try:
                pos = nx.spring_layout(G, seed=42, k=0.8)
            except Exception:
                pos = nx.random_layout(G, seed=42)

            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=1.2)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=220)

            # Fokusnode bestimmen
            focus = None
            for n in G.nodes:
                if n.startswith(node_type + ":"):
                    if norm_title(n.split(":",1)[1]) == node_norm:
                        focus = n
                        break

            labels = {}
            if focus and focus in G:
                labels[focus] = focus
                for nb in list(G.neighbors(focus))[:14]:
                    labels[nb] = nb

            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
            ax.set_title(
                f"Subgraph (Hops={int(input.graph_hops())}) — "
                f"Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}"
            )
            ax.axis("off")
            return fig

        # Fallback ohne networkx
        ax.axis("off")
        lines = []
        for _, r in d.head(25).iterrows():
            lines.append(f"{r['source_type']}:{r['source']}  --{r['relation']}-->  {r['target_type']}:{r['target']}")
        ax.text(0.01, 0.99, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)
        ax.set_title("Graph-Fallback (erste 25 Kanten)")
        return fig

app = App(app_ui, server)
