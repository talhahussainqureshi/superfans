"""Super-Fan analysis pipeline for REWE and EDEKA store reviews.

Reads the raw review exports in data/raw, scores each review for how strongly
it reads as a "Super-Fan" review (rating + sentiment), and renders a static
HTML/Chart.js dashboard summarizing the results per brand.

Run from anywhere; all paths are resolved relative to the project root
(one level above this file), so `python src/superfans_analysis.py` and
`python superfans_analysis.py` from inside src/ both work.
"""

from __future__ import annotations

import http.server
import json
import logging
import re
import socketserver
import threading
import warnings
import webbrowser
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file only, no GUI backend needed
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from langdetect import DetectorFactory, detect
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("superfans")

# ── NLTK bootstrap ──────────────────────────────────────────────
for corpus in [
    "stopwords", "punkt", "wordnet", "sentiwordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]:
    try:
        nltk.data.find(f"taggers/{corpus}" if corpus.startswith("averaged") else corpus)
    except LookupError:
        nltk.download(corpus, quiet=True)

STOP = set(stopwords.words("english"))
SID = SentimentIntensityAnalyzer()
DetectorFactory.seed = 0
warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths & constants ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
DASH_DIR = OUTPUT_DIR / "dashboard"
RESULTS_DIR = BASE_DIR / "results"  # tracked in git: static charts for viewing on GitHub

RAW_REWE = DATA_RAW / "rewe_store_reviews_main_topic.xlsx"
CLEAN_REWE = DATA_PROCESSED / "rewe_en.xlsx"
EDEKA_FILE = DATA_RAW / "edeka_store_reviews_main_topic.xlsx"
CSV_OPTS = dict(index=True, float_format="%.3f", decimal=",")
PORT = 8000

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DASH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────
_lang_cache: dict[str, str] = {}


def fast_detect(txt: str) -> str:
    if txt in _lang_cache:
        return _lang_cache[txt]
    try:
        lang = detect(txt)
    except Exception:
        lang = "unk"
    _lang_cache[txt] = lang
    return lang


def clean(txt: str) -> str:
    txt = re.sub(r"[^\w\s]", "", txt.lower())
    return " ".join(t for t in txt.split() if t not in STOP)


def pick(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")


# ── ensure English-only REWE ────────────────────────────────────────
def ensure_rewe_en() -> None:
    """Cache an English-column-only copy of the raw REWE export.

    The raw file has duplicate columns per language (e.g. review_de, review_en);
    this keeps whichever column in each group is most often detected as English.
    """
    if CLEAN_REWE.exists():
        return
    raw = pd.read_excel(RAW_REWE)

    def stem(c: str) -> str:
        return c.lower().replace("-", "_").split("_")[0]

    groups: dict[str, list[str]] = {}
    for col in raw.columns:
        groups.setdefault(stem(col), []).append(col)

    chosen = {}
    for s, cols in groups.items():
        best, best_ratio = None, -1.0
        for c in cols:
            sample = raw[c].dropna().head(300)
            ratio = sum(fast_detect(t) == "en" for t in sample) / len(sample) if len(sample) else 0
            if ratio > best_ratio:
                best_ratio, best = ratio, c
        chosen[s] = best

    raw[[chosen[k] for k in sorted(chosen)]].to_excel(CLEAN_REWE, index=False)
    log.info("Cleaned REWE columns -> %s", CLEAN_REWE.name)


# ── per-brand processing ─────────────────────────────────────────
def process_brand(brand: str, df: pd.DataFrame) -> pd.DataFrame:
    rev = pick(df, "review_text", "review", "text", "comment")
    rat = pick(df, "rating", "stars", "score")
    theme = pick(df, "theme", "category", "topic", "aspect")
    city = pick(df, "city", "town", "location", "store_city", "filiale")

    # optional sentiment label column
    if "sentiment" in df.columns and "sentiment_cat" not in df.columns:
        df = df.rename(columns={"sentiment": "sentiment_cat"})

    df = df.rename(columns={rev: "review", rat: "rating", theme: "theme", city: "city"})
    df = df[pd.to_numeric(df["rating"], errors="coerce") >= 4].copy()
    df["lang"] = df["review"].map(fast_detect)
    df = df[df["lang"].isin({"en", "de"})]

    # collect sentiment labels per review, if the source provided any
    sent_map = {}
    if "sentiment_cat" in df.columns:
        sent_map = df.groupby("review")["sentiment_cat"].apply(list).to_dict()

    grouped = df.groupby("review", as_index=False).agg({
        "rating": "first",
        "city": "first",
        "theme": lambda s: ", ".join(sorted({t.strip() for t in s if pd.notna(t)})),
    })

    grouped["clean"] = grouped["review"].apply(clean)
    grouped["vader"] = grouped["clean"].apply(lambda t: SID.polarity_scores(t)["compound"])
    grouped["vader_n"] = (grouped["vader"] + 1) / 2

    def pos_share(row: pd.Series) -> float:
        """Share of positive sentiment labels for this review, falling back to VADER."""
        labels = sent_map.get(row["review"], [])
        if labels:
            pos = sum(str(l).lower() == "positive" for l in labels)
            neu = sum(str(l).lower() == "neutral" for l in labels)
            neg = sum(str(l).lower() == "negative" for l in labels)
            total = pos + neu + neg
            if total:
                return pos / total
        return row["vader_n"]

    grouped["pos_share"] = grouped.apply(pos_share, axis=1)

    grouped["rating_n"] = grouped["rating"] / 5
    grouped["sf_score"] = (
        0.40 * grouped["rating_n"]
        + 0.35 * grouped["pos_share"]
        + 0.25 * grouped["vader_n"]
    )
    grouped["is_superfan"] = grouped["sf_score"] >= 0.75
    grouped["brand"] = brand

    export_cols = ["review", "city", "theme", "rating", "vader", "pos_share", "sf_score", "is_superfan"]
    grouped[export_cols].to_csv(DATA_PROCESSED / f"{brand}_4plus_reviews.csv", **CSV_OPTS)

    # dedicated results file: identified Super-Fans only, strongest first
    results_cols = ["review", "city", "theme", "rating", "vader", "pos_share", "sf_score"]
    superfans = (
        grouped.loc[grouped["is_superfan"], results_cols]
        .sort_values("sf_score", ascending=False)
    )
    superfans.to_csv(DATA_PROCESSED / f"{brand}_superfans.csv", **CSV_OPTS)

    sf, other = grouped["is_superfan"].value_counts().reindex([True, False], fill_value=0)
    log.info("%s: %d Super-Fans, %d Others saved", brand, sf, other)
    log.info("%s: Super-Fan results -> %s_superfans.csv", brand, brand)
    return grouped


# ── theme / lift helpers ─────────────────────────────────────────
def theme_counter(df: pd.DataFrame, brand: str, is_superfan: bool) -> Counter:
    subset = df[(df["brand"] == brand) & (df["is_superfan"] == is_superfan)]["theme"].dropna()
    return Counter(t for row in subset for t in map(str.strip, row.split(",")) if t)


def lift_series(df: pd.DataFrame, brand: str, top_n: int = 10) -> pd.Series:
    """Themes that appear disproportionately more often among Super-Fans than Others."""
    sf_counts = theme_counter(df, brand, True)
    other_counts = theme_counter(df, brand, False)
    sf_total = sum(sf_counts.values()) or 1
    other_total = sum(other_counts.values()) or 1
    lift = (pd.Series(sf_counts) / sf_total) - (pd.Series(other_counts) / other_total)
    lift = lift[lift > 0].dropna()
    return lift.reindex(lift.abs().sort_values(ascending=False).index).head(top_n)


def top_predictive_words(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Words whose presence best predicts Super-Fan status, via TF-IDF + logistic regression.

    Deliberately uses review *text* rather than `rating`/`vader`/`pos_share` as
    features: those three columns define `sf_score` directly, so a model trained
    on them would just recover its own label instead of producing a real insight.
    """
    if df["is_superfan"].nunique() < 2:
        return pd.Series(dtype=float)

    vectorizer = TfidfVectorizer(max_features=500, min_df=5)
    features = vectorizer.fit_transform(df["clean"])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(features, df["is_superfan"])

    coef = pd.Series(clf.coef_[0], index=vectorizer.get_feature_names_out())
    return coef.reindex(coef.abs().sort_values(ascending=False).index).head(top_n).round(2)


# ── static chart export (tracked in git, visible on GitHub) ───────
def export_static_charts(all_df: pd.DataFrame, summary: pd.DataFrame, top_words: pd.Series) -> None:
    """Render a handful of PNG charts to results/ so findings are visible on
    GitHub without cloning and running the dashboard."""
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    # 01: Super-Fans vs Others per brand
    counts = (
        all_df.groupby(["brand", "is_superfan"]).size().unstack(fill_value=0)
        .rename(columns={True: "Super-Fans", False: "Others"})
    )
    ax = counts.plot(kind="bar", color=["#4e79a7", "#f28e2b"], figsize=(6, 4))
    ax.set_title("Super-Fans vs Others by Brand")
    ax.set_xlabel("")
    ax.set_ylabel("Reviews (rating >= 4)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_superfan_counts.png")
    plt.close()

    # 02: top lift themes, REWE and EDEKA side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, brand in zip(axes, ["rewe", "edeka"]):
        lift = (lift_series(all_df, brand, top_n=8) * 100).sort_values()
        ax.barh(lift.index, lift.values, color="#59a14f")
        ax.set_title(f"{brand.capitalize()}: themes over-represented among Super-Fans")
        ax.set_xlabel("Lift vs Others (percentage points)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_top_lift_themes.png")
    plt.close()

    # 03: top predictive words (TF-IDF logistic regression coefficients)
    if not top_words.empty:
        ordered = top_words.sort_values()
        colors = ["#e15759" if v < 0 else "#59a14f" for v in ordered.values]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(ordered.index, ordered.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Words most predictive of Super-Fan status")
        ax.set_xlabel("Logistic regression coefficient (text-only model)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "03_top_predictive_words.png")
        plt.close()

    # 04: mean rating / sentiment, Super-Fans vs Others
    # Plotted on normalized 0-1 columns (rating_n, vader_n, pos_share) rather than
    # raw `rating` (0-5) and `vader` (-1 to 1) — mixing those scales on one axis
    # makes the sentiment bars nearly invisible next to the rating bar.
    normalized = (
        all_df.groupby("is_superfan")[["rating_n", "vader_n", "pos_share"]]
        .mean()
        .rename(index={True: "Super-Fans", False: "Others"})
        .rename(columns={"rating_n": "rating (0-1)", "vader_n": "sentiment (0-1)", "pos_share": "pos_share"})
        .round(3)
    )
    ax = normalized.plot(kind="bar", figsize=(6, 4), color=["#4e79a7", "#f28e2b", "#76b7b2"])
    ax.set_title("Mean rating / sentiment: Super-Fans vs Others (normalized 0-1)")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_mean_comparison.png")
    plt.close()

    log.info("Static charts -> %s", RESULTS_DIR)


# ── Chart.js helper ──────────────────────────────────────────────
def chart_js(
    cid: str,
    label: str,
    data: list[tuple[str, float]],
    *,
    horiz: bool = False,
    ctype: str = "bar",
    color: str | list[str] = "#4e79a7",
    legend: bool = False,
) -> str:
    labels = [l for l, _ in data] or ["(none)"]
    values = [v for _, v in data] or [0]

    color_js = f"'{color}'" if isinstance(color, str) else json.dumps(color)
    orient = "'y'" if horiz and ctype == "bar" else "'x'"
    legend_js = str(legend).lower()

    return f"""
new Chart(document.getElementById('{cid}'), {{
  type: '{ctype}',
  data: {{
    labels: {labels},
    datasets: [{{
      label: '{label}',
      data: {values},
      backgroundColor: {color_js}
    }}]
  }},
  options: {{
    indexAxis: {orient},
    plugins: {{
      legend: {{
        display: {legend_js}
      }}
    }}
  }}
}});"""


# ── main ──────────────────────────────────────────────────────────
def main() -> None:
    ensure_rewe_en()

    frames = {
        "rewe": pd.read_excel(CLEAN_REWE),
        "edeka": pd.read_excel(EDEKA_FILE),
    }
    all_df = pd.concat([process_brand(b, d) for b, d in frames.items()], ignore_index=True)

    summary = (
        all_df.groupby("is_superfan")[["rating", "vader", "pos_share"]]
        .mean()
        .rename(index={True: "Super-Fans", False: "Others"})
        .round(2)
    )

    top_words = top_predictive_words(all_df)
    export_static_charts(all_df, summary, top_words)

    # ── HTML skeleton ──────────────────────────────────────────────
    html = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Super-Fan Dashboard</title>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<style>
 body{font-family:Arial;margin:2rem}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));grid-gap:2rem}
 canvas{width:100%;height:auto}
 table{border-collapse:collapse;margin-top:1rem}
 th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}
 th:first-child,td:first-child{text-align:left}
</style></head><body>
<h1>Super-Fan Overview</h1>
"""

    palette = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab",
    ]

    for brand in ["rewe", "edeka"]:
        cap = brand.capitalize()
        top_sf = theme_counter(all_df, brand, True).most_common(10)
        top_other = theme_counter(all_df, brand, False).most_common(10)
        top_lift = [(k, round(v * 100, 1)) for k, v in lift_series(all_df, brand).items()]

        city_counts = (
            all_df[(all_df["brand"] == brand) & (all_df["is_superfan"])]
            .groupby("city").size().sort_values(ascending=False).head(10).items()
        )

        html += f"""
<h2>{cap} - Themes</h2>
<div class='grid'>
  <div><h3>Super-Fans</h3><canvas id='{brand}_sf'></canvas></div>
  <div><h3>Others</h3><canvas id='{brand}_oth'></canvas></div>
  <div><h3>Positive Lift</h3><canvas id='{brand}_lift'></canvas></div>
  <div><h3>Super-Fans by City</h3><canvas id='{brand}_city'></canvas></div>
</div>
<script>
{chart_js(f'{brand}_sf', 'Super-Fans', top_sf, horiz=True, color='#4e79a7')}
{chart_js(f'{brand}_oth', 'Others', top_other, horiz=True, color='#f28e2b')}
{chart_js(f'{brand}_lift', 'Lift (+pp)', top_lift, ctype='doughnut',
          color=palette[:len(top_lift)], legend=True)}
{chart_js(f'{brand}_city', 'SF count', list(city_counts), horiz=True, color='#af7aa1')}
</script>
"""

    html += f"""
<h2>Mean Comparison</h2>
{summary.to_html(border=0)}
<h2>Top Predictive Words (Super-Fans vs Others)</h2>
<p style='max-width:600px;color:#555'>Words whose presence in a review best predicts
Super-Fan status, from a TF-IDF + logistic regression model fit on review text
(not on the rating/sentiment inputs that define the Super-Fan score itself).</p>
{top_words.to_frame('coef').to_html(border=0) if not top_words.empty else '<p>(skipped - not enough data)</p>'}
</body></html>"""

    (DASH_DIR / "index.html").write_text(html, encoding="utf-8")

    handler = http.server.SimpleHTTPRequestHandler
    server = socketserver.TCPServer(
        ("", PORT),
        lambda *a, dir=str(DASH_DIR), **kw: handler(*a, directory=dir, **kw),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open_new_tab(f"http://localhost:{PORT}/index.html")
    log.info("Dashboard available at http://localhost:%d/ (press Enter to stop)", PORT)
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    server.shutdown()


# ── entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
