# analysis.py
# Helper functions used by main.py and the Streamlit app.
# The notebook does the same steps inline so it can be read top to bottom.

import re
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")  # no popup windows, we just want the figure objects
import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
analyzer = SentimentIntensityAnalyzer()


def load_posts(path_or_buffer):
    """Read the CSV. Only the 'text' column is required, 'date' is optional."""
    df = pd.read_csv(path_or_buffer)
    if "text" not in df.columns:
        raise ValueError("CSV needs a 'text' column")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def extract_hashtags(text):
    # lowercase so #AI and #ai count as the same tag
    return [t.lower() for t in HASHTAG_RE.findall(text)]


def add_features(df):
    """Adds the hashtag list and a VADER sentiment score for every post."""
    df = df.copy()
    df["hashtags"] = df["text"].apply(extract_hashtags)
    # remove the hashtags before scoring so only the actual sentence is judged
    no_tags = df["text"].apply(lambda t: HASHTAG_RE.sub("", t))
    df["sentiment"] = no_tags.apply(lambda t: analyzer.polarity_scores(t)["compound"])
    # 0.05 / -0.05 are the cutoffs the VADER authors suggest
    df["label"] = pd.cut(df["sentiment"], [-1.01, -0.05, 0.05, 1.01],
                         labels=["Negative", "Neutral", "Positive"])
    return df


def hashtag_frequency(df, top_n=15):
    counts = Counter(t for tags in df["hashtags"] for t in tags)
    return pd.DataFrame(counts.most_common(top_n), columns=["hashtag", "count"])


def hashtag_sentiment(df, top_n=15):
    # one row per (post, hashtag) so a post with 2 tags counts for both
    exploded = df.explode("hashtags").dropna(subset=["hashtags"])
    g = exploded.groupby("hashtags")["sentiment"].agg(["mean", "count"])
    g = g.sort_values("count", ascending=False).head(top_n).reset_index()
    return g.rename(columns={"hashtags": "hashtag", "mean": "avg_sentiment"})


def hashtag_trend(df, tags):
    exploded = df.explode("hashtags").dropna(subset=["hashtags", "date"])
    exploded = exploded[exploded["hashtags"].isin(tags)].copy()
    exploded["day"] = exploded["date"].dt.date
    return exploded.groupby(["day", "hashtags"]).size().unstack(fill_value=0)


def cooccurrence(df, top_n=10):
    """Which hashtags show up together in the same post most often."""
    pairs = Counter()
    for tags in df["hashtags"]:
        for a, b in combinations(sorted(set(tags)), 2):
            pairs[(a, b)] += 1
    rows = [(f"#{a} + #{b}", c) for (a, b), c in pairs.most_common(top_n)]
    return pd.DataFrame(rows, columns=["pair", "count"])


# ---- plotting ----

def plot_bar(freq):
    fig, ax = plt.subplots(figsize=(9, 5))
    d = freq.iloc[::-1]  # reversed so the biggest bar ends up on top
    ax.barh("#" + d["hashtag"], d["count"], color="#4C78A8")
    for y, v in enumerate(d["count"]):
        ax.text(v, y, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("Number of posts")
    ax.set_title("Top hashtags by frequency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_wordcloud(df):
    counts = Counter(t for tags in df["hashtags"] for t in tags)
    fig, ax = plt.subplots(figsize=(9, 5))
    if counts:
        wc = WordCloud(width=900, height=500, background_color="white")
        ax.imshow(wc.generate_from_frequencies(counts), interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Hashtag word cloud")
    fig.tight_layout()
    return fig


def plot_sentiment(sent):
    fig, ax = plt.subplots(figsize=(9, 5))
    d = sent.iloc[::-1]
    colors = ["#54A24B" if v > 0.05 else "#E45756" if v < -0.05 else "#9D9D9D"
              for v in d["avg_sentiment"]]
    ax.barh("#" + d["hashtag"], d["avg_sentiment"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Average VADER compound score")
    ax.set_title("Average sentiment per hashtag")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_trend(trend):
    fig, ax = plt.subplots(figsize=(9, 5))
    trend.plot(ax=ax, marker="o", ms=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Posts per day")
    ax.set_title("Hashtag usage over time")
    ax.legend(title="Hashtag", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
