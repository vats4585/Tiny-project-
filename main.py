# main.py
# Runs the whole analysis from the command line and saves the charts.
# Usage: python main.py --data data/posts.csv --top 10

import argparse
import os

import analysis as an


def main():
    p = argparse.ArgumentParser(description="Hashtag Frequency Visualization System")
    p.add_argument("--data", default="data/posts.csv", help="path to the posts CSV")
    p.add_argument("--top", type=int, default=15, help="how many hashtags to show")
    p.add_argument("--out", default="outputs", help="folder for the charts")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    df = an.add_features(an.load_posts(a.data))
    freq = an.hashtag_frequency(df, a.top)
    sent = an.hashtag_sentiment(df, a.top)

    print(f"Posts analysed: {len(df)}")
    print("\nTop hashtags:\n", freq.to_string(index=False))
    print("\nHashtags that appear together most:\n", an.cooccurrence(df).to_string(index=False))

    an.plot_bar(freq).savefig(f"{a.out}/top_hashtags.png", dpi=150)
    an.plot_wordcloud(df).savefig(f"{a.out}/wordcloud.png", dpi=150)
    an.plot_sentiment(sent).savefig(f"{a.out}/hashtag_sentiment.png", dpi=150)
    if "date" in df.columns:
        top5 = list(freq["hashtag"].head(5))
        an.plot_trend(an.hashtag_trend(df, top5)).savefig(f"{a.out}/trend.png", dpi=150)
    freq.to_csv(f"{a.out}/hashtag_frequency.csv", index=False)
    print(f"\nDone. Charts are in the '{a.out}' folder.")


if __name__ == "__main__":
    main()
