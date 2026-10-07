# Hashtag Frequency Visualization System

Tiny project for **Social Media Analytics and Sentiment Analysis**
M.Sc. Machine Learning and AI

**Author:** Harit Mengar (Enrollment No. 2305101010055)

## What this project does

People use hashtags to tag the topic of a post, but with thousands of posts it is impossible to see which topics are popular just by reading. This project takes a CSV of posts, pulls out all the hashtags, counts them, checks if the posts around each hashtag are positive or negative, and draws charts to show it.

## What you get

- Top hashtags bar chart and a word cloud
- Average sentiment for each hashtag (using VADER)
- Hashtags that appear together in the same post
- Daily trend of the most used hashtags
- A Jupyter notebook that walks through everything step by step
- A small Streamlit dashboard where you can upload your own CSV

## Files

| File | What it is |
|---|---|
| `hashtag_analysis.ipynb` | Main notebook with the full analysis, charts and observations |
| `analysis.py` | Functions used by the script and the dashboard |
| `main.py` | Runs the analysis from the terminal and saves charts to `outputs/` |
| `app.py` | Streamlit dashboard |
| `generate_sample_data.py` | Makes the sample dataset |
| `data/posts.csv` | 500 synthetic posts |
| `outputs/` | Saved charts |

## How to run

I used a virtual environment so nothing clashes with other packages.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Then pick one:

```bash
jupyter notebook hashtag_analysis.ipynb     # the notebook
python main.py                              # charts saved to outputs/
python -m streamlit run app.py              # dashboard
```

If `data/posts.csv` is missing, run `python generate_sample_data.py` first.

## Using other data

Any CSV with a `text` column works. A `date` column is optional and is only needed for the trend chart.

## Method in short

1. Extract hashtags with the regex `#(\w+)` and lowercase them
2. Count them with `collections.Counter`
3. Remove the hashtags from the text, then score the sentence with VADER (compound score from -1 to +1)
4. Use `explode()` so a post with two hashtags counts for both when averaging sentiment
5. Plot with Matplotlib and WordCloud

## Limitations

- The data is synthetic (the Twitter/X API is paid now), so the results only prove the pipeline works
- VADER does not handle sarcasm or Hindi-English mixed text well
- Similar hashtags like `#ML` and `#MachineLearning` are counted separately

## Future work

- Run it on real data from Reddit or Mastodon
- Compare VADER with a transformer sentiment model
- Merge similar hashtags automatically
