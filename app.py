# app.py
# Small Streamlit dashboard on top of analysis.py.
# Run with: python -m streamlit run app.py
import streamlit as st

import analysis as an

st.set_page_config(page_title="Hashtag Frequency Visualization", layout="wide")
st.title("Hashtag Frequency Visualization System")
st.caption("Social Media Analytics and Sentiment Analysis - Tiny Project (Master's in ML and AI)")

uploaded = st.sidebar.file_uploader("Upload your own CSV (needs a text column, date is optional)", type="csv")
top_n = st.sidebar.slider("Top N hashtags", 5, 30, 15)

df = an.add_features(an.load_posts(uploaded or "data/posts.csv"))
freq = an.hashtag_frequency(df, top_n)
sent = an.hashtag_sentiment(df, top_n)

c1, c2, c3 = st.columns(3)
c1.metric("Posts", len(df))
c2.metric("Unique hashtags", len({t for tags in df["hashtags"] for t in tags}))
c3.metric("Avg sentiment", f"{df['sentiment'].mean():.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["Frequency", "Word Cloud", "Sentiment", "Trends"])
with tab1:
    l, r = st.columns([2, 1])
    l.pyplot(an.plot_bar(freq))
    r.dataframe(freq, hide_index=True)
    st.subheader("Co-occurring hashtags")
    st.dataframe(an.cooccurrence(df), hide_index=True)
with tab2:
    st.pyplot(an.plot_wordcloud(df))
with tab3:
    st.pyplot(an.plot_sentiment(sent))
    st.bar_chart(df["label"].value_counts())
with tab4:
    if "date" in df.columns:
        picks = st.multiselect("Hashtags", list(freq["hashtag"]), default=list(freq["hashtag"].head(5)))
        if picks:
            st.pyplot(an.plot_trend(an.hashtag_trend(df, picks)))
    else:
        st.info("No 'date' column in data.")
