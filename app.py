import streamlit as st
import re
import string
import pandas as pd
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

# ========== NLTK Setup ==========
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ========== Streamlit Page Setup ==========
st.set_page_config(page_title="Hindustan Foods Report Analyzer", layout="wide")

st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .header-logo {
        width: 90px;
    }
    .header-text {
        text-align: left;
    }
    .header-title {
        font-size: 2rem;
        color: #003366;
        font-weight: bold;
        margin-bottom: 5;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #00509E;
        margin-top: 0.1rem;
    }
    .block-container {
        padding-top: 3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown("""
<div class="header-container">
    <img class="header-logo" src="https://i0.wp.com/mysuruinfrahub.com/wp-content/uploads/2022/11/HFL.jpg">
    <div class="header-text">
        <div class="header-title">Hindustan Foods Limited – Report Analyzer</div>
        <div class="header-subtitle">NLP Dashboard: Sentiment • Word Cloud • Topics • TF-IDF</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ========== Helper Functions ==========
def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pages_data = []
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        pages_data.append({'page': page_num + 1, 'text': text})
    return pd.DataFrame(pages_data)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        if not isinstance(row.get('text'), str):
            continue
        for sent in sent_tokenize(row['text']):
            blob = TextBlob(sent)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            records.append({
                "sentence": sent,
                "polarity": polarity,
                "sentiment": sentiment
            })
    return pd.DataFrame(records)

def show_frequent_words(text_series: pd.Series, top_n=20) -> pd.DataFrame:
    word_list = " ".join(text_series.astype(str)).split()
    counts = Counter(word_list)
    common = counts.most_common(top_n)
    return pd.DataFrame(common, columns=["Word", "Frequency"])

def generate_wordcloud(text_series: pd.Series):
    full = " ".join(text_series.astype(str))
    if not full.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(full)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def create_tfidf_matrix(text_series: pd.Series):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words="english", ngram_range=(1,2))
    matrix = vectorizer.fit_transform(text_series.astype(str))
    return matrix, vectorizer

def topic_modeling(tfidf_matrix, vectorizer, num_topics=10) -> pd.DataFrame:
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method="online")
    lda.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[:-11:-1]
        top_words = ", ".join(feature_names[i] for i in top_idx)
        weight = round(comp[top_idx].sum(), 2)
        topics.append({"Topic": f"Topic {idx+1}", "Top Words": top_words, "Weight": weight})
    return pd.DataFrame(topics)


# ========== Two-column Layout ==========
col_upload, col_main = st.columns([1.2, 4], gap="large")

# ---- Left Column: Upload ----
with col_upload:
    st.markdown("### 📂 Upload Annual Report")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    st.markdown("---")

# ---- Right Column: Main Dashboard ----
with col_main:
    if uploaded_file is None:
        st.info("👈 Please upload a PDF report to begin the analysis.")
        st.stop()

    with st.spinner("Reading and processing PDF..."):
        df_raw = read_pdf(uploaded_file)
    df_raw["cleaned_text"] = df_raw["text"].apply(preprocess_text)

    tabs = st.tabs([
        "1. Sentiment",
        "2. Frequent Words & Word Cloud",
        "3. TF-IDF Preview",
        "4. Topic Modeling"
    ])

    # --- Tab 1: Sentiment ---
    with tabs[0]:
        st.header("Sentiment Analysis")
        sentiment_df = analyze_sentiment(df_raw)
        if sentiment_df.empty:
            st.warning("No sentences found for sentiment analysis.")
        else:
            counts = sentiment_df["sentiment"].value_counts()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Sentiment Bar Chart")
                st.bar_chart(counts)
            with col2:
                st.subheader("Sentiment Pie Chart (with Legend)")
                fig, ax = plt.subplots(figsize=(3, 3))
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=None,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#2ecc71', '#e74c3c', '#f1c40f'],
                    textprops={'fontsize': 8}
                )
                ax.legend(
                    wedges,
                    [f"{label} ({val / counts.sum() * 100:.1f}%)" for label, val in zip(counts.index, counts)],
                    title="Sentiment",
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    fontsize=8
                )
                ax.axis("equal")
                st.pyplot(fig)
            st.subheader("Sample Sentences with Sentiment")
            st.dataframe(sentiment_df.head(10))
            st.download_button(
                "Download Sentiment CSV",
                data=sentiment_df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

    # --- Tab 2: Frequent Words & Word Cloud ---
    with tabs[1]:
        st.header("Most Frequent Words & Word Cloud")
        freq_df = show_frequent_words(df_raw["cleaned_text"])
        if freq_df.empty:
            st.warning("No words to show.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Frequent Words")
                st.dataframe(freq_df)
                st.bar_chart(freq_df.set_index("Word"))
            with col2:
                st.subheader("Word Cloud")
                wc_fig = generate_wordcloud(df_raw["cleaned_text"])
                if wc_fig:
                    st.pyplot(wc_fig)

    # --- Tab 3: TF-IDF Preview ---
    with tabs[2]:
        st.header("TF-IDF Matrix (Preview)")
        tfidf_matrix, vectorizer = create_tfidf_matrix(df_raw["cleaned_text"])
        st.write(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        names = vectorizer.get_feature_names_out()[:10]
        preview = pd.DataFrame(tfidf_matrix[:5, :10].toarray(), columns=names, index=[f"Page {i+1}" for i in range(5)])
        st.subheader("Sample TF-IDF Values")
        st.dataframe(preview)

    # --- Tab 4: Topic Modeling ---
    with tabs[3]:
        st.header("Topic Modeling with LDA (Top 10 Topics)")
        topics_df = topic_modeling(tfidf_matrix, vectorizer, num_topics=10)
        if topics_df.empty:
            st.warning("No topics generated.")
        else:
            st.dataframe(topics_df)
            st.subheader("Topic Weight Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(topics_df["Topic"], topics_df["Weight"], color="#3498db")
            ax.set_ylabel("Weight")
            ax.set_title("Topic Weights")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.subheader("Topic Summaries")
            for _, row in topics_df.iterrows():
                st.write(f"• **{row['Topic']}**: {row['Top Words']}")

