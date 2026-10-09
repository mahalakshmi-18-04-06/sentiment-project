import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocess import clean_text
from scraper import NEWS_SOURCES, scrape_multiple
from bert_analyzer import BertSentimentAnalyzer

# ---------- Auto-download NLTK data (needed for cloud) ----------
import nltk
for pkg, path in [('stopwords', 'corpora/stopwords'),
                  ('punkt', 'tokenizers/punkt')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SentimentScope — Real-Time News Sentiment",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 400px); }
    #MainMenu, footer, header { visibility: hidden; }

    .brand-bar {
        display: flex; align-items: center; gap: 12px;
        padding: 6px 0 18px 0;
        border-bottom: 1px solid #e5e9f0; margin-bottom: 24px;
    }
    .brand-logo {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, #6366f1, #2a5298);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 22px; font-weight: 700;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }
    .brand-name {
        font-size: 1.4rem; font-weight: 800;
        color: #1a1f36; letter-spacing: -0.5px; margin: 0;
    }
    .brand-tagline { font-size: 0.85rem; color: #6b7280; margin: 0; }
    .brand-badge {
        margin-left: auto; background: #ecfdf5; color: #047857;
        padding: 6px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; border: 1px solid #a7f3d0;
    }

    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6366f1 100%);
        padding: 40px 44px; border-radius: 18px; color: white;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.25);
        position: relative; overflow: hidden;
    }
    .hero h1 {
        color: white !important; margin: 0 0 12px 0;
        font-size: 2.1rem; font-weight: 800;
        letter-spacing: -0.8px; line-height: 1.2;
    }
    .hero p { color: #cdd9f0 !important; margin: 0; font-size: 1.05rem; max-width: 700px; }
    .hero-badges { margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
    .hero-badge {
        background: rgba(255,255,255,0.15); backdrop-filter: blur(8px);
        padding: 6px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 500;
        border: 1px solid rgba(255,255,255,0.25);
    }

    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #1a1f36;
        margin-top: 28px; margin-bottom: 14px;
        display: flex; align-items: center; gap: 8px;
    }
    .section-header::before {
        content: ""; width: 4px; height: 20px;
        background: linear-gradient(180deg, #6366f1, #2a5298);
        border-radius: 2px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff; padding: 20px 22px;
        border-radius: 14px; border: 1px solid #eaeef5;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(30, 60, 114, 0.1);
        border-color: #c7d2fe;
    }
    div[data-testid="stMetric"] label {
        font-weight: 600; color: #6b7280 !important;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1f36; font-weight: 800; font-size: 1.8rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        border-right: 1px solid #eaeef5;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #1a1f36; font-weight: 700;
    }
    section[data-testid="stSidebar"] hr { margin: 16px 0; border-color: #eaeef5; }

    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid #eaeef5; margin-bottom: 16px;
    }
    .sidebar-brand-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #6366f1, #2a5298);
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 18px;
    }
    .sidebar-brand-text {
        font-weight: 800; color: #1a1f36;
        font-size: 1.05rem; letter-spacing: -0.3px;
    }
    .sidebar-brand-sub { font-size: 0.72rem; color: #9ca3af; font-weight: 500; }

    .info-card {
        background: linear-gradient(135deg, #f0f4ff, #e0e7ff);
        border: 1px solid #c7d2fe; border-radius: 12px;
        padding: 14px 16px; margin-top: 12px;
    }
    .info-card-title {
        font-size: 0.75rem; font-weight: 700; color: #4338ca;
        text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
    }
    .info-card-row {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; color: #4b5563; padding: 4px 0;
    }
    .info-card-row b { color: #1a1f36; font-weight: 700; }

    span[data-baseweb="tag"] {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #2a5298 100%);
        color: white; border: none; border-radius: 10px;
        padding: 12px 20px; font-weight: 700; font-size: 0.9rem;
        letter-spacing: 0.2px; transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        color: white;
    }

    button[data-baseweb="tab"] {
        font-size: 0.95rem; font-weight: 600;
        color: #6b7280; padding: 12px 20px;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: #4338ca; }
    div[data-baseweb="tab-highlight"] { background-color: #6366f1; }
    div[data-baseweb="tab-border"] { background-color: #eaeef5; }

    div[data-testid="stDataFrame"] {
        border: 1px solid #eaeef5; border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stAlert"] { border-radius: 12px; border-left-width: 4px; }
    div[data-testid="stAlert"] p { font-size: 0.88rem; }

    .stDownloadButton > button {
        background: #ffffff; color: #4338ca;
        border: 2px solid #c7d2fe; border-radius: 10px;
        font-weight: 700; transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        background: #6366f1; color: white; border-color: #6366f1;
    }

    .onboard-card {
        background: #ffffff; border: 1px solid #eaeef5;
        border-radius: 14px; padding: 22px 24px; height: 100%;
        transition: all 0.2s ease;
    }
    .onboard-card:hover {
        border-color: #c7d2fe;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.08);
        transform: translateY(-2px);
    }
    .onboard-icon {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; margin-bottom: 12px;
    }
    .onboard-title {
        font-size: 1rem; font-weight: 700;
        color: #1a1f36; margin-bottom: 6px;
    }
    .onboard-desc { font-size: 0.85rem; color: #6b7280; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/sentiment_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return model, tfidf

@st.cache_resource
def load_bert():
    return BertSentimentAnalyzer()

model, tfidf = load_artifacts()
bert = load_bert()

# =========================================================
# BRAND BAR
# =========================================================
st.markdown("""
<div class="brand-bar">
    <div class="brand-logo">🔮</div>
    <div>
        <div class="brand-name">SentimentScope</div>
        <div class="brand-tagline">Real-Time News Sentiment Intelligence</div>
    </div>
    <div class="brand-badge">● Live</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Real-Time Sentiment Analysis with Web Scraping</h1>
    <p>Scrape live news headlines from multiple sources, classify their sentiment instantly, and uncover trends in seconds — powered by machine learning.</p>
    <div class="hero-badges">
        <span class="hero-badge">🧠 ML-Powered</span>
        <span class="hero-badge">🌐 Multi-Source</span>
        <span class="hero-badge">⚡ Real-Time</span>
        <span class="hero-badge">📊 Visual Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-icon">🔮</div>
    <div>
        <div class="sidebar-brand-text">SentimentScope</div>
        <div class="sidebar-brand-sub">v1.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### Configuration")

selected_sources = st.sidebar.multiselect(
    "Select news sources:",
    options=list(NEWS_SOURCES.keys()),
    default=["BBC World", "The Hindu"],
    help="Pick one or more sources to scrape headlines from",
)

per_source = st.sidebar.slider(
    "Headlines per source:",
    min_value=5, max_value=20, value=10,
    help="Number of headlines to fetch from each source",
)

st.sidebar.markdown("")
use_bert = st.sidebar.checkbox(
    "Compare with BERT",
    value=False,
    help="Also classify each headline with BERT (slower but more accurate). "
         "Best for comparing where Logistic Regression and BERT disagree.",
)

run = st.sidebar.button("Scrape & Analyze", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Model Info")

st.sidebar.markdown("""
<div class="info-card">
    <div class="info-card-title">Production Model</div>
    <div class="info-card-row"><span>Algorithm</span> <b>Logistic Reg.</b></div>
    <div class="info-card-row"><span>Training size</span> <b>200,000 tweets</b></div>
    <div class="info-card-row"><span>Test accuracy</span> <b>77.89%</b></div>
    <div class="info-card-row"><span>CV score</span> <b>77.20%</b></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Dashboard",
    "🧠 Model Comparison",
    "🔬 Model Information",
    "ℹ️ About"
])

# =========================================================
# TAB 1: LIVE DASHBOARD
# =========================================================
with tab1:
    if run:
        if not selected_sources:
            st.warning("Please select at least one news source from the sidebar.")
        else:
            with st.spinner("Scraping live headlines..."):
                items = scrape_multiple(selected_sources, per_source=per_source)

            if not items:
                st.error("No headlines scraped. Try different sources.")
            else:
                with st.spinner("Analyzing sentiment with Logistic Regression..."):
                    texts = [it["headline"] for it in items]
                    sources_ = [it["source"] for it in items]

                    cleaned = [clean_text(t) for t in texts]
                    vectors = tfidf.transform(cleaned)

                    preds = model.predict(vectors)

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vectors)[:, 1]
                    else:
                        probs = [0.5] * len(preds)

                    labels = ["Positive" if p == 1 else "Negative" for p in preds]
                    confidence = [round(p * 100, 1) for p in probs]

                df = pd.DataFrame({
                    "Headline": texts,
                    "Source": sources_,
                    "Sentiment": labels,
                    "P(Positive)": [f"{c}%" for c in confidence],
                    "_conf_num": confidence,
                    "_prob": probs,
                })

                # ---- Optional BERT comparison on the SAME headlines ----
                if use_bert:
                    with st.spinner(f"Running BERT on {len(texts)} headlines (this may take ~1 minute)..."):
                        bert_results = bert.predict_batch(texts)

                    df["BERT Pred"] = [r["label"] for r in bert_results]
                    df["BERT Conf"] = [f"{r['confidence']}%" for r in bert_results]
                    df["Agree?"] = [
                        "✅" if labels[i] == bert_results[i]["label"] else "❌"
                        for i in range(len(texts))
                    ]

                    agree_count = sum(1 for a in df["Agree?"] if a == "✅")
                    agree_pct = round(agree_count / len(df) * 100, 1)
                    st.info(
                        f"**BERT Comparison Enabled** — LR and BERT agree on "
                        f"**{agree_count}/{len(df)}** headlines ({agree_pct}%). "
                        f"Where they disagree, BERT is usually more accurate on formal news headlines."
                    )

                # ---------- METRICS ----------
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Items", len(df))
                col2.metric("Positive", (df["Sentiment"] == "Positive").sum())
                col3.metric("Negative", (df["Sentiment"] == "Negative").sum())
                col4.metric("Avg P(Positive)", f"{df['_conf_num'].mean():.1f}%")

                # ---------- TABLE ----------
                st.markdown('<div class="section-header">Analyzed Headlines</div>', unsafe_allow_html=True)
                display_df = df.drop(columns=["_conf_num", "_prob"])
                st.dataframe(display_df, use_container_width=True, height=400)

                # ---------- PIE + WORD CLOUD ----------
                st.markdown('<div class="section-header">Visual Insights</div>', unsafe_allow_html=True)
                colA, colB = st.columns(2)

                with colA:
                    st.markdown("**Sentiment Distribution**")
                    counts = df["Sentiment"].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.pie(
                        counts, labels=counts.index,
                        autopct="%1.1f%%",
                        colors=["#10b981", "#ef4444"],
                        startangle=90,
                        textprops={"fontsize": 12, "fontweight": "bold"},
                        wedgeprops={"edgecolor": "white", "linewidth": 2},
                    )
                    ax.axis("equal")
                    st.pyplot(fig)
                    plt.close(fig)

                with colB:
                    st.markdown("**Word Cloud**")
                    all_text = " ".join(cleaned)
                    if all_text.strip():
                        wc = WordCloud(
                            width=800, height=500,
                            background_color="white",
                            colormap="viridis",
                            prefer_horizontal=0.9,
                        ).generate(all_text)
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        ax2.imshow(wc, interpolation="bilinear")
                        ax2.axis("off")
                        st.pyplot(fig2)
                        plt.close(fig2)
                    else:
                        st.info("No text to display.")

                # ---------- TOP HEADLINES ----------
                st.markdown('<div class="section-header">Top Headlines by Confidence</div>', unsafe_allow_html=True)
                colTop1, colTop2 = st.columns(2)

                with colTop1:
                    st.markdown("**🟢 Top 5 Most Positive**")
                    top_pos = df[df["Sentiment"] == "Positive"].nlargest(5, "_conf_num")
                    if len(top_pos):
                        for _, r in top_pos.iterrows():
                            st.success(f"**{r['P(Positive)']}** · {r['Headline'][:90]}")
                    else:
                        st.info("No positive headlines")

                with colTop2:
                    st.markdown("**🔴 Top 5 Most Negative**")
                    top_neg = df[df["Sentiment"] == "Negative"].nsmallest(5, "_conf_num")
                    if len(top_neg):
                        for _, r in top_neg.iterrows():
                            st.error(f"**{r['P(Positive)']}** · {r['Headline'][:90]}")
                    else:
                        st.info("No negative headlines")

                # ---------- SOURCE BREAKDOWN ----------
                st.markdown('<div class="section-header">Sentiment by Source</div>', unsafe_allow_html=True)
                source_stats = df.groupby(["Source", "Sentiment"]).size().unstack(fill_value=0)

                if not source_stats.empty:
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    source_stats.plot(
                        kind="bar", stacked=True,
                        color=["#ef4444", "#10b981"],
                        ax=ax3, edgecolor="white", linewidth=1,
                    )
                    ax3.set_ylabel("Number of Headlines", fontweight="bold")
                    ax3.set_xlabel("News Source", fontweight="bold")
                    ax3.legend(title="Sentiment", frameon=False)
                    ax3.spines["top"].set_visible(False)
                    ax3.spines["right"].set_visible(False)
                    plt.xticks(rotation=15)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)
                else:
                    st.info("No source data available.")

                # ---------- DOWNLOAD ----------
                st.markdown("---")
                csv = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results (CSV)",
                    csv, "sentiment_results.csv", "text/csv",
                )
    else:
        st.markdown('<div class="section-header">Get Started</div>', unsafe_allow_html=True)
        st.markdown("Configure your sources in the sidebar and click **Scrape & Analyze** to run a live analysis.")

        st.markdown("<br>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown("""
            <div class="onboard-card">
                <div class="onboard-icon">🌐</div>
                <div class="onboard-title">Live Scraping</div>
                <div class="onboard-desc">Fetches the latest headlines from multiple news sources in real time.</div>
            </div>
            """, unsafe_allow_html=True)
        with colB:
            st.markdown("""
            <div class="onboard-card">
                <div class="onboard-icon">🧠</div>
                <div class="onboard-title">ML Classification</div>
                <div class="onboard-desc">TF-IDF + Logistic Regression trained on 200,000 tweets for sentiment prediction.</div>
            </div>
            """, unsafe_allow_html=True)
        with colC:
            st.markdown("""
            <div class="onboard-card">
                <div class="onboard-icon">📊</div>
                <div class="onboard-title">Visual Analytics</div>
                <div class="onboard-desc">Interactive charts, word clouds, and per-source sentiment breakdown.</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2: MODEL COMPARISON (Manual text input)
# =========================================================
with tab2:
    st.markdown("### 🧠 Model Comparison: Classical ML vs BERT")
    st.markdown("Type or paste any text to compare predictions from both models side-by-side.")

    st.markdown("---")

    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        sample_text = st.text_area(
            "Enter text to compare (one per line):",
            value=(
                "I love this product! Best purchase ever.\n"
                "This is terrible. Worst experience of my life.\n"
                "Cornell students voice frustration at public hearing over rape case.\n"
                "Israel honours Captain Smit Machchhar's valour with billboards in Mumbai.\n"
                "Breaking the blister pack: Counterfeit medicines network busted\n"
                "PM Modi speaks to Flydubai pilot Captain Smit Machchhar\n"
                "Hundreds of French schools closed after teacher attacked\n"
                "India wins gold medal at Asian Games"
            ),
            height=200,
            key="compare_input"
        )
    with col_input2:
        st.markdown("**Tips:**")
        st.markdown("- One sentence per line")
        st.markdown("- Try news headlines")
        st.markdown("- Try informal tweets")
        run_compare = st.button("Compare", use_container_width=True, key="cmp_btn")

    if run_compare:
        lines = [line.strip() for line in sample_text.split("\n") if line.strip()]

        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            with st.spinner("Running both models..."):
                cleaned = [clean_text(t) for t in lines]
                vectors = tfidf.transform(cleaned)
                lr_preds = model.predict(vectors)
                if hasattr(model, "predict_proba"):
                    lr_probs = model.predict_proba(vectors)[:, 1]
                else:
                    lr_probs = [0.5] * len(lr_preds)

                bert_results = bert.predict_batch(lines)

            rows = []
            for i, text in enumerate(lines):
                lr_label = "Positive" if lr_preds[i] == 1 else "Negative"
                lr_conf = round(lr_probs[i] * 100, 1)
                rows.append({
                    "Text": text[:80] + ("..." if len(text) > 80 else ""),
                    "LR Prediction": lr_label,
                    "LR Conf": f"{lr_conf}%",
                    "BERT Prediction": bert_results[i]["label"],
                    "BERT Conf": f"{bert_results[i]['confidence']}%",
                    "Agree?": "✅" if lr_label == bert_results[i]["label"] else "❌",
                })

            df_compare = pd.DataFrame(rows)
            total = len(df_compare)
            agree = (df_compare["Agree?"] == "✅").sum()
            agree_pct = round(agree / total * 100, 1)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tested", total)
            col2.metric("Models Agree", f"{agree}/{total}")
            col3.metric("Agreement Rate", f"{agree_pct}%")

            st.markdown("---")
            st.markdown("#### Side-by-Side Predictions")
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

            st.markdown("#### Confidence Comparison")
            fig, ax = plt.subplots(figsize=(10, max(3, len(lines) * 0.5)))
            x = range(len(lines))
            width = 0.35

            lr_confs = [float(r["LR Conf"].strip("%")) for r in rows]
            bert_confs = [float(r["BERT Conf"].strip("%")) for r in rows]

            ax.barh([i - width/2 for i in x], lr_confs, width,
                    label="Logistic Regression", color="#6366f1", alpha=0.85)
            ax.barh([i + width/2 for i in x], bert_confs, width,
                    label="BERT", color="#10b981", alpha=0.85)

            ax.set_yticks(list(x))
            ax.set_yticklabels([f"#{i+1}" for i in x])
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 105)
            ax.legend(loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            st.markdown("#### Key Observation")
            st.info(
                f"**Logistic Regression** (trained on informal tweets) and "
                f"**BERT** (context-aware transformer) agree on **{agree_pct}%** of cases.\n\n"
                "Where they **disagree**, BERT is usually more accurate on formal news headlines — "
                "this demonstrates the **domain shift** problem and how transformers solve it."
            )

# =========================================================
# TAB 3: MODEL INFORMATION
# =========================================================
with tab3:
    st.markdown("### Model Performance Comparison")
    st.markdown("Models trained on 200,000 tweets from the Sentiment140 dataset.")

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression ⭐",
            "Linear SVM",
            "Naive Bayes",
            "Random Forest",
            "Gradient Boosting",
            "AdaBoost",
            "Bagging (Decision Tree)",
        ],
        "Accuracy": [0.7789, 0.7730, 0.7642, 0.7614, 0.6902, 0.6362, 0.6327],
        "Category": [
            "Classical (Linear)",
            "Classical (Linear)",
            "Classical (Probabilistic)",
            "Ensemble — Bagging",
            "Ensemble — Boosting",
            "Ensemble — Boosting",
            "Ensemble — Bagging",
        ],
        "Notes": [
            "Best accuracy — production model",
            "Margin-based linear classifier",
            "Fast, strong probabilistic baseline",
            "Ensemble of decision trees with bootstrap sampling",
            "Sequential boosting; slower on sparse data",
            "Adaptive boosting; overfits sparse TF-IDF",
            "Explicit bagging; poor fit for sparse text",
        ],
    })
    st.dataframe(model_results, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Ensemble Learning (Unit 5)")
    st.markdown("""
**Bagging** — Bootstrap Aggregating:
- Trains multiple base learners on different bootstrap samples
- Aggregates predictions (voting for classification)
- Examples: Random Forest, Bagging Classifier

**Boosting** — Sequential ensemble learning:
- Trains models sequentially, each correcting the previous model's errors
- AdaBoost: reweights misclassified samples higher
- Gradient Boosting: fits each new tree to residual errors of the previous

**Key Empirical Finding:**

| Category | Best Model | Accuracy |
|---|---|---|
| Classical (Linear) | Logistic Regression | **77.89%** |
| Classical (Probabilistic) | Naive Bayes | 76.42% |
| Ensemble (Bagging) | Random Forest | 76.14% |
| Ensemble (Boosting) | Gradient Boosting | 69.02% |

**Conclusion:** On high-dimensional sparse text data (10,000 TF-IDF features),
classical linear models **outperform tree ensembles by 8–14 percentage points**.
This is because linear decision boundaries are more sample-efficient in sparse
feature spaces than axis-aligned tree splits.
""")

    st.markdown("---")
    st.markdown("### Preprocessing Pipeline")
    st.markdown("""
    Every headline goes through these steps before classification:
    1. **Lowercase** — "Hello" → "hello"
    2. **Remove URLs** — strips `http://...` links
    3. **Remove @mentions and #hashtags** — cleans social artifacts
    4. **Remove punctuation and numbers** — keeps only letters
    5. **Tokenize** — splits into individual words
    6. **Remove stopwords** — drops "the", "is", "and", etc.
    7. **Stem** — "running" → "run"
    """)

    st.markdown("---")
    st.markdown("### Feature Extraction")
    st.markdown("""
    **TF-IDF (Term Frequency–Inverse Document Frequency)** converts clean text into a
    10,000-dimensional numeric vector. Each word gets a weight:
    - High weight → rare, informative word (e.g., "frustration")
    - Low weight → common word (e.g., "say", "today")

    We use **unigrams + bigrams** (`ngram_range=(1, 2)`) to capture phrases like "not good".
    """)

    st.markdown("---")
    st.markdown("### Known Limitation: Domain Shift")
    st.info("""
    The model was trained on **informal tweets** ("I love this"), but news headlines use
    **formal language** ("Cornell students voice frustration"). This causes lower confidence
    on news data — a classic problem called **domain shift**.

    **Planned fix:** Retrain on a mixed dataset (tweets + news + reviews).
    """)

# =========================================================
# TAB 4: ABOUT
# =========================================================
with tab4:
    st.markdown("### About This Project")
    st.markdown("""
    **Title:** Real-Time Sentiment Analysis with Web Scraping

    **Objective:** Build an end-to-end machine learning system that scrapes live news headlines,
    classifies their sentiment in real time, and presents results through an interactive dashboard.

    **Tech Stack:**
    - **Python 3.12** — core language
    - **pandas, numpy** — data handling
    - **nltk** — text preprocessing (stopwords, stemming)
    - **scikit-learn** — TF-IDF, model training, evaluation
    - **transformers, torch** — BERT model for comparison
    - **BeautifulSoup4, requests** — web scraping
    - **Streamlit** — dashboard framework
    - **matplotlib, wordcloud** — visualization
    - **joblib** — model persistence

    **Machine Learning Workflow:**
    1. Load Sentiment140 dataset (1.6M tweets)
    2. Preprocess text (clean, tokenize, stem)
    3. Vectorize with TF-IDF (10,000 features, bigrams)
    4. Train 7 models: classical + ensemble (Unit 5)
    5. Select best model (Logistic Regression, 77.89%)
    6. Cross-validate (5-fold) to confirm no overfitting
    7. Save model + vectorizer to disk
    8. Load in Streamlit for real-time inference
    9. Compare with BERT (transformer) for domain shift analysis
    """)

    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
    **Sentiment140** — 1.6 million tweets labeled as positive or negative.
    Downloaded from Kaggle. We use a balanced subset of 200,000 tweets (100k each).
    
    """)

    st.markdown("---")
    st.markdown("### Future Scope")
    st.markdown("""
    - **BERT/Transformers** — already integrated in the Model Comparison tab
    - **Multi-language support** — analyze Hindi, Telugu news
    - **Aspect-based sentiment** — "Camera is great, but battery is bad"
    - **Trend tracking** — sentiment over time
    - **Auto-alerts** — notify when sentiment drops suddenly
    - **Fine-tune on news domain** — fix the domain shift problem
    """)

    st.markdown("---")
    st.markdown("### Links")
    st.markdown("""
    - 🌐 **Live App:** https://ml-sentiment-analyzer.streamlit.app
    - 💻 **GitHub:** https://github.com/mahalakshmi-18-04-06/sentiment-project
    """)
    