# 🔮 SentimentScope

### Real-Time Sentiment Analysis with Web Scraping

An end-to-end machine learning system that scrapes live news headlines from multiple sources, classifies their sentiment using a trained ML model, and presents the results through an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

👉 **[Try the live app](https://your-url-here.streamlit.app)** *(update after deployment)*

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 **Multi-Source Scraping** | Fetch headlines from BBC, The Hindu, Times of India, NDTV, Reuters, Al Jazeera |
| 🧠 **ML Classification** | Logistic Regression trained on 200,000 tweets (77.89% accuracy) |
| 📊 **Rich Visual Analytics** | Pie charts, word clouds, per-source breakdown, top headlines |
| ⚡ **Real-Time Inference** | Scrape → classify → visualize in 3–5 seconds |
| 📁 **CSV Export** | Download results with one click |
| 🎨 **Professional UI** | Branded header, 3 tabs, colored metric cards |

---

## 🏗️ Architecture
User Input (News Sources)
↓
Web Scraper (BeautifulSoup + requests)
↓
Text Preprocessor (NLTK — clean, tokenize, stem)
↓
TF-IDF Vectorizer (10,000 features, bigrams)
↓
Logistic Regression Classifier (77.89%)
↓
Streamlit Dashboard (charts + tables)

text

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data Handling | pandas, numpy |
| NLP | NLTK (stopwords, PorterStemmer) |
| ML | scikit-learn (TfidfVectorizer, LogisticRegression) |
| Scraping | requests, BeautifulSoup4 |
| Dashboard | Streamlit |
| Visualization | matplotlib, wordcloud |
| Persistence | joblib |

---

## 📊 Model Comparison

Four classical ML models were trained on a balanced subset of 200,000 labeled tweets.

| Model | Accuracy | Notes |
|---|---|---|
| **Logistic Regression** ⭐ | **77.89%** | **Selected for production** |
| Linear SVM | 77.30% | Close second |
| Naive Bayes | 76.42% | Fastest |
| Random Forest | 76.14% | Slower, less accurate |

**Cross-validation (5-fold):** 77.20% — confirms the model is not overfitting.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip
- Git (optional, for cloning)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/sentiment-project.git
cd sentiment-project

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
🎬 Usage
Train the model (one-time)
bash
python train.py
This will:

Load the Sentiment140 dataset from data/sentiment140.csv

Preprocess 200,000 tweets

Train 4 models and compare accuracy

Save the best model + TF-IDF vectorizer to models/

Run the dashboard
bash
streamlit run app.py
Then open http://localhost:8501 in your browser.

📁 Project Structure
text
sentiment_project/
├── data/
│   └── sentiment140.csv              # Dataset (not committed — see below)
├── models/
│   ├── sentiment_model.pkl           # Trained classifier
│   └── tfidf_vectorizer.pkl          # TF-IDF vectorizer
├── preprocess.py                     # Text cleaning pipeline
├── train.py                          # Model training script
├── scraper.py                        # Multi-source web scraping
├── app.py                            # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
📈 Dataset
Sentiment140 — 1.6 million labeled tweets (Go, Bhayani & Huang, 2009).

Labels: 0 = Negative, 4 = Positive

We use a balanced subset of 200,000 tweets (100k each)

Download: Kaggle — Sentiment140

Note: The CSV is excluded from this repo due to its size (~230 MB). Download it manually and place it in data/sentiment140.csv before running train.py.

🧹 Preprocessing Pipeline
Each headline/text goes through these steps before classification:

Lowercase — "Hello" → "hello"

Remove URLs — strips http://... links

Remove mentions & hashtags — cleans social media artifacts

Remove punctuation & numbers — keeps only letters

Tokenize — splits text into words

Remove stopwords — drops "the", "is", "and", etc.

Stem — "running" → "run"

📐 Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) converts each cleaned text into a 10,000-dimensional numeric vector.

High weight → rare, informative words (e.g., "frustration")

Low weight → common words (e.g., "say", "today")

Uses unigrams + bigrams to capture phrases like "not good"

⚠️ Known Limitation
The model was trained on informal tweets ("I love this 😍") but is applied to formal news headlines ("Cornell students voice frustration"). This causes lower confidence on news data — a classic domain shift problem.

Planned fix: Retrain on a mixed dataset (tweets + news + reviews) and compare with BERT.

🚀 Future Scope
BERT / Transformers — boost accuracy to 90%+

Multi-language support — Hindi, Telugu, Tamil news

Aspect-based sentiment — "Camera is great, but battery is bad"

Trend tracking — sentiment over time

Auto-alerts — notify when sentiment drops suddenly

Fine-tune on news domain — fix domain shift

📜 License
This project is licensed under the MIT License.

🙏 Acknowledgements
Dataset: Sentiment140 — Go, Bhayani & Huang (2009)

Frameworks: scikit-learn, Streamlit, NLTK, BeautifulSoup

Inspiration: Modern NLP pipelines and real-time analytics dashboards

👤 Author
Your R.MAHA LAKSHMI
B.Tech — Artificial Intelligence & Machine Learning
Aditya University

📧 your-email@example.com
🔗 GitHub

