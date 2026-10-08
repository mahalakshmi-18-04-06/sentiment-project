import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import clean_text

# ---------- 1. LOAD DATA ----------
print("Loading dataset...")
df = pd.read_csv(
    "data/sentiment140.csv",
    encoding='latin-1',
    header=None,
    names=['target', 'id', 'date', 'flag', 'user', 'text']
)

# ---------- 2. KEEP ONLY POSITIVE / NEGATIVE ----------
df = df[df['target'].isin([0, 4])].copy()
df['target'] = df['target'].map({0: 0, 4: 1})  # 0=Neg, 1=Pos

# ---------- 3. BALANCE + SAMPLE (fast training) ----------
pos = df[df['target'] == 1].sample(100000, random_state=42)
neg = df[df['target'] == 0].sample(100000, random_state=42)
df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Working with {len(df)} rows")

# ---------- 4. CLEAN TEXT ----------
print("Cleaning text (may take a few minutes)...")
df['clean'] = df['text'].apply(clean_text)
df = df[df['clean'].str.strip() != ""]

# ---------- 5. SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['target'],
    test_size=0.2, random_state=42, stratify=df['target']
)

# ---------- 6. TF-IDF ----------
print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf  = tfidf.transform(X_test)

# ---------- 7. TRAIN MULTIPLE MODELS ----------
# ---------- 7. TRAIN MULTIPLE MODELS ----------
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         MultinomialNB(),
    "Linear SVM":          LinearSVC(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    # --- Unit 5: Ensemble Learning additions ---
    "Bagging (DT)":        BaggingClassifier(
                                estimator=DecisionTreeClassifier(max_depth=10),
                                n_estimators=50,
                                random_state=42,
                                n_jobs=-1,
                            ),
    "AdaBoost":            AdaBoostClassifier(
                                n_estimators=100,
                                random_state=42,
                            ),
    "Gradient Boosting":   GradientBoostingClassifier(
                                n_estimators=100,
                                random_state=42,
                            ),
}

results = {}
best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_tf, y_train)
    y_pred = model.predict(X_test_tf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# ---------- 8. CROSS-VALIDATION ----------
print(f"\n=== Cross-Validation on Best Model ({best_name}) ===")
cv_scores = cross_val_score(best_model, X_train_tf, y_train, cv=5)
print("CV Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

# ---------- 9. SAVE ----------
joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print(f"\n✅ Saved best model ({best_name}) with accuracy {best_acc:.4f}")

# ---------- 10. SUMMARY ----------
print("\n=== Model Comparison ===")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name:25s} : {acc:.4f}")