# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ---------- Config ----------
TRAIN_FILE = 'train_data.txt'
TEST_FILE = 'test_data.txt'               # optional (no labels)
TEST_SOL_FILE = 'test_data_solution.txt'  # has true labels (if provided)
MODEL_OUT = 'movie_genre_model.pkl'
VECT_OUT = 'tfidf_vectorizer.pkl'
LE_OUT = 'label_encoder.pkl'
RANDOM_STATE = 42

# ---------- Load train data (separator " ::: ") ----------
def load_trible_col_file(path):
    # file format: id ::: title ::: genre ::: plot
    df = pd.read_csv(path, sep=' ::: ', engine='python', header=None,
                     names=['ID', 'Title', 'Genre', 'Plot'])
    df['Plot'] = df['Plot'].astype(str).str.strip()
    df['Genre'] = df['Genre'].astype(str).str.strip()
    return df

print("Loading training data...")
train = load_trible_col_file(TRAIN_FILE)
print("Total train samples:", len(train))

# ---------- Features / labels ----------
X_text = train['Plot']
y = train['Genre']

# ---------- Encode labels ----------
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(X_text)

# ---------- Train/validation split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15,
                                                  stratify=y_enc, random_state=RANDOM_STATE)

# ---------- Train classifier ----------
print("Training Logistic Regression with class_weight='balanced' ...")
clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
clf.fit(X_train, y_train)

# ---------- Validate ----------
y_val_pred = clf.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification report (validation):")
print(classification_report(y_val, y_val_pred, target_names=le.classes_, zero_division=0))

# ---------- If test solution exists, evaluate on it ----------
if os.path.exists(TEST_SOL_FILE):
    print("\nLoading test solution and evaluating...")
    test_sol = load_trible_col_file(TEST_SOL_FILE)
    X_test = vectorizer.transform(test_sol['Plot'])
    y_test_enc = le.transform(test_sol['Genre'])
    y_test_pred = clf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test_enc, y_test_pred))
    print("\nClassification report (test):")
    print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_, zero_division=0))

# ---------- Save model artifacts ----------
print("\nSaving model, vectorizer, and label encoder...")
joblib.dump(clf, MODEL_OUT)
joblib.dump(vectorizer, VECT_OUT)
joblib.dump(le, LE_OUT)
print("Saved:", MODEL_OUT, VECT_OUT, LE_OUT)
