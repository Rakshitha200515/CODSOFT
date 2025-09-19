# train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib

# Create model folder if not exists
os.makedirs('model', exist_ok=True)

# ----------------- Dataset & Features -----------------
DATA_PATH = 'data/Churn_Modelling.csv'
TARGET = 'Exited'  # Churn column

# Features to use (skip RowNumber, CustomerId, Surname)
FEATURES = ['CreditScore','Geography','Gender','Age','Tenure','Balance',
            'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

df = pd.read_csv(DATA_PATH)
y = df[TARGET]
X = df[FEATURES]

# ----------------- Feature Processing -----------------
num_features = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
cat_features = ['Geography','Gender','HasCrCard','IsActiveMember']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # <- corrected
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# ----------------- Models -----------------
models = {
    'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'gbt': GradientBoostingClassifier(random_state=42)
}

# ----------------- Train/Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ----------------- Training & Evaluation -----------------
results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    print(f'{name}: AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}')
    results[name] = {'pipe': pipe, 'auc': auc}

# Pick the best model by AUC
best_model_name = max(results, key=lambda k: results[k]['auc'])
best_pipe = results[best_model_name]['pipe']
print(f'Best model: {best_model_name}')

# ----------------- Save Model -----------------
joblib.dump(best_pipe, 'model/churn_model.joblib')
print('Saved trained model to model/churn_model.joblib')
