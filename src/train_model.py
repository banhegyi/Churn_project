import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ---------------------------
# 1️⃣ Adat betöltés
# ---------------------------
DATA_PATH = os.path.join('data', 'Telco-Customer-Churn.csv')
df = pd.read_csv(DATA_PATH)

# Feltételezve, hogy a target oszlop 'Churn'
target = 'Churn'

# ---------------------------
# 2️⃣ Feature-ek szétválasztása
# ---------------------------
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove(target)  # target ne legyen benne

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

X = df.drop(columns=[target])
y = df[target]

# ---------------------------
# 3️⃣ Preprocessing pipeline
# ---------------------------
# OneHotEncoder a kategóriákhoz (sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # numerikus marad
)

# ---------------------------
# 4️⃣ Modell pipeline
# ---------------------------
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# ---------------------------
# 5️⃣ Train-test split és tanítás
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# ---------------------------
# 6️⃣ Modell mentése
# ---------------------------
os.makedirs('models', exist_ok=True)
MODEL_PATH = os.path.join('models', 'churn_model.pkl')
joblib.dump(clf, MODEL_PATH)

print(f"✅ Model trained and saved to {MODEL_PATH}")
