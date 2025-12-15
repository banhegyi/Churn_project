import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------
# 1️⃣ CSV betöltése
# ---------------------------
df = pd.read_csv("data/Telco-Customer-Churn.csv")  # módosítsd a saját CSV-re

# Példa: célváltozó
target_column = 'Churn'  # cseréld a tényleges target oszlopra

# ---------------------------
# 2️⃣ Feature-ek
# ---------------------------
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
if target_column in categorical_features:
    categorical_features.remove(target_column)

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_column in numeric_features:
    numeric_features.remove(target_column)

# ---------------------------
# 3️⃣ Preprocessor pipeline
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# ---------------------------
# 4️⃣ Input X és y előkészítése
# ---------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# ---------------------------
# 5️⃣ Pipeline illesztése
# ---------------------------
X_t = preprocessor.fit_transform(X)

# ---------------------------
# 6️⃣ Mentés
# ---------------------------
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)

# ---------------------------
# 7️⃣ Output visszaadása
# ---------------------------
print("Preprocessing completed!")
print(f"Preprocessor saved at: {preprocessor_path}")
print(f"Transformed X shape: {X_t.shape}")
print(f"Target y shape: {y.shape}")
