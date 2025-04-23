import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

# Load data
df = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")

# Define columns
input_cols = ["Opened", "Closed", "Coverage", "SubCoverage", "Reason", "SubReason"]
classification_targets = ["Disposition", "Conclusion"]
regression_target = "Recovery"

# Drop rows with nulls in relevant columns
categorical_inputs = ["Coverage", "SubCoverage", "Reason", "SubReason"]
for col in categorical_inputs:
    df[col] = df[col].fillna("Unknown")

# Drop rows where any target column is missing
target_cols = ["Disposition", "Conclusion", "Recovery"]
df = df.dropna(subset=["Opened", "Closed"] + target_cols)

# Convert date columns to datetime and compute duration
df["Opened"] = pd.to_datetime(df["Opened"])
df["Closed"] = pd.to_datetime(df["Closed"])
df["Duration"] = (df["Closed"] - df["Opened"]).dt.days

# Encode categorical features
label_encoders = {}
for col in ["Coverage", "SubCoverage", "Reason", "SubReason"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features to use
features = ["Duration", "Coverage", "SubCoverage", "Reason", "SubReason"]

# Limit dataset for speed
df_sample = df.sample(n=1000, random_state=42)

# Filter out rare Conclusion classes BEFORE classification
value_counts = df_sample["Conclusion"].value_counts()
valid_classes = value_counts[value_counts >= 2].index
df_sample = df_sample[df_sample["Conclusion"].isin(valid_classes)]

# --- Classification: Disposition and Conclusion ---
for target in classification_targets:
    print(f"\n--- XGBoost Classification Report for {target} ---")
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_sample[target])
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample[features], y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(n_estimators=50, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# --- Regression: Recovery ---
print("\n--- XGBoost Regression Report for Recovery ---")
y = df_sample[regression_target]
X_train, X_test, y_train, y_test = train_test_split(df_sample[features], y, test_size=0.2, random_state=42)

reg_model = XGBRegressor(n_estimators=50)
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
