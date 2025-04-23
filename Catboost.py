import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor

# Load data
df = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")

# Define columns
input_cols = ["Opened", "Closed", "Coverage", "SubCoverage", "Reason", "SubReason"]
classification_targets = ["Disposition", "Conclusion"]
regression_target = "Recovery"

# Drop rows with nulls
df = df.dropna(subset=input_cols + classification_targets + [regression_target])

# Convert date columns to datetime and compute duration
df["Opened"] = pd.to_datetime(df["Opened"])
df["Closed"] = pd.to_datetime(df["Closed"])
df["Duration"] = (df["Closed"] - df["Opened"]).dt.days

# Categorical features CatBoost can handle natively
categorical_cols = ["Coverage", "SubCoverage", "Reason", "SubReason"]

# Use a smaller sample for speed
df_sample = df.sample(n=1000, random_state=42)

value_counts = df_sample["Conclusion"].value_counts()
valid_classes = value_counts[value_counts >= 2].index
df_sample = df_sample[df_sample["Conclusion"].isin(valid_classes)]

# --- Classification: Disposition and Conclusion ---
for target in classification_targets:
    print(f"\n--- CatBoost Classification Report for {target} ---")
    
    y = df_sample[target]
    X = df_sample[["Duration"] + categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train, cat_features=categorical_cols)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# --- Regression: Recovery ---
print("\n--- CatBoost Regression Report for Recovery ---")

y = df_sample[regression_target]
X = df_sample[["Duration"] + categorical_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = CatBoostRegressor(verbose=0)
reg_model.fit(X_train, y_train, cat_features=categorical_cols)

y_pred = reg_model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
