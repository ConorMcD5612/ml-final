import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor

# === Load Dataset ===
df = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")

# === Define columns to use ===
# Input features (including dates)
input_cols = ["Opened", "Closed", "Coverage", "SubCoverage", "Reason", "SubReason"]

# Classification targets
classification_targets = ["Disposition", "Conclusion"]

# Regression target
regression_target = "Recovery"

# === Remove rows with missing values in any relevant columns ===
df.dropna(subset=input_cols + classification_targets + [regression_target], inplace=True)

# === Process Dates and Create Duration Feature ===
df["Opened"] = pd.to_datetime(df["Opened"])
df["Closed"] = pd.to_datetime(df["Closed"])
df["Duration"] = (df["Closed"] - df["Opened"]).dt.days

# === CatBoost handles categorical features natively ===
categorical_cols = ["Coverage", "SubCoverage", "Reason", "SubReason"]

# === Reduce dataset size for faster training ===
df_sample = df.sample(n=1000, random_state=42)

# === Filter out rare classes in 'Conclusion' target to avoid stratification issues ===
class_counts = df_sample["Conclusion"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_sample = df_sample[df_sample["Conclusion"].isin(valid_classes)]

# === Classification: Train separate models for Disposition and Conclusion ===
for target in classification_targets:
    print(f"\n--- CatBoost Classification Report for {target} ---")

    # Define inputs and target
    X = df_sample[["Duration"] + categorical_cols]
    y = df_sample[target]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and train CatBoost classifier
    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train, cat_features=categorical_cols)

    # Predict and print metrics
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# === Regression: Predicting the 'Recovery' amount ===
print("\n--- CatBoost Regression Report for Recovery ---")

# Define inputs and target
X = df_sample[["Duration"] + categorical_cols]
y = df_sample[regression_target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train CatBoost regressor
reg_model = CatBoostRegressor(verbose=0)
reg_model.fit(X_train, y_train, cat_features=categorical_cols)

# Predict and print regression metrics
y_pred = reg_model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
