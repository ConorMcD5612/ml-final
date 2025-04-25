import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

# Load dataset
df = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")

# Define relevant columns
input_columns = ["Opened", "Closed", "Coverage", "SubCoverage", "Reason", "SubReason"]
classification_targets = ["Disposition", "Conclusion"]
regression_target = "Recovery"

# Fill missing values in categorical columns with "Unknown"
categorical_cols = ["Coverage", "SubCoverage", "Reason", "SubReason"]
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# Remove rows with missing dates or target values
df = df.dropna(subset=["Opened", "Closed", "Disposition", "Conclusion", "Recovery"])

# Convert date columns to datetime format
df["Opened"] = pd.to_datetime(df["Opened"])
df["Closed"] = pd.to_datetime(df["Closed"])

# Create a new feature: duration of complaint in days
df["Duration"] = (df["Closed"] - df["Opened"]).dt.days

# Encode categorical columns as integers using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder

# Define final feature set for modeling
features = ["Duration"] + categorical_cols

# Sample a smaller subset for faster training
df_sample = df.sample(n=1000, random_state=42)

# Remove rare classes in Conclusion to avoid stratification issues
valid_conclusions = df_sample["Conclusion"].value_counts()
valid_classes = valid_conclusions[valid_conclusions >= 2].index
df_sample = df_sample[df_sample["Conclusion"].isin(valid_classes)]

# ---- Classification Section ----
for target in classification_targets:
    print(f"\n--- XGBoost Classification Report for {target} ---")

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_sample[target])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample[features], y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost classifier
    clf = XGBClassifier(n_estimators=50, eval_metric='mlogloss')
    clf.fit(X_train, y_train)

    # Predict and print metrics
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# ---- Regression Section ----
print("\n--- XGBoost Regression Report for Recovery ---")

# Target and features
y = df_sample[regression_target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_sample[features], y, test_size=0.2, random_state=42
)

# Train XGBoost regressor
reg = XGBRegressor(n_estimators=50)
reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
