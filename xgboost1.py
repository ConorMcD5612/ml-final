import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load the same dataset as the neural network
df = pd.read_csv("Insurance_Data.csv")

# Drop same columns: Status + numeric (cols 1,9,10,11)
df.drop(df.columns[[1, 9, 10, 11]], axis=1, inplace=True)

# Drop rows with any nulls
df.dropna(inplace=True)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df["Disposition"])

# Encode features
X = pd.get_dummies(df.drop(columns=["Disposition"]))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42)

# XGBoost model for multiclass classification
num_classes = len(np.unique(y))
model = xgb.XGBClassifier(
    objective="multi:softprob",  # returns probabilities for mlogloss
    num_class=num_classes,
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=500,
    early_stopping_rounds=10
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", acc)
print(classification_report(y_test, y_pred, target_names=le.classes_))
