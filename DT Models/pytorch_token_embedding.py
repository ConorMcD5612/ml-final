import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ----------------------------
# 1. Load and Prepare the Data
# ----------------------------

df = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")

# Drop rows with missing important data
cols_needed = ["Opened", "Closed", "Coverage", "SubCoverage", "Reason", "SubReason",
               "Disposition", "Conclusion", "Recovery"]
df.dropna(subset=cols_needed, inplace=True)

# Compute duration in days
df["Opened"] = pd.to_datetime(df["Opened"])
df["Closed"] = pd.to_datetime(df["Closed"])
df["Duration"] = (df["Closed"] - df["Opened"]).dt.days

# Log-transform Recovery for stability
df["Recovery"] = np.log1p(df["Recovery"])

# ----------------------------
# 2. Encode Categorical Features
# ----------------------------

categorical_cols = ["Coverage", "SubCoverage", "Reason", "SubReason"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode targets
le_disp = LabelEncoder()
le_conc = LabelEncoder()
df["Disposition"] = le_disp.fit_transform(df["Disposition"])
df["Conclusion"] = le_conc.fit_transform(df["Conclusion"])

# ----------------------------
# 3. Train/Test Split
# ----------------------------

# Use a small sample for speed
df = df.sample(n=1000, random_state=42)

# Features and targets
X = df[categorical_cols + ["Duration"]].values
y_disp = df["Disposition"].values
y_conc = df["Conclusion"].values
y_rec = df["Recovery"].values

# Split all targets along with features
X_train, X_val, y_disp_train, y_disp_val, y_conc_train, y_conc_val, y_rec_train, y_rec_val = train_test_split(
    X, y_disp, y_conc, y_rec, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Custom Dataset
# ----------------------------

class InsuranceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_disp, y_conc, y_rec):
        self.X_cat = torch.LongTensor(X[:, :4])            # categorical inputs
        self.duration = torch.FloatTensor(X[:, 4:5])       # duration (numerical)
        self.y_disp = torch.LongTensor(y_disp)             # target 1
        self.y_conc = torch.LongTensor(y_conc)             # target 2
        self.y_rec = torch.FloatTensor(y_rec.reshape(-1, 1))  # regression target

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.duration[idx], self.y_disp[idx], self.y_conc[idx], self.y_rec[idx]

# Create DataLoaders
train_ds = InsuranceDataset(X_train, y_disp_train, y_conc_train, y_rec_train)
val_ds = InsuranceDataset(X_val, y_disp_val, y_conc_val, y_rec_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

# ----------------------------
# 5. Neural Network with Embeddings
# ----------------------------

class MultiTaskModel(nn.Module):
    def __init__(self, cardinalities, emb_dim=8, hidden=128, dropout=0.3, num_disp=13, num_conc=45):
        super().__init__()
        # Embedding layers for categorical inputs
        self.embeds = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cardinalities])
        input_size = len(cardinalities) * emb_dim + 1  # embeddings + duration

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )

        # Output heads
        self.head_disp = nn.Linear(hidden, num_disp)
        self.head_conc = nn.Linear(hidden, num_conc)
        self.head_rec = nn.Linear(hidden, 1)

    def forward(self, x_cat, duration):
        # Embed and concatenate all categorical inputs
        emb = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
        x = torch.cat([emb, duration], dim=1)
        shared = self.backbone(x)

        # Return predictions for each task
        return self.head_disp(shared), self.head_conc(shared), self.head_rec(shared)

# ----------------------------
# 6. Train the Model
# ----------------------------

# Cardinality for each categorical feature
cardinalities = [len(label_encoders[col].classes_) for col in categorical_cols]

# Init model
model = MultiTaskModel(cardinalities,
                       num_disp=len(le_disp.classes_),
                       num_conc=len(le_conc.classes_))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer & Loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_disp = nn.CrossEntropyLoss()
loss_conc = nn.CrossEntropyLoss()
loss_rec = nn.MSELoss()

# Early stopping setup
best_val_loss = float('inf')
patience, wait = 5, 0

for epoch in range(100):
    model.train()
    for x_cat, duration, y1, y2, y3 in train_loader:
        x_cat, duration, y1, y2, y3 = x_cat.to(device), duration.to(device), y1.to(device), y2.to(device), y3.to(device)

        # Forward pass
        out1, out2, out3 = model(x_cat, duration)

        # Multi-task loss
        loss = loss_disp(out1, y1) + loss_conc(out2, y2) + loss_rec(out3, y3)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_cat, duration, y1, y2, y3 in val_loader:
            x_cat, duration, y1, y2, y3 = x_cat.to(device), duration.to(device), y1.to(device), y2.to(device), y3.to(device)
            out1, out2, out3 = model(x_cat, duration)
            val_loss += (loss_disp(out1, y1) + loss_conc(out2, y2) + loss_rec(out3, y3)).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ----------------------------
# 7. Evaluate Best Model
# ----------------------------

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Store predictions and true values
disp_preds, conc_preds, rec_preds = [], [], []
disp_true, conc_true, rec_true = [], [], []

with torch.no_grad():
    for x_cat, duration, y1, y2, y3 in val_loader:
        x_cat, duration = x_cat.to(device), duration.to(device)
        out1, out2, out3 = model(x_cat, duration)

        # Predictions
        disp_preds.extend(out1.argmax(1).cpu().numpy())
        conc_preds.extend(out2.argmax(1).cpu().numpy())
        rec_preds.extend(out3.cpu().numpy().flatten())

        # Ground truths
        disp_true.extend(y1.numpy())
        conc_true.extend(y2.numpy())
        rec_true.extend(y3.numpy())

# Print evaluation results
print("\n--- Final Evaluation ---")
print("Disposition Accuracy:", accuracy_score(disp_true, disp_preds))
print("Conclusion Accuracy:", accuracy_score(conc_true, conc_preds))
print("Recovery R² Score:", r2_score(rec_true, rec_preds))
print("Recovery MSE:", mean_squared_error(rec_true, rec_preds))
