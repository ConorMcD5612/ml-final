import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_csv("Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv")
df = df.dropna(subset=["Opened", "Closed", "Coverage", "SubCoverage", "Reason", "SubReason",
                       "Disposition", "Conclusion", "Recovery"])

df["Opened"] = pd.to_datetime(df["Opened"])
df["Closed"] = pd.to_datetime(df["Closed"])
df["Duration"] = (df["Closed"] - df["Opened"]).dt.days
df["Recovery"] = np.log1p(df["Recovery"])  # Log-transform for stability

# Encode categorical features
cat_cols = ["Coverage", "SubCoverage", "Reason", "SubReason"]
label_encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
for col in cat_cols:
    df[col] = label_encoders[col].transform(df[col])

# Encode targets
le_disp = LabelEncoder().fit(df["Disposition"])
le_conc = LabelEncoder().fit(df["Conclusion"])
df["Disposition"] = le_disp.transform(df["Disposition"])
df["Conclusion"] = le_conc.transform(df["Conclusion"])

# Use a sample
df = df.sample(n=1000, random_state=42)

# Split features and targets
X = df[cat_cols + ["Duration"]].values
y_disp = df["Disposition"].values
y_conc = df["Conclusion"].values
y_recovery = df["Recovery"].values

# Train/val split
X_train, X_val, y_disp_train, y_disp_val, y_conc_train, y_conc_val, y_rec_train, y_rec_val = train_test_split(
    X, y_disp, y_conc, y_recovery, test_size=0.2, random_state=42)

# Dataset class
class InsuranceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_disp, y_conc, y_rec):
        self.X_cat = torch.LongTensor(X[:, :4])
        self.duration = torch.FloatTensor(X[:, 4].reshape(-1, 1))
        self.y_disp = torch.LongTensor(y_disp)
        self.y_conc = torch.LongTensor(y_conc)
        self.y_rec = torch.FloatTensor(y_rec.reshape(-1, 1))

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.duration[idx], self.y_disp[idx], self.y_conc[idx], self.y_rec[idx]

train_ds = InsuranceDataset(X_train, y_disp_train, y_conc_train, y_rec_train)
val_ds = InsuranceDataset(X_val, y_disp_val, y_conc_val, y_rec_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

# Model
class EmbeddingModel(nn.Module):
    def __init__(self, cardinalities, emb_dim, hidden=128, dropout=0.3, num_disp=13, num_conc=45):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cardinalities])
        input_size = len(cardinalities) * emb_dim + 1

        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )
        self.head_disp = nn.Linear(hidden, num_disp)
        self.head_conc = nn.Linear(hidden, num_conc)
        self.head_rec = nn.Linear(hidden, 1)

    def forward(self, x_cat, duration):
        emb = torch.cat([emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeds)], dim=1)
        x = torch.cat([emb, duration], dim=1)
        shared = self.backbone(x)
        return self.head_disp(shared), self.head_conc(shared), self.head_rec(shared)

# Init model
cardinalities = [len(label_encoders[col].classes_) for col in cat_cols]
model = EmbeddingModel(cardinalities, emb_dim=8,
                       num_disp=len(le_disp.classes_),
                       num_conc=len(le_conc.classes_))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_disp = nn.CrossEntropyLoss()
loss_conc = nn.CrossEntropyLoss()
loss_rec = nn.MSELoss()

# Training loop with early stopping
best_val_loss = float('inf')
patience = 5
wait = 0

for epoch in range(100):
    model.train()
    for x_cat, duration, y_disp, y_conc, y_rec in train_loader:
        x_cat, duration, y_disp, y_conc, y_rec = x_cat.to(device), duration.to(device), y_disp.to(device), y_conc.to(device), y_rec.to(device)
        out_disp, out_conc, out_rec = model(x_cat, duration)
        loss = loss_disp(out_disp, y_disp) + loss_conc(out_conc, y_conc) + loss_rec(out_rec, y_rec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_cat, duration, y_disp, y_conc, y_rec in val_loader:
            x_cat, duration, y_disp, y_conc, y_rec = x_cat.to(device), duration.to(device), y_disp.to(device), y_conc.to(device), y_rec.to(device)
            out_disp, out_conc, out_rec = model(x_cat, duration)
            val_loss += (loss_disp(out_disp, y_disp) +
                         loss_conc(out_conc, y_conc) +
                         loss_rec(out_rec, y_rec)).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# Reload best model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Evaluation
all_disp_pred, all_conc_pred, all_rec_pred = [], [], []
all_disp_true, all_conc_true, all_rec_true = [], [], []

with torch.no_grad():
    for x_cat, duration, y_disp, y_conc, y_rec in val_loader:
        x_cat, duration = x_cat.to(device), duration.to(device)
        out_disp, out_conc, out_rec = model(x_cat, duration)
        all_disp_pred.extend(out_disp.argmax(1).cpu().numpy())
        all_conc_pred.extend(out_conc.argmax(1).cpu().numpy())
        all_rec_pred.extend(out_rec.cpu().numpy().flatten())
        all_disp_true.extend(y_disp.numpy())
        all_conc_true.extend(y_conc.numpy())
        all_rec_true.extend(y_rec.numpy())

print("\n--- Final Evaluation ---")
print("Disposition Accuracy:", accuracy_score(all_disp_true, all_disp_pred))
print("Conclusion Accuracy:", accuracy_score(all_conc_true, all_conc_pred))
print("Recovery R² Score:", r2_score(all_rec_true, all_rec_pred))
print("Recovery MSE:", mean_squared_error(all_rec_true, all_rec_pred))
