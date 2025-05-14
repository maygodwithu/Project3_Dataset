import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Custom Dataset
class TabularDataset(Dataset):
    def __init__(self, csv_file, scaler):
        df = pd.read_csv(csv_file)
        self.X = scaler.transform(df.drop(['ID_code', 'target'], axis=1).values)
        self.y = df['target'].values.astype(float)
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Define DNN model
class DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 3. Load data & prepare scaler
train_df = pd.read_csv("./data/train_small.csv")
scaler = StandardScaler()
scaler.fit(train_df.drop(['ID_code', 'target'], axis=1))

# 4. Dataset & DataLoader
train_dataset = TabularDataset("./data/train_small.csv", scaler=scaler)
val_dataset = TabularDataset("./data/valid_small.csv", scaler=scaler)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 5. Initialize model
input_dim = train_dataset[0][0].shape[0]
model = DNN(input_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 7. Evaluation
# 평가 모드
model.eval()
y_true = []
y_probs = []

# 검증 데이터 예측
with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb)
        y_probs.extend(preds.squeeze().numpy())
        y_true.extend(yb.squeeze().numpy())

# 이진 분류 결과
y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

# 평가 지표 계산
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

# 결과 출력
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation ROC AUC:  {auc:.4f}")

