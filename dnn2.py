import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ===============================
# 1. DNN 모델 정의 
# ===============================
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
        x = torch.sigmoid(self.fc3(x))  # 이진 분류이므로 sigmoid
        return x

# ==================================
# 2. 학습/검증 데이터 로드 및 전처리
# ==================================
# 2-1. 데이터 파일 불러오기
train_df = pd.read_csv("./data/train_small.csv")
val_df = pd.read_csv("./data/valid_small.csv")

# 2-2. feature와 target 분리
X_train = train_df.drop(['ID_code', 'target'], axis=1).values
y_train = train_df['target'].values

X_val = val_df.drop(['ID_code', 'target'], axis=1).values
y_val = val_df['target'].values

# 2-3. StandardScaler로 정규화 (평균 0, 표준편차 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 2-4. Torch Tensor로 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

# 2-5. TensorDataset과 DataLoader 생성
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

# ======================
# 3. 모델, 손실함수, 옵티마이저 정의
# ======================
input_dim = X_train.shape[1]
model = DNN(input_dim)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# 4. 학습 루프 (에폭 반복)
# ======================
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

# ======================
# 5. 검증 평가 (Accuracy, ROC AUC)
# ======================
model.eval()
y_true = []
y_probs = []

with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb)
        y_probs.extend(preds.squeeze().numpy())  # 확률값
        y_true.extend(yb.squeeze().numpy())      # 정답값

# 0.5 기준으로 이진 예측
y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

# 평가 지표 계산
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation ROC AUC:  {auc:.4f}")

