import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 📁 실험 디렉토리 및 로그 저장 경로
exp_name = "resnet34_cifar100_exp"
snapshot_root = "./snapshots"
os.makedirs(os.path.join(snapshot_root, exp_name), exist_ok=True)
log_path = os.path.join(snapshot_root, exp_name, "out")

# 로그 리디렉션
sys.stdout = Logger(log_path)

print(f"🔧 Experiment started at {datetime.now()}")
print(f"📁 Log saved to: {log_path}")

# ---------------- 학습 설정 ----------------
batch_size = 128
epochs = 30
lr = 0.1
save_path = os.path.join(snapshot_root, exp_name, "resnet34_cifar100.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 평균/표준편차 (직접 계산한 값)
mean = (0.5071, 0.4866, 0.4409)
std = (0.2673, 0.2564, 0.2762)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# 모델 구성
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------- 학습 루프 ----------------
final_loss = 0.0
final_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss:.3f}, Train Accuracy: {acc:.2f}%")
    scheduler.step()

    # 마지막 에폭 저장용
    final_loss = running_loss
    final_acc = acc

# ---------------- 결과 저장 ----------------
torch.save(model.state_dict(), save_path)
print("\n✅ Training complete.")
print(f"📈 Final Train Loss: {final_loss:.3f}")
print(f"📊 Final Train Accuracy: {final_acc:.2f}%")
print(f"💾 Model saved to: {save_path}")
