import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import datetime

# ✅ Logger 정의
class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# ✅ 디바이스 설정 및 로그 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = "resnet34_cifar100_exp"
snapshot_root = "./snapshots"
os.makedirs(os.path.join(snapshot_root, exp_name), exist_ok=True)
log_path = os.path.join(snapshot_root, exp_name, "out_reliability_diagram_temperature_scaling")
sys.stdout = Logger(log_path)

print(f"🔧 ECE Evaluation started at {datetime.now()}")
print(f"📁 Log saved to: {log_path}")

# ✅ CIFAR-100 Test 데이터셋
mean = (0.5071, 0.4866, 0.4409)
std = (0.2673, 0.2564, 0.2762)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

# ✅ ResNet34 모델 정의 및 weight 로드
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)
checkpoint_path = os.path.join(snapshot_root, exp_name, "resnet34_cifar100.pth")
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

# ✅ ECE 계산 함수
def compute_reliability_and_ece(model, dataloader, device, n_bins=15, verbose_under_100=True):
    model.eval()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
    bin_corrects = torch.zeros(n_bins).to(device)
    bin_confidences = torch.zeros(n_bins).to(device)
    bin_counts = torch.zeros(n_bins).to(device)
    total_samples = 0

    all_logits, all_confs, all_labels, all_preds = [], [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            corrects = preds.eq(labels)

            all_logits.append(logits.cpu())
            all_confs.append(confs.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

            total_samples += labels.size(0)
            for i in range(n_bins):
                in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
                bin_counts[i] += in_bin.sum()
                if in_bin.sum() > 0:
                    bin_corrects[i] += corrects[in_bin].float().sum()
                    bin_confidences[i] += confs[in_bin].sum()

    all_logits = torch.cat(all_logits)
    all_confs = torch.cat(all_confs)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    nonzero = bin_counts > 0
    accs = bin_corrects[nonzero] / bin_counts[nonzero]
    confs = bin_confidences[nonzero] / bin_counts[nonzero]
    bin_centers = ((bin_boundaries[:-1] + bin_boundaries[1:]) / 2)[nonzero]
    filtered_counts = bin_counts[nonzero]
    bin_boundaries_np = bin_boundaries.cpu().numpy()
    ece = torch.sum((filtered_counts / total_samples) * torch.abs(accs - confs)).item()

    print(f"\n📊 Confidence Bin Info:")
    for idx in range(n_bins):
        low = bin_boundaries_np[idx]
        high = bin_boundaries_np[idx + 1]
        count = int(bin_counts[idx].item())
        correct = int(bin_corrects[idx].item())
        if count > 0:
            acc = correct / count * 100
            print(f"[{low:.4f}, {high:.4f}) → {count} samples, {correct} correct → Accuracy: {acc:.2f}%")
        else:
            print(f"[{low:.4f}, {high:.4f}) → 0 samples")

    return bin_centers.cpu().numpy(), accs.cpu().numpy(), confs.cpu().numpy(), filtered_counts.cpu().numpy(), total_samples, ece

# ✅ 시각화 함수
def draw_fancy_reliability_diagram(bin_centers, accs, confs, bin_counts, total_samples, ece, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    width = 1.0 / len(bin_centers)

    plt.figure(figsize=(5, 5))
    plt.bar(bin_centers, accs, width=width * 0.9, color='blue', edgecolor='black', label='Outputs', alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    for x, acc, conf in zip(bin_centers, accs, confs):
        lower = min(acc, conf)
        upper = max(acc, conf)
        plt.fill_between([x - width / 2, x + width / 2], lower, upper,
                         color='red', alpha=0.3, hatch='//', edgecolor='r', linewidth=0,
                         label='Gap' if x == bin_centers[0] else "")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram: {name}")
    plt.legend(loc='upper left')
    plt.text(0.02, 0.6, f"ECE = {ece * 100:.2f}%", fontsize=12,
             bbox=dict(facecolor='lavender', edgecolor='gray'))
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{name}_fancy.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[{name}] Reliability diagram saved at: {save_path}")

# ✅ Temperature Scaling 실험 함수
def evaluate_multiple_temperatures_with_plots(model, test_loader, device, T_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ece_list = []

    class TemperatureScaler(nn.Module):
        def __init__(self, temperature: float):
            super().__init__()
            self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)
        def forward(self, logits):
            return logits / self.temperature

    class WrappedModel(nn.Module):
        def __init__(self, base_model, temp_scaler):
            super().__init__()
            self.base_model = base_model
            self.temp_scaler = temp_scaler
        def forward(self, x):
            logits = self.base_model(x)
            return self.temp_scaler(logits)

    for T in T_values:
        print(f"\n🧪 Evaluating T = {T}")
        temp_scaler = TemperatureScaler(temperature=T).to(device)
        wrapped_model = WrappedModel(model, temp_scaler).to(device)

        bin_centers, accs, confs, bin_counts, total_samples, ece = compute_reliability_and_ece(
            wrapped_model, test_loader, device, verbose_under_100=False
        )
        ece_list.append(ece)

        draw_fancy_reliability_diagram(
            bin_centers, accs, confs, bin_counts, total_samples, ece,
            name=f"T={T}", output_dir=output_dir
        )

    return T_values, ece_list

# ✅ T vs. ECE 시각화
def plot_temperature_vs_ece(T_values, ece_list, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(T_values, [ece * 100 for ece in ece_list], marker='o', linestyle='-', color='purple')
    plt.xlabel("Temperature (T)")
    plt.ylabel("ECE (%)")
    plt.title("ECE vs Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 T vs ECE plot saved at: {save_path}")


# ✅ 실행: 다양한 T에 대해 평가
T_values_to_try = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
T_values, ece_list = evaluate_multiple_temperatures_with_plots(
    model=model,
    test_loader=test_loader,
    device=device,
    T_values=T_values_to_try,
    output_dir=os.path.join(snapshot_root, exp_name)
)

# ✅ 결과 요약 출력
print("\n📊 Temperature vs. ECE Summary:")
for T, ece in zip(T_values, ece_list):
    print(f"T = {T:.2f} → ECE = {ece * 100:.2f}%")

# ✅ T vs. ECE 그래프 저장
plot_temperature_vs_ece(
    T_values, ece_list,
    save_path=os.path.join(snapshot_root, exp_name, "temperature_vs_ece.png")
)
