---
title: "On Calibration of Modern Neural Networks"
date: 2025-05-19 18:00:00 +0900
categories:
  - Paper-Review
  - Neural Networks
tags:
  - Machine Learning
description: 
toc: true
comments: false
cdn: 
image:
math: true
pin: false
mermaid: false
---

## 📌 연구 개요
**Confidence Calibration**은 모델이 예측한 확률이 실제 정답일 가능성과 얼마나 일치하는지를 나타내는 개념이다. 예를 들어, 어떤 이미지에 대해 모델이 고양이일 확률을 0.9라고 예측했을 때, 이러한 예측이 잘 보정(calibrated)되어 있다면 실제로 그 이미지가 고양이일 확률도 약 90%가 되어야 한다는 것이다.

**On Calibration of Modern Neural Networks** 논문은 빠르게 발전하고 연구되어져 오고 있는 ResNet, DenseNet 등과 같은 현대적인 신경망 모델들이 높은 분류 정확도를 달성함에도 불구하고, 오히려 확률 보정(calibration) 성능은 더 나빠졌다는 사실을 실험적으로 보이고 있다. 과거의 얕은 신경망 모델들은 예측 확률이 실제 정답 확률과 비교적 잘 일치했지만, 깊고 복잡한 구조를 가진 최신 모델들은 자신 있게 예측을 하면서도 그 확률이 실제 정답률과 불일치하는 경향이 있다는 점을 짚고 있다. 한 마디로 말해 최신 모델들은 Overconfident 되어져 있다는 점을 발견하였다.


![Desktop View](/assets/img/paper-review/On_Calibration_of_modern_NN/figure1.png)
_Figure 1.1: Confidence histograms (top) and reliability diagrams (bottom) for a 5-layer LeNet (left) and a 110-layer ResNet (right) on CIFAR-100. Refer to the text below for detailed illustration._


이러한 문제는 자율주행, 의료 진단, 법률 판단 등과 같이 예측 결과에 대한 신뢰도가 매우 중요한 응용 분야에서 특히 도드라질 수 있기에 모델이 단순히 정확할 뿐만 아니라, 자신의 예측이 얼마나 확실한지에 대한 표현 또한 신뢰할 수 있어야 한다.

본 논문에서는 이러한 문제를 해결하기 위해 다양한 사후 보정(post-hoc calibration) 방법들을 실험적으로 비교하고, 그 중에서도 Temperature Scaling이라는 단 하나의 스칼라 파라미터만을 사용하는 간단한 방법이 매우 효과적이라는 사실을 밝혀냈다. 본 글에서는 이의 실험 코드도 구현하였다.

## 🔍 신경망의 Overconfidence 원인 분석

최근의 신경망 모델들은 높은 정확도를 자랑하지만, 그 **confidence (예측 확률)** 는 실제 정답률과 잘 맞지 않는 경우가 많다. 이 현상을 **miscalibration (불완전한 보정)** 이라고 하며, 그 원인과 관련 요소들을 우선 분석하였다.


![Desktop View](/assets/img/paper-review/On_Calibration_of_modern_NN/figure2.png)
_Figure 2: The effect of network depth (far left), width (middle left), Batch Normalization (middle right), and weight decay (far right) on
miscalibration, as measured by ECE (lower is better)_


### 1. 모델 용량의 증가 (Model Capacity)

- 최근 딥러닝 모델들은 레이어 수와 필터 수가 급격히 증가하여, 학습 데이터를 더 잘 맞출 수 있는 **모델 용량(capacity)** 을 갖추게 되었다.
- 하지만 모델 용량이 커질수록 오히려 **confidence가 실제 정확도보다 과도하게 높아지는 과신, 즉 overconfidence** 하는 경향이 나타납니다.

실험 결과 (ResNet on CIFAR-100):
- 깊이(depth)를 증가시키면 Error은 줄어드나 ECE가 증가
- 필터 수(width)를 증가시키면 Error은 확연히 줄어드나, ECE가 증가

> 높은 capacity는 overfitting을 야기할 수 있으며, 이러한 경우 정확도는 좋아져도 확률의 품질은 나빠진다.
{: .prompt-tip }

---

### 2. Batch Normalization의 영향

- **Batch Normalization**은 딥러닝 모델의 학습을 안정화시키고 빠르게 만드는 기법으로, 현대 아키텍처에서 필수적으로 사용된다.
- 하지만, BN을 사용한 모델들은 **정확도는 올라가지만 calibration은 오히려 나빠지는** 현상이 나타납니다.

실험 결과 (6-layer ConvNet on CIFAR-100):
- BN을 적용한 ConvNet은 정확도가 약간 올라가지만(Error 감소), ECE는 뚜렷하게 증가

> BN은 내부 활성 분포를 정규화하여 학습을 더 잘 되게 하지만, 결과적으로 출력 확률이 더 과신된(overconfident) 상태로 나타나 calibration에는 부정적인 영향을 미치게 된다.
{: .prompt-tip }
---

### 3.3 Weight Decay 감소의 영향

- 전통적으로 **weight decay** 는 과적합을 막기 위한 정규화 방법으로 널리 사용되어 왔으며, overfitting을 방지하기 위해 가중치에 패널티를 주는 정규화 기법이다.
- 최근에는 BN의 정규화 효과 때문에 weight decay 사용량이 줄어드는 추세이다.
- 하지만 실험에서는 **weight decay를 증가시킬수록 calibration은 개선되는 경향**을 보입니다.

실험 결과 (ResNet-110 on CIFAR-100):
- Weight decay를 증가시키면 분류 정확도(Error)는 특정 구간에서 최적점을 찍고 이후 다시 증가
- ECE는 weight decay가 증가할수록 감소하는 경향을 보임

> 즉, 정확도를 최대화하는 weight decay 설정과 calibration을 최적화하는 설정은 서로 다를 수 있으며, 정확도는 유지되더라도 confidence는 왜곡될 수 있다.
{: .prompt-tip }
---

### 3.4 NLL 과적합 현상 (Overfitting to NLL)

![Desktop View](/assets/img/paper-review/On_Calibration_of_modern_NN/figure3.png)
_Figure 3: Test error and NLL of a 110-layer ResNet with stochastic depth on CIFAR-100 during training_

- 실험에서는 learning rate가 낮아지는 구간에서 test error는 계속 줄어드는 반면, NLL은 다시 증가하는 현상을 보였습니다.
- 이는 모델이 **정확도는 높이지만 confidence가 실제보다 과도한 상태로 학습이 진행되고 있음**을 의미합니다.

실험 결과 (ResNet-110 + stochastic depth on CIFAR-100):
- Epoch 250 이후 learning rate 감소
- 이후 test error는 감소 (29% → 27%)하지만, NLL은 증가

> 정확도(0/1 loss)는 줄지만, NLL에 대한 과적합으로 인해 **confidence가 비정상적으로 높아지는 보정 오류**가 발생합니다.




### 📐 Calibration의 정의 및 측정 방법

본 논문에서는 다중 클래스 분류 문제를 다루고 있으며, 딥러닝 모델은 주어진 입력 $X \in \mathcal{X}, \quad Y \in \{1, \dots, K\}$를 예측하는다고 가정한다. 모델의 예측 확률은 다음과 같이 정의된다.

$$
h(X) = (\hat{Y}, \hat{P})
$$

여기서 $\hat{Y}$는 예측된 클래스, $\hat{P}$는 각 클래스에 대한 확률 분포이며, softmax 출력의 최댓값으로 정의된다.

그럼 완벽하게 보정된 모델의 정의는 어떻게 될까? 본 논문에서는 다음과 같이 정의하고 있다.

$$
P(\hat{Y} = Y \mid \hat{P} = p) = p, \quad \forall p \in [0, 1]
$$

위 식에서 알 수 있듯이 모델이 **완벽히 보정(calibrated)**되어 있다는 것은, 예측한 확률 값이 실제 정답률과 일치하는 것을 의미한다. 예를 들어 모델이 100개의 샘플에 대해 모두 0.8의 confidence를 출력했다면, 실제로 그 중 약 80개가 맞아야 완벽히 보정된 것이다.

## 📊 실전에서는 어떻게 측정하는가?

### 🔍 Reliability Diagram (신뢰도 다이어그램)

예측 확률 $\hat{P}$를 구간으로 잘게 나누고, 각 구간에서의 **실제 정답률(accuracy)**과 **평균 confidence**를 비교한다. 만약 모델이 잘 보정되어져 있다면, 각 구간에서는 아래의 관계식이 성립해야한다는 것이다.

$$
\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)
$$

$$
\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i
$$

$$
\text{Accuracy}(B_m) \approx \text{Confidence}(B_m)
$$

여기서 $\text{acc}(B_m)$는 구간 $B_m$에 속하는 샘플들의 실제 정답률, $\text{conf}(B_m)$는 구간 $B_m$에 속하는 샘플들의 평균 confidence를 의미한다. 만약 모델이 잘 보정되어 있다면, 두 값은 서로 비슷해야 한다.

### 📏 Expected Calibration Error (ECE)

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

## 🛠️ 4. Calibration Methods

본 장에서는 이미 학습된 모델에 대해 **확률 보정을 위한 사후 처리(Post-hoc) 방법**들을 소개합니다. 이들은 모델의 예측 구조나 정확도는 유지하면서, **예측 확률(confidence)**이 실제 정답률과 더 잘 일치하도록 만들어줍니다.

---

### 📘 4.1 이진 분류에서의 Calibration

모델은 입력 \( X \)에 대해 다음과 같은 예측을 수행한다고 가정합니다:

$$
h(X) = (\hat{Y}, \hat{P})
$$

여기서  
- \( \hat{Y} \): 예측된 클래스  
- \( \hat{P} \): 예측 확률, 일반적으로 softmax 또는 sigmoid 출력의 최대값

---

#### 🔹 Histogram Binning

- 예측 확률 \( \hat{P} \in [0, 1] \) 구간을 균등하게 나누고,
- 각 구간(bin)마다 실제 정답률의 평균을 계산하여 보정된 확률로 사용

예:
- 0.7–0.8 구간에 100개의 샘플이 있고, 75개가 정답이면 → 보정 확률은 0.75

---

#### 🔹 Isotonic Regression

- 단조 증가 함수로 확률을 보정
- 구간 간 계단형으로 조정되며 유연하지만 과적합 가능성 존재

---

#### 🔹 Platt Scaling

- 로짓(logit)을 입력으로 하는 로지스틱 회귀 기반 보정 방법

$$
\hat{q}_i = \sigma(a z_i + b)
$$

- 파라미터 \( a, b \)는 validation set에서 학습
- 간단하고 안정적이며, 이진 분류에 효과적

---

### 📘 4.2 다중 클래스 분류에서의 확장

다중 클래스에서는 softmax를 사용하여 다음과 같이 확률을 계산합니다:

$$
\hat{P} = \max_k \left( \frac{e^{z_k}}{\sum_j e^{z_j}} \right)
$$

---

#### 🔹 Matrix Scaling

- 로짓 벡터 \( z \)에 선형 변환 적용:

$$
\hat{q} = \text{softmax}(W z + b)
$$

- \( W \in \mathbb{R}^{K \times K} \), \( b \in \mathbb{R}^K \)
- 강력한 표현력을 가지지만 파라미터 수가 많아 과적합 우려

---

#### 🔹 Vector Scaling

- Matrix Scaling의 간소화 버전으로 \( W \)를 대각 행렬로 제한

$$
\hat{q} = \text{softmax}(D z + b)
$$

- 파라미터 수가 \( 2K \)개로 줄어들며 계산량과 과적합 가능성이 감소

---

#### 🌡️ Temperature Scaling

- 가장 간단하면서도 강력한 보정 기법
- 로짓을 스칼라 \( T \)로 나누고 softmax 적용

$$
\hat{q} = \text{softmax}(z / T)
$$

- \( T > 1 \): 확률이 분산됨 (과신 완화)
- \( T = 1 \): 원래 모델과 동일
- \( T < 1 \): 확률이 더 sharp해짐

---

### 📌 Temperature Scaling의 특징

- 단 하나의 파라미터 \( T \)만 조정
- 모델의 예측 클래스는 유지되며, confidence만 조정됨
- 다중 클래스에서도 손쉽게 적용 가능

---

### 🧪 실험 중 Temperature Scaling의 효과

- Epoch 250 이후 learning rate 감소
- 이후 test error는 감소 (29% → 27%)하지만, NLL은 증가
- 이는 모델이 더 정확해졌지만, 확률 분포는 더 과신하게 되었음을 의미
- Temperature Scaling은 이러한 overconfidence를 효과적으로 완화함

---



### 실습

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# transform 없이 tensor만 받기
transform = transforms.ToTensor()
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(trainset, batch_size=50000, shuffle=False)

data = next(iter(loader))[0]  # (50000, 3, 32, 32)
mean = data.mean(dim=(0, 2, 3))
std = data.std(dim=(0, 2, 3))

print("CIFAR-100 평균:", mean)
print("CIFAR-100 표준편차:", std)
```
{: file='cifar10'}