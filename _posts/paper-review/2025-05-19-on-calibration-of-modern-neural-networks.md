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

## 신경망의 Overconfidence 원인 분석

최근의 신경망 모델들은 높은 정확도를 자랑하지만, 그 **confidence (예측 확률)** 는 실제 정답률과 잘 맞지 않는 경우가 많다. 이 현상을 **miscalibration (불완전한 보정)** 이라고 하며, 본 장에서는 그 원인과 관련 요소들을 분석합니다.

### 3.1 모델 용량의 증가 (Model Capacity)

- 최근 딥러닝 모델들은 레이어 수(depth)와 필터 수(width)가 급격히 증가하여, 학습 데이터를 더 잘 맞출 수 있는 **모델 용량(capacity)** 을 갖추게 되었습니다.
- 하지만 모델 용량이 커질수록 오히려 **confidence가 실제 정확도보다 과도하게 높아지는 과신(overconfidence)** 경향이 나타납니다.

실험 결과 (ResNet on CIFAR-100):
- 깊이(depth)를 증가시키면 ECE도 증가
- 너비(width)를 증가시켜도 ECE 증가

> 모델은 학습 중 NLL(negative log-likelihood)을 계속 줄이기 위해 확률 예측값(confidence)을 점점 더 1에 가깝게 만들게 되며, 이는 과신을 초래합니다.
{: .prompt-tip }

---

## 3.2 Batch Normalization의 영향

- **Batch Normalization(BN)**은 학습을 빠르고 안정적으로 만들어주는 기술로 널리 사용됩니다.
- 하지만, BN을 사용한 모델들은 **정확도는 올라가지만 calibration은 오히려 나빠지는** 현상이 나타납니다.

실험 결과:
- BN을 적용한 ConvNet은 정확도가 약간 올라가지만, ECE는 뚜렷하게 증가

> BN은 내부 분포의 변화를 안정시켜 훈련을 용이하게 하지만, confidence는 더 과도하게 높게 나올 수 있습니다.

---

## 3.3 Weight Decay 감소의 영향

- 전통적으로 weight decay는 과적합을 막기 위한 정규화 방법으로 널리 사용되었습니다.
- 최근에는 BN의 정규화 효과 때문에 weight decay 사용량이 줄어드는 추세입니다.
- 하지만 실험에서는 **weight decay를 증가시킬수록 calibration은 개선되는 경향**을 보입니다.

실험 결과:
- 작은 weight decay → 과신, 높은 ECE
- 적절한 weight decay → 낮은 ECE, 개선된 calibration

> 즉, 정확도를 최대화하는 weight decay 설정과 calibration을 최적화하는 설정은 서로 다를 수 있으며, 정확도는 유지되더라도 confidence는 왜곡될 수 있습니다.

---

## 3.4 NLL 과적합 현상 (Overfitting to NLL)

- 실험에서는 learning rate가 낮아지는 구간에서 test error는 계속 줄어드는 반면, NLL은 다시 증가하는 현상을 보였습니다.
- 이는 모델이 **정확도는 높이지만 confidence가 실제보다 과도한 상태로 학습이 진행되고 있음**을 의미합니다.

예시: ResNet-110 + stochastic depth (on CIFAR-100)
- Epoch 250 이후 learning rate 감소
- 이후 test error는 감소 (29% → 27%)하지만, NLL은 증가

> 정확도(0/1 loss)는 줄지만, NLL에 대한 과적합으로 인해 **confidence가 비정상적으로 높아지는 보정 오류**가 발생합니다.




## 📐 Calibration의 정의 및 측정 방법

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
