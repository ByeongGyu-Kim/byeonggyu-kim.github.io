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

# 📌 연구 개요
**Confidence Calibration**은 모델이 예측한 확률이 실제 정답일 가능성과 얼마나 일치하는지를 나타내는 개념이다. 예를 들어, 어떤 이미지에 대해 모델이 고양이일 확률을 0.9라고 예측했을 때, 이러한 예측이 잘 보정(calibrated)되어 있다면 실제로 그 이미지가 고양이일 확률도 약 90%가 되어야 한다는 것이다.

**On Calibration of Modern Neural Networks** 논문은 빠르게 발전하고 연구되어져 오고 있는 ResNet, DenseNet 등과 같은 현대적인 신경망 모델들이 높은 분류 정확도를 달성함에도 불구하고, 오히려 확률 보정(calibration) 성능은 더 나빠졌다는 사실을 실험적으로 보이고 있다. 과거의 얕은 신경망 모델들은 예측 확률이 실제 정답 확률과 비교적 잘 일치했지만, 깊고 복잡한 구조를 가진 최신 모델들은 자신 있게 예측을 하면서도 그 확률이 실제 정답률과 불일치하는 경향이 있다는 점을 짚고 있다. 한 마디로 말해 최신 모델들은 Overconfident 되어져 있다는 점을 발견하였다.

<!-- ![Desktop View](/assets/img/paper-review/On_Calibration_of_modern_NN/figure1.png)
_Figure 1.1: Confidence histograms (top) and reliability diagrams (bottom) for a 5-layer LeNet (left) and a 110-layer ResNet (right) on CIFAR-100. Refer to the text below for detailed illustration._ -->


이러한 문제는 자율주행, 의료 진단, 법률 판단 등과 같이 예측 결과에 대한 신뢰도가 매우 중요한 응용 분야에서 특히 도드라질 수 있기에 모델이 단순히 정확할 뿐만 아니라, 자신의 예측이 얼마나 확실한지에 대한 표현 또한 신뢰할 수 있어야 한다.



본 논문에서는 이러한 문제를 해결하기 위해 다양한 사후 보정(post-hoc calibration) 방법들을 실험적으로 비교하고, 그 중에서도 Temperature Scaling이라는 단 하나의 스칼라 파라미터만을 사용하는 간단한 방법이 매우 효과적이라는 사실을 밝혀냈다. 본 글에서는 이의 실험 코드도 구현하였다.

