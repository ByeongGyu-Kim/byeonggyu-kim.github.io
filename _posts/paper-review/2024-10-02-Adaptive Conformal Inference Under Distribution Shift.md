---
title: "Adaptive Conformal Inference Under Distribution Shift"
date: 2025-02-02 00:11:35 +0900
categories:
  - Paper-Review
  - Trustworthy Machine Learning
tags:
  - Conformal prediction
  - Adaptive Conformal Inference(ACI)
  - Distribution shift
description: 
toc: true
comments: false
cdn: 
image:
math: true
pin: false
mermaid: false
---
# Summary
df

## 1. Introduction

현대 사회에서는 **Machine Learning 알고리즘**은 점점 더 많은 분야에서 중요한 의사결정을 하는데 활용되고 있습니다. 
예를 들어, 자율주행 자동차의 딥러닝 기반 모델은 주변 사물을 인식하고 이를 기반으로 어떻게 주행을 이어갈 지 결정하게 되고,
사법 시스템에서는 재범 가능성을 예측하기 위해 100개 이상의 특징을 결합한 복잡한 모델이 가석방 여부를 결정하는 등 다양한 분야에서 활용되고 있습니다.
이러한 블랙박스(black-box) 모델의 인기가 증가함에 따라, 잘못된 예측이 초래할 수 있는 비용도 커지고 있습니다. 잘못된 판단은 안전사고나 부당한 판결로 이어질 수 있기 때문에, 이러한 위험을 최소화하기 위해 모델의 예측에 대한 **불확실성(uncertainty)**을 정량화하는 도구 개발이 필수적입니다.  

본 논문은 **온라인 학습(online learning)** 환경에서 목표 값을 높은 확률로 포함하는 **예측 집합(prediction set)**을 구성하는 방법을 다루고 있습니다.
이 환경에서는 모델이 순차적으로 $$(X_t, Y_t)$$ 형태의 데이터(공변량-반응 쌍)를 받으며, 이전에 관측된 데이터와 새로운 공변량 $$X_t$$를 바탕으로 $$Y_t$$의 값을 포함하는 예측 집합 $$\hat{C}_t$$를 만들어야 합니다.
이때, $$Y_t$$가 $$\hat{C}_t$$에 포함될 확률이 최소 $$100(1 - \alpha)\%$$가 되도록 보장하는 것이 목표입니다. 여기서 $$\alpha$$는 사전에 설정한 신뢰 수준입니다.
이 문제를 해결하기 위해 논문은 컨포멀 추론(conformal inference) 기법을 사용합니다. 이 기법은 어떤 블랙박스 예측 알고리즘의 출력이라도 예측 집합으로 변환할 수 있는 일반적인 방법론을 제공합니다.
전통적인 컨포멀 추론은 교환성(exchangeability), 즉 데이터가 동일한 분포를 따르고 독립적이라는 가정을 필요로 합니다.
그러나 실제 세계의 많은 상황에서는 이러한 가정이 성립하지 않습니다. 예를 들어, 금융 시장은 새로운 법률 제정이나 세계적인 사건에 따라 시장 행동이 급격히 변하기 때문에 **비정상(non-stationary)**적인 데이터 분포가 발생합니다.
이를 해결하기 위해 논문은 **적응형 컨포멀 추론(adaptive conformal inference)**을 제안합니다. 이 방법은 시간에 따라 변화하는 데이터 분포를 고려하여 예측 집합을 조정하며, 데이터 생성 과정에 대한 가정 없이도 신뢰할 수 있는 예측 성능을 유지할 수 있습니다​

## 2. Conformal Inference
컨포멀 추론은 주어진 회귀 모델을 통해  값을 예측할 때, 그 예측값이 얼마나 신뢰할 수 있는지를 평가하는 방법입니다. 이를 위해 적합도 점수(conformity score) 를 정의하여, 새로운 데이터 포인트가 기존 데이터와 얼마나 일치하는지 측정합니다.

## 2-1. 적합도 점수 정의

회귀 모델의 예측값을 라고 할 때, 에 대한 후보 값 가 얼마나 잘 맞는지를 측정하는 적합도 점수는 다음과 같이 정의할 수 있습니다.



또는, 회귀 모델이 의 번째 분위수(quantile)를 예측할 수 있다면, **컨포멀 분위수 회귀(Conformal Quantile Regression, CQR)**를 사용하여 다음과 같이 정의할 수 있습니다.



여기서 는 의 번째 분위수에 대한 추정값을 의미합니다.

1.1.2 예측 집합 구성

적합도 점수 를 이용해 예측 집합을 만들기 위해, 보정 집합(calibration set) 을 사용합니다. 이 보정 집합을 통해 적합도 점수의 분위수를 다음과 같이 정의합니다.



이때, 는 인디케이터 함수로, 조건이 참일 때 1, 그렇지 않으면 0을 반환합니다.

최종적으로 예측 집합 는 다음과 같이 정의됩니다.



이 예측 집합은 마진 커버리지(marginal coverage) 보장을 제공합니다.



1.1.3 교환성 가정의 한계

위 방식은 데이터가 **교환 가능(exchangeable)**하다는 가정 하에 유효합니다. 즉, 훈련 데이터와 테스트 데이터가 동일한 분포에서 독립적으로 샘플링된 경우에만 신뢰할 수 있습니다. 그러나 실제 응용에서는 데이터의 분포가 시간이 지남에 따라 변할 수 있습니다. 예를 들어:

금융 시장: 새로운 법률이나 글로벌 이벤트로 인해 시장 행동이 급격히 변할 수 있습니다.

환경 변화: 모델이 새로운 환경에 배치되면서 데이터 분포가 변할 수 있습니다.


![Desktop View](/assets/img/paper-review/aci/fig1.1.png)
_Figure 1.1: Example of different representations_