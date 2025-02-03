---
title: "Adaptive Conformal Inference Under Distribution Shift"
date: 2025-02-02 00:11:35 +0900
categories:
  - paper-review
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
# Introduction

현대 사회에서 **Machine Learning 알고리즘**은 점점 더 많은 분야에서 중요한 의사결정을 하는데 활용되고 있습니다. 
예를 들어, 자율주행 자동차의 딥러닝 기반 모델은 주변 사물을 인식하고 이를 기반으로 어떻게 주행을 이어갈 지 결정하게 되고,
사법 시스템에서는 재범 가능성을 예측하기 위해 100개 이상의 특징을 결합한 복잡한 모델이 가석방 여부를 결정하는 등 다양한 분야에서 활용되고 있습니다.
이러한 블랙박스(black-box) 모델의 인기가 증가함에 따라, 잘못된 예측이 초래할 수 있는 비용도 커지고 있습니다. 잘못된 판단은 안전사고나 부당한 판결로 이어질 수 있기 때문에, 이러한 위험을 최소화하기 위해 모델의 예측에 대한 **불확실성(uncertainty)**을 정량화하는 도구 개발이 필수적입니다.  
본 논문은 **온라인 학습(online learning)** 환경에서 목표 값을 높은 확률로 포함하는 **예측 집합(prediction set)**을 구성하는 방법을 다루고 있습니다.
이 환경에서는 모델이 순차적으로 $$(X_t, Y_t)$$ 형태의 데이터(공변량-반응 쌍)를 받으며, 이전에 관측된 데이터와 새로운 공변량 $$X_t$$를 바탕으로 $$Y_t$$의 값을 포함하는 예측 집합 $$\hat{C}^t$$를 만들어야 합니다.
이때, $$Y_t$$가 $$\hat{C}_t$$에 포함될 확률이 최소 $$100(1 - \alpha)\%$$가 되도록 보장하는 것이 목표입니다. 여기서 $$\alpha$$는 사전에 설정한 신뢰 수준입니다.
이 문제를 해결하기 위해 논문은 컨포멀 추론(conformal inference) 기법을 사용합니다. 이 기법은 어떤 블랙박스 예측 알고리즘의 출력이라도 예측 집합으로 변환할 수 있는 일반적인 방법론을 제공합니다.
전통적인 컨포멀 추론은 교환성(exchangeability), 즉 데이터가 동일한 분포를 따르고 독립적이라는 가정을 필요로 합니다.
그러나 실제 세계의 많은 상황에서는 이러한 가정이 성립하지 않습니다. 예를 들어, 금융 시장은 새로운 법률 제정이나 세계적인 사건에 따라 시장 행동이 급격히 변하기 때문에 **비정상(non-stationary)**적인 데이터 분포가 발생합니다.
이를 해결하기 위해 논문은 **적응형 컨포멀 추론(adaptive conformal inference)**을 제안합니다. 이 방법은 시간에 따라 변화하는 데이터 분포를 고려하여 예측 집합을 조정하며, 데이터 생성 과정에 대한 가정 없이도 신뢰할 수 있는 예측 성능을 유지할 수 있습니다​

## Conformal Inference
The key is to get informal knowledge about the world into a computer.
One approach, called **knowledge base**, have sought to hard-code knowledge about the world in formal languages... but none has led to a major success.
(e.g., Cyc(Lenat and Guha, 1989) do not capture the person Fred while shaving.)

### Machine learning

Thus, the capability to acquire their **own** knowledge, by **extracting patterns from raw data**, is needed. a.k.a **machine learning**.

#### Simple machine learning algorithms

The famous & simple ML algorithms are **logistic regression**, which can determine whether to recommend cesarean delivery, and **naive Bayes**, which can separate legitimate e-mail from spam.
However, these simple ML algorithms depend heavily on the **representation**(or **feature** in it) of the data, so they cannot influence how features are defined in any way.



#### Representation learning

One solution to this problem is **representation learning**.
This discovers not only the mapping but also the representation itself.
It allows AI systems to rapidly adapt to new tasks, with minimal human intervention.
The quintessential[^fn-nth-2] example of a representation learning algorithm is the **autoencoder**[^fn-nth-3].

When designing features or algorithms for learning features, our goal is usually to separate the **factors[^fn-nth-4] of variation**, which include the speaker's age, their sex, their accent and the words they are speaking.
The major source of difficulty in many real-world AI applications comes from these factors of variation.

## Deep learning


Of course, it is difficult to extract such high-level & abstract features, such as a speaker's accent, from raw data.
A deep learning system can represent the concept of an image of a person by **combining simpler concepts**, such as corners and contours, which are in turn defined in terms of edges.

The quintessential example of a deep learning model is the feedforward deep network, called **multilayer perceptron** (MLP)[^fn-nth-5].

![Desktop View](/assets/img/paper-review/aci/fig1.1.png)
_Figure 1.1: Example of different representations_