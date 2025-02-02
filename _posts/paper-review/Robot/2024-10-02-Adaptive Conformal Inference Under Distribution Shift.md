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

현대 사회에서 **Machine Learning 알고리즘**은 점점 더 많은 분야에서 중요한 의사결정에 활용되고 있습니다. 
예를 들어, 자율주행 자동차의 딥러닝 기반 모델은 주변 사물을 인식하고 이를 기반으로 주행 결정을 내리게 되고

- **사법 시스템**: 재범 가능성을 예측하기 위해 100개 이상의 특징을 결합한 복잡한 모델이 가석방(parole) 결정을 지원합니다.

이러한 **블랙박스(black-box)** 모델의 인기가 증가함에 따라, **잘못된 예측**이 초래할 수 있는 **비용**도 커지고 있습니다. 
따라서 모델의 예측에 대한 **불확실성(uncertainty)**을 정량화하는 도구 개발이 필수적입니다.


### Hard-code knowledge

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