---
title: "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"
date: 2025-02-04 18:00:00 +0900
categories:
  - Paper-Review
  - Vision Language Action
tags:
  - Lifelong Learning
description: 
toc: true
comments: false
cdn: 
image:
math: true
pin: false
mermaid: false
---
# 연구 개요
본 논문은 로봇 조작 작업에서 **평생 학습(Lifelong Learning in Decision Making, LLDM)** 연구를 체계적으로 수행할 수 있도록 설계된 새로운 벤치마크 **LIBERO**를 소개한다.  
LIBERO는 기존의 이미지 및 텍스트 중심의 평생 학습 연구와 달리, **절차적(procedural) 지식과 선언적(declarative) 지식**을 모두 포함하는 **의사 결정 학습(Decision-Making Learning)** 에 중점을 둔다.

## LLDM의 5가지 주요 연구 주제
본 논문은 LLDM에서 중요한 5가지 연구 주제를 탐구한다.

- **지식 전이(knowledge transfer)와 다양한 분포 변화(distribution shift)에 대한 연구**
- **신경망 아키텍처 설계(neural architecture design)**
- **평생 학습 알고리즘(lifelong learning algorithm) 설계**
- **학습자의 작업 순서(task ordering)에 대한 견고성(robustness)**
- **사전 학습된(pre-trained) 모델을 LLDM에서 어떻게 활용할 것인가?**

---

## 실험 결과

### 1. 정책 아키텍처 설계(policy architecture design)의 중요성
- **Transformer 아키텍처**는 **시간적 정보(temporal information) 추상화**에서 **순환 신경망(Recurrent Neural Network, RNN)** 보다 우수하다.
- **Vision Transformer(ViT)** 는 **다양한 객체(object)가 포함된 시각적 정보가 풍부한 작업**에서 효과적이다.
- **합성곱 신경망(Convolutional Neural Networks, CNNs)** 은 **절차적 지식(procedural knowledge)** 을 중심으로 하는 작업에서 더 좋은 성능을 보인다.

### 2. 평생 학습 알고리즘의 전이 학습 성능 문제
- 평생 학습 알고리즘은 **망각(forgetting) 방지**에는 효과적이지만, **순차적 미세 조정(sequential finetuning)** 보다 **전방 전이(forward transfer) 성능**이 낮다.
- 기존 평생 학습 알고리즘이 **기존 작업을 유지하는 능력**은 우수하지만, **새로운 작업에 대한 전이 학습 능력**은 떨어진다.

### 3. 사전 학습된 언어 임베딩(pretrained language embeddings) 사용 결과
- **의미적으로 풍부한 작업 설명(semantically-rich task descriptions)** 을 **사전 학습된 언어 임베딩**과 함께 사용하는 것은 **단순한 작업 ID(task ID)** 만 사용하는 것보다 성능이 향상되지 않았다.

### 4. 지도 학습(supervised pretraining)이 LLDM 성능에 미치는 영향
- **대규모 오프라인 데이터셋**에서의 **기본적인 지도 학습(supervised pretraining)** 은 **오히려 학습자의 성능을 저하시킬 수 있다.**
- 단순한 지도 학습을 통한 사전 학습이 이후 LLDM 작업에서 학습자의 성능에 부정적인 영향을 미칠 수 있음을 확인했다.

