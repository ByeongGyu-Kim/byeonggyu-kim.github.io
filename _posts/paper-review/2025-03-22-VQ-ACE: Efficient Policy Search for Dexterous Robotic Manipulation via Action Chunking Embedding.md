---
title: "VQ-ACE: Efficient Policy Search for Dexterous Robotic Manipulation via Action Chunking Embedding"
date: 2025-03-22 18:00:00 +0900
categories:
  - Paper-Review
  - Robotics
  - Reinforcement Learning
tags:
  - Dexterous Manipulation
  - Vector Quantization
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
본 논문은 고차원 손 조작 동작, 영어로 dexterous manipulation라고 하는 문제에서의 정책 탐색을 개선하기 위해 제안된 Vector Quantized Action Chunking Embedding, 줄여서 VQ-ACE라는 프레임워크를 소개한다.
VQ-ACE는 사람 손의 복잡한 동작 시퀀스를 벡터 양자화된 이산 latent 공간으로 임베딩하여, 행동 공간의 차원을 줄이면서도 중요한 동작 특성을 유지한다.
이 latent 표현은 **샘플링 기반 모델 예측 제어(MPC)**와 **강화학습(RL)**에 통합되어 정책 탐색의 효율성과 자연스러움을 동시에 향상시킨다.

## Chapter 3. Action Chunking Embedding

본 장에서는 **복잡한 손 조작 동작을 저차원 이산 latent로 임베딩하는 방법과 그 활용**에 대해 설명한다.  
VQ-ACE는 크게 다음 세 부분으로 구성된다.

- **데이터 수집(Data Collection)**
- **벡터 양자화된 액션 청크 임베딩**
- **임베딩을 활용한 두 가지 제어 방법**  
  - Latent Sampling MPC  
  - Action Chunked RL

---

### 3.1 데이터 수집

- **모션 캡처 장갑**을 이용하여 11 DoF 로봇 손에 맞게 변환된 **54분 분량의 사람 손 동작 데이터**를 수집 (50Hz)
- 동작은 일상 물체 조작, 장난감 조작, 케이블 꼬기, 수화 등 다양한 활동 포함

---

### 3.2 Vector-Quantized Action Chunk Embedding

#### ✅ 구조
- **CVAE (Conditional VAE)** 기반 인코더-디코더 구조 사용
- 입력: 현재 관절 상태 \( q_t \), 행동 시퀀스 \( a_{t:t+n} \)
- 인코더 출력: 이산 latent 벡터 \( z_{k:k+m} \) → **벡터 양자화(Nearest Neighbor Look-up)**
- 디코더 입력: \( z_{k:k+m} \), \( q_t \) → 예측된 행동 시퀀스 \( \hat{a}_{t:t+n} \)

#### ✅ 특징
- **Transformer 기반 구조** 사용
- 시계열 정보를 반영하기 위해 **Positional Embedding + Causal Mask** 적용
- 시간 인덱스 t와 latent 인덱스 k 간의 관계를 아래와 같이 정의:

\[
t(k) = \frac{k \cdot n}{m}
\]

#### ✅ 손실 함수 (Loss)
\[
L = L_{\text{recon}} + \lambda_{\text{commit}} L_{\text{commit}}
\]

- \( L_{\text{recon}} \): L1 재구성 손실  
- \( L_{\text{commit}} \): 벡터 양자화된 latent에 대한 커밋 손실 (EMA 방식으로 업데이트)

---

### 3.3 Latent Sampling MPC (예측 기반 샘플링 제어)

- 기존 MPC는 spline 기반 행동을 샘플링하지만, VQ-ACE는 **latent token 조합**을 샘플링하여 행동을 생성
- 새로운 latent 조합 생성 방식:

\[
z^{(i)}_j = \begin{cases}
z_j, & \text{probability } 1-p \\
e_r, & \text{probability } p, \quad r \sim \text{Uniform}(1,K)
\end{cases}
\]

- 샘플링된 latent로부터 디코더를 통해 행동 생성 → **MuJoCo 시뮬레이터에서 평가**
- **성능 우수한 샘플을 기준으로 다음 샘플 생성** (time-shift + noise 추가)

---

### 3.4 Action Chunked Reinforcement Learning

- 강화학습에서 **정책이 선택한 action chunk**를 기반으로 다단계 행동 수행
- 행동 = **청크 + 잔차**:  
  \[
  u(t) = a_t + \delta_t
  \]

- chunk는 특정 조건이 만족되면 갱신되고, 그렇지 않으면 유지
- chunk 선택은 agent의 **상태(xs)** 와 **액션(us)** 공간에 포함됨

#### ✅ chunk selection 메커니즘

- 선택 조건:

\[
\text{triggert} = \mathbb{1} \left\{ \max(xs(t) + us(t)) > 1 \right\}
\]

- 갱신:

\[
A_{t+1} = 
\begin{cases}
\psi(q_t, \arg\max(xs + us)), & \text{if triggered} \\
A_t, & \text{otherwise}
\end{cases}
\]

- chunk selection은 정책의 일부로 포함되어 학습됨 → **효율적인 탐색 가능**

---

이 장의 핵심은, 기존의 고차원 연속 행동 공간을 **이산 latent 토큰 기반의 청크 형태로 단순화**하고, 이를 이용해 **정책 탐색 및 제어를 보다 효율적이고 자연스럽게 수행**할 수 있도록 한 것이다.
