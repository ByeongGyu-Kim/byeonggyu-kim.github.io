---
title: "An Experimental Study of GPS Spoofing and Takeover Attacks on UAVs"
date: 2025-02-04 18:00:00 +0900
categories:
  - [Paper-Review, Security]
tags:
  - Conformal prediction
  - GPS
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
본 논문은 상용 무인항공기(UAV)가 GPS 스푸핑 공격에 얼마나 취약한지를 실험적으로 분석하고, UAV를 완전히 제어하기 위한 조건과 전략을 제시합니다. 연구팀은 실시간 GPS 신호 생성기(RtGSG)를 개발하여 UAV의 위치와 속도를 조작하는 실험을 수행했으며, 고정된 위치와 동적 경로 스푸핑이 UAV에 미치는 영향을 평가했습니다. 
실험 결과, UAV는 GPS 외에도 비전 센서나 관성 센서를 활용하여 일부 스푸핑 공격을 방어할 수 있었지만, 정교한 실시간 신호 조작을 통해 UAV의 방향과 속도를 제어하는 것이 가능함을 입증했습니다. 
다만, 고도 제어는 어려워 No-Fly Zone 유도나 센서 융합 오류를 활용한 강제 착륙이 필요했습니다. 
이 연구는 UAV 보안 취약점을 드러내는 동시에, 방어 전략 개발의 필요성을 강조합니다.