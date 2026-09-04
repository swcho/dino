# codistillation과 DINO의 차이

## 카드 요약

**Q.** codistillation과 DINO의 차이는?

**A.** codistillation은 student와 teacher가 같은 아키텍처를 쓰며 학습 중 distillation을 하지만, teacher도 student로부터 증류를 받는다. DINO에서는 teacher가 student의 평균(EMA)으로만 갱신된다.

DINO 논문(Caron et al., 2021) Related Work의 원문은 이렇다.

> Finally, our work is also related to *codistillation* [1] where student and teacher have the same architecture and use distillation during training. However, the teacher in *codistillation* is also distilling from the student, while it is updated with an average of the student in our work.

즉 **"같은 아키텍처 + 학습 중 distillation"이라는 겉모습은 공유하지만, teacher가 무엇으로부터 갱신되는가가 정반대**다. codistillation의 teacher는 *자기 자신의 손실을 최소화하며 상대로부터 배우는* 또 하나의 학습 주체이고, DINO의 teacher는 *스스로 학습하지 않고 student의 가중치 평균으로만 존재하는* 파생물이다.

---

## 1. codistillation(Anil et al., 2018)이란

원 논문은 Rohan Anil, Gabriel Pereyra, Alexandre Passos, Robert Ormandi, George E. Dahl, Geoffrey E. Hinton, *"Large scale distributed neural network training through online distillation"* (arXiv:1804.03235, ICLR 2018). DINO 참고문헌 [1]이다.

### 원래 목적: 표현 학습이 아니라 **분산 학습 가속**

배경 문제는 "GPU/머신을 더 붙여도 학습이 더 빨라지지 않는 지점"이다. 논문 표현으로 "as the number of machines increases, there are diminishing improvements to the time needed to train a high quality model, to a point where adding workers does not further improve training time." 실제로 Common Crawl 언어모델 실험에서 워커를 32 → 64 → 128까지 늘릴 때는 스텝 수가 줄었지만 **256에서는 추가 이득이 전혀 없었고** wall-clock으로는 오히려 손해였다. 저자들은 "현실적인 세팅에서 100 GPU 워커를 크게 넘어 확장하는 건 매우 어렵다"고 적는다.

codistillation은 이 벽을 넘기 위한 방법이다. 남는 계산 자원을 "더 큰 배치"가 아니라 "**여러 개의 독립 모델 복제본(replica)**"에 쓴다. 각 복제본은 자기 SGD를 돌리되, 주기적으로 서로의 예측을 맞춰간다. 통신은 매 스텝의 gradient가 아니라 **가끔씩 파일시스템에 떨어뜨리는 체크포인트 가중치**뿐이다. 논문의 정리: "In each iteration of synchronous/asynchronous distributed SGD, each worker needs to send and receive an amount of information proportional to the entire model size. When using codistillation to distribute training each worker only needs to very rarely read parameter checkpoints from the other models."

그리고 결정적으로 **예측(prediction)은 gradient보다 낡아도 훨씬 잘 견딘다.** 가중치나 gradient는 스케일 재조정·hidden unit 순열·특징 공간 회전 때문에 "통계적으로 식별 불가능(not statistically identifiable)"해서 서로 조금만 어긋나도 평균이 무의미해지지만, 출력 유닛은 손실과 데이터가 강제하는 명확하고 일관된 의미를 갖는다. 실제로 비동기에서는 **수만 스텝 낡은 예측**, 대배치 동기에서는 50스텝(80만 샘플) 낡은 예측을 써도 최종 품질에 거의 영향이 없었다.

또한 이름의 "online"이 가리키듯, 기존 distillation의 2단계(① teacher/앙상블을 끝까지 학습 → ② 그 teacher로 student를 증류) 파이프라인을 **1단계로 합친다**. 여러 모델을 앙상블한 것과 비슷한 품질 이득을 단일 모델 추론 비용으로, 그리고 더 짧은 시간에 얻는 것이 목표다(Common Crawl에서 2-way codistillation은 baseline과 같은 오차에 **2배 적은 스텝**으로 도달; 2단계 앙상블 증류는 총 27K 스텝이 필요한 데 비해 codistillation은 10K 스텝).

### 목적함수와 gradient 흐름

Algorithm 1을 그대로 옮기면, 복제본 $i$ 의 업데이트는 다음 손실에 대한 $\nabla_{\theta_i}$ 다.

$$
\mathcal{L}_i \;=\; \underbrace{\varphi\big(y,\; F(\theta_i, x)\big)}_{\text{정답 레이블 손실}} \;+\; \underbrace{\psi\!\left(\frac{1}{N-1}\sum_{j \neq i} F(\theta_j, x),\;\; F(\theta_i, x)\right)}_{\text{다른 복제본들의 평균 예측을 따라가는 항}}
$$

여기서 핵심 세 가지.

1. **레이블 $y$가 필요하다.** codistillation은 지도학습 세팅이다(원 논문 실험은 Criteo CTR, ImageNet, Common Crawl 언어모델). 정답 손실 $\varphi$ 가 항상 켜져 있고, distillation 항은 그것을 **대체하는 게 아니라 위에 더해진다**. $\psi$ 로는 cross entropy를 쓰며 teacher 분포를 soft target으로 취급한다.
2. **teacher 항으로는 gradient가 흐르지 않는다.** 업데이트는 $\nabla_{\theta_i}$ 뿐이고 teacher 항은 *다른* 복제본의 파라미터 $\theta_j$ 로만 매개되므로 $\theta_i$ 에 대한 의존이 없어 상수 soft target으로 들어간다. 게다가 그 $\theta_j$ 는 다른 워커가 떨어뜨린 **낡은 체크포인트를 로컬 메모리에 읽어들인 것**이라 애초에 최적화 변수가 아니다. (주의: 원 논문은 "stop-gradient"라는 용어를 쓰지 않는다. 이건 용어가 아니라 Algorithm 1의 구조에서 따라 나오는 성질이다.)
3. **그럼에도 관계는 대칭이다.** 완전히 같은 형태의 손실 $\mathcal{L}_j$ 가 다른 워커에서 동시에 돌기 때문이다. 논문 문장: "the distinction between teacher and student is unnecessary and two or more models all distilling from each other can also be useful", 그리고 "the key characteristic of codistillation is **the simultaneous training of a model and its teacher**."

즉 $i$가 $j$에게 배우고 **동시에** $j$도 $i$에게 배운다. 각 복제본은 상대에게는 teacher, 자신에게는 student다. 역할이 고정되어 있지 않고, 정보는 **양방향으로** 흐른다. 이것이 DINO 논문이 말한 "the teacher in codistillation is also distilling from the student"의 의미다. 논문도 이를 장점으로 명시한다 — "The codistillation protocol simplifies the choice of teacher model and **restores symmetry** between the various models."

실무적 디테일 몇 가지.

- **burn-in**: 학습 초기에는 예측이 무의미하거나 오히려 해로우므로 distillation 항을 끄고 정답 손실만 돌리다가, $n_{\text{burn-in}}$ 스텝 이후 **on/off 스위치로 한 번에 켠다**(ImageNet은 3000스텝 후). 램프업 스케줄을 일부러 쓰지 않는다.
- **체크포인트 교환 주기**: 50 / 100 / 250 스텝을 스윕했고 50스텝 간격이 가장 좋았으며 그보다 늘리면 학습 곡선이 약간 나빠진다.
- **복제본 수**: 실험은 대부분 $N=2$ 이고, 각 복제본이 128 GPU 동기 SGD 그룹이라 총 256 GPU — 즉 순수 256 GPU 동기 SGD가 이득을 못 내던 바로 그 지점이다.
- **데이터 분할이 중요**: 두 복제본에 같은 데이터를 주면 baseline보다 조금 나은 정도지만, **서로 다른 부분집합**을 주면 훨씬 좋아진다. 서로 다른 데이터에 대한 정보를 실제로 주고받고 있다는 증거다.
- **두 번째 주장(재현성)**: 재학습 간 예측 churn을 35% 줄여 앙상블과 비슷한 재현성을 서빙 비용 증가 없이 얻는다.

---

## 2. DINO의 구조

![DINO 구조: student/teacher, centering, sg, ema](fig-1.jpeg)

Figure 2의 도식에 카드의 답이 그대로 그려져 있다. 그림에서 읽어야 할 화살표는 넷이다.

- **입력 분기**: 하나의 이미지 $x$에서 두 개의 서로 다른 증강 view $x_1, x_2$를 만들어 각각 student $g_{\theta_s}$, teacher $g_{\theta_t}$ 에 넣는다. 두 네트워크는 **정확히 동일한 아키텍처**(predictor조차 없다)에 파라미터만 다르다 — 여기까지가 codistillation과 닮은 부분이다.
- **teacher 쪽 `sg` 표시**: teacher 출력 $p_2$ 로 가는 경로에 이중 빗금(stop-gradient)이 그려져 있다. 손실 $-p_2 \log p_1$ 의 gradient는 **오직 student 가지로만** 흐른다.
- **teacher 쪽에만 있는 `centering` 박스**: student 경로에는 softmax만 있는데 teacher 경로에는 softmax 앞에 centering이 하나 더 붙어 있다. 구조가 좌우 비대칭이라는 시각적 증거다.
- **`ema` 화살표**: student → teacher **한 방향**으로만 그려져 있다. teacher에서 student로 되돌아가는 화살표는 그림 어디에도 없다.

의사코드로 보면 더 분명하다.

```python
loss = H(t1, s2)/2 + H(t2, s1)/2
loss.backward()          # gradient는 student만 받는다
update(gs)               # SGD
gt.params = l*gt.params + (1-l)*gs.params   # teacher는 EMA로만 갱신
C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach()                     # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1)  # center + sharpen
    return -(t * log(s)).sum(dim=1).mean()
```

teacher 갱신식은 momentum encoder의 EMA다.

$$
\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\,\theta_s
$$

$\lambda$ 는 학습 동안 0.996에서 1까지 코사인 스케줄로 올라간다. **teacher는 자기 손실을 갖지 않는다.** $\theta_t$ 를 낮추는 목적함수가 존재하지 않고, 오직 $\theta_s$ 의 과거 궤적에 대한 결정론적 함수로 정의된다. 논문은 이를 지수 감쇠를 곁들인 **Polyak–Ruppert 평균**, 즉 학습 도중 계속 갱신되는 모델 앙상블로 해석한다.

---

## 3. 세 축의 대비

### 축 1 — 정보 흐름의 대칭성 / 비대칭성

| | codistillation | DINO |
|---|---|---|
| 역할 | 고정되지 않음. 모든 복제본이 서로에게 teacher이자 student | 고정됨. teacher와 student가 구분된 역할 |
| gradient | 각 복제본이 **자기 손실로 SGD 업데이트**를 받음 | student만 SGD, teacher로는 gradient가 흐르지 않음(`sg`) |
| teacher 항에 gradient가 안 가는 이유 | 구조적 귀결. teacher는 다른 워커의 **낡은 체크포인트**라 애초에 최적화 변수가 아님. 반대 방향 손실이 대칭을 복원 | 설계된 단방향성. 반대 방향 손실 자체가 존재하지 않음 |
| teacher 파라미터의 출처 | **독립적인 SGD 학습**(자기 손실 있음) | student의 EMA $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ (자기 손실 없음) |
| 두 브랜치가 보는 입력 | 같은 데이터셋의 **서로 다른 샤드**(분할이 클수록 유리) | 같은 이미지의 **서로 다른 증강 view**(multi-crop, local-to-global) |
| 정보 흐름 | 양방향, 두 개의 궤적이 서로 끌어당김 | 단방향, 하나의 궤적과 그 시간 평균 |

여기서 가장 오해하기 쉬운 지점: **양쪽 다 teacher 출력을 상수 취급한다.** 그래서 "DINO만 gradient를 막는다"고 외우면 틀린다. 차이는 gradient가 막힌 대상이 **자기 손실을 갖고 따로 SGD로 학습되는 살아 있는 모델인가(codistillation)**, 아니면 **학습 대상이 아예 아닌 파생 가중치인가(DINO)** 다. codistillation에서 그 차단은 대칭 손실 쌍에 의해 반대편에서 상쇄되지만, DINO에는 상쇄해 줄 반대편이 없다.

DINO에서는 애초에 궤적이 하나뿐이다. teacher는 두 번째 학습 주체가 아니라 student 궤적의 시간 평균일 뿐이므로, 정보는 student → (평균) → teacher → (타깃) → student 로 순환하되 **파라미터 갱신 방향은 언제나 student에서 teacher로만** 흐른다.

![momentum teacher가 student를 앞서고, teacher 선택이 성능을 가른다](fig-2.jpeg)

Figure 6이 이 비대칭이 왜 이득인지 보여준다. **왼쪽**: momentum teacher(주황)가 학습 내내 student(파랑)보다 위에 있다. 평균이 개별 iterate보다 좋은 모델이라는 Polyak–Ruppert 효과 덕에, student는 항상 자기보다 나은 타깃을 쫓게 된다. **오른쪽**: teacher 구성 방식별 top-1을 보면 momentum 72.8, previous epoch 66.6인데 반해 **student copy와 previous iteration은 0.1로 붕괴**한다. teacher가 student와 너무 가까워져 대칭에 근접하는 순간 학습이 망가진다는 뜻이고, 이것이 DINO가 대칭을 굳이 피하는 이유다.

### 축 2 — 목적

| | codistillation | DINO |
|---|---|---|
| 풀려는 문제 | **분산 학습 가속·확장성**. 배치를 더 못 키우는 지점 이후에도 자원을 유용하게 쓰기 | **자기지도 표현 학습**. 레이블 없이 좋은 feature 얻기 |
| 레이블 | 필요(지도학습). distillation은 정답 손실 위의 보조 항 | 불필요. distillation 항이 **유일한** 목적함수 |
| 통신 | 간헐적 가중치 브로드캐스트로 all-reduce gradient 통신 대체 | 통신 절감이 목적이 아님. 두 네트워크는 같은 워커에 공존 |
| 얻으려는 것 | 앙상블급 품질을 단일 모델 추론 비용으로, 더 짧은 시간에(2배 적은 스텝). 덤으로 재학습 간 예측 churn 35% 감소 | 다운스트림에 쓸 backbone $f$ 의 표현. k-NN·linear probe·attention map |
| 최종 산출물 | 복제본 중 **아무거나 하나**를 골라 서빙 | student가 아니라 **teacher(EMA 가중치)**를 배포 |
| distillation을 보는 관점 | 사전학습 후 압축 단계가 아니라 **학습 중 온라인** 절차 | 사전학습 후처리가 아니라 **자기지도 목적함수 그 자체**로 캐스팅 |

DINO 논문의 표현으로는 "knowledge distillation, instead of being used as a post-processing step to self-supervised pre-training, is directly cast as a self-supervised objective." 그래서 DINO는 스스로를 **self-distillation with no labels**라고 부른다.

같은 Related Work 문단에서 대비되는 다른 축도 함께 기억해 두면 좋다. 기존 SSL+KD 연구들은 **미리 학습된 고정 teacher**를 쓰는데, DINO의 teacher는 학습 중 동적으로 만들어진다. codistillation의 teacher도 동적이지만 그 동적임의 정체가 다르다 — codistillation은 "따로 학습되는 동료", DINO는 "내 과거의 평균"이다.

### 축 3 — 붕괴(collapse) 위험과 방지 장치

이 축이 두 방법의 실질적 차이를 가장 크게 만든다.

**codistillation은 붕괴가 구조적으로 큰 문제가 아니다.** 정답 레이블 손실 $\varphi(y, F(\theta_i,x))$ 이 항상 켜져 있어서 출력을 입력과 무관한 상수로 만들면 그 항이 폭발한다. 레이블이 자연스러운 앵커 역할을 하므로, 두 복제본이 "아무 입력에나 똑같은 상수 출력"으로 합의하는 자명한 해에 빠지지 않는다. 그래서 codistillation은 대칭 구조를 그대로 둬도 된다 — 실무적으로 필요한 건 초반 burn-in 정도이고, 그 목적도 붕괴 방지가 아니라 **모델 다양성을 초반에 더 오래 유지**하려는 것이다. (논문이 경계하는 실패 모드는 붕괴가 아니라 그 반대다: teacher가 학습셋에 과적합하면 레이블 이상의 정보를 주지 못해 distillation이 무의미해진다 — "For distillation to provide value, the teacher must provide information beyond the training label.")

**DINO는 레이블이 없으므로 앵커가 없다.** 목적함수가 "teacher 출력과 student 출력을 맞춰라"뿐이면 두 네트워크가 입력과 무관한 동일 상수를 뱉는 것이 완벽한 최적해다. 그래서 DINO는 붕괴 방지 장치를 명시적으로 넣는다. 논문은 붕괴를 두 형태로 나눈다.

- **한 차원 지배(one dimension dominates)**: 출력이 항상 특정 차원 하나로 쏠림
- **균일 분포(uniform)**: 출력이 항상 모든 차원에 고르게 퍼짐

이를 각각 잡는 두 연산이 **centering**과 **sharpening**이다.

- **Centering**: teacher 출력에 배치 평균 기반 bias $c$ 를 더한다. $g_t(x) \leftarrow g_t(x) + c$, 그리고 $c \leftarrow m c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$. **한 차원 지배를 막지만, 균일 분포로의 붕괴를 부추긴다.** 1차 배치 통계에만 의존하므로 배치 크기 의존성이 작다(batch size 8까지도 동작).
- **Sharpening**: teacher softmax 온도 $\tau_t$ 를 낮게 준다(0.04 → 0.07 warm-up, 0.06 초과 시 붕괴). **균일 분포를 막지만, 한 차원 지배를 부추긴다.**

둘은 서로 반대 방향으로 밀기 때문에 **함께 써야 균형이 잡힌다.** 논문은 교차 엔트로피를 분해해 이를 진단한다.

$$
H(P_t, P_s) = h(P_t) + D_{KL}(P_t \,\|\, P_s)
$$

![centering/sharpening 각각 단독일 때의 붕괴](fig-3.jpeg)

Figure 7이 그 실험이다. **오른쪽(KL divergence)**: sharpening만(파랑), centering만(빨강) 쓴 경우 KL이 0으로 수렴한다 — 출력이 입력과 무관한 상수가 되었다는 뜻, 즉 붕괴다. 둘 다 쓴 경우(주황)만 KL이 0보다 유의미하게 떠 있다. **왼쪽(target entropy)**: 붕괴의 *종류*가 갈린다. centering 없이 sharpening만 쓰면 엔트로피가 **0**으로(한 차원 지배), sharpening 없이 centering만 쓰면 $-\log(1/K)$, 즉 그래프 상단의 평평한 선으로(균일 분포) 간다. 둘을 함께 쓴 주황 곡선만 그 사이 중간값에 안착한다.

여기에 **momentum teacher 자체도 붕괴 방지에 기여한다**는 점이 중요하다. Appendix의 SwAV 비교 표(Table 15)를 보면, momentum이 있을 때는 centering만으로도 76.1을 얻지만 momentum을 student의 stop-gradient 복사본으로 바꾸면 centering만으로는 **0.1로 붕괴**하고 Sinkhorn-Knopp 같은 더 강한 연산이 필요해진다. 즉 EMA로 만든 비대칭성 자체가 안정화 장치의 일부다. 대칭을 포기한 대가로 얻은 것이 안정성인 셈이다.

---

## 4. 한 줄 정리

| | codistillation | DINO |
|---|---|---|
| 아키텍처 | 동일 | 동일 |
| 학습 중 distillation | 예 | 예 |
| **teacher가 학습되는 방식** | **자기 손실로 SGD (student로부터도 증류받음)** | **student의 EMA, gradient 없음** |
| 대칭성 | 대칭(양방향) | 비대칭(단방향) |
| 목적 | 분산 학습 가속 | 자기지도 표현 학습 |
| 레이블 | 필요 | 불필요 |
| 붕괴 방지 | 정답 손실이 앵커 | centering + sharpening + momentum teacher |

암기용 문장: **"codistillation은 서로 배우는 동료 둘, DINO는 나와 내 과거 평균."**

---

## 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021 — arXiv:2104.14294. §2 Related work, §3.1 Approach, §5.2–5.3, Appendix D/E.
- Anil, Pereyra, Passos, Ormandi, Dahl, Hinton, *Large scale distributed neural network training through online distillation*, ICLR 2018 — arXiv:1804.03235. (DINO 참고문헌 [1]) Algorithm 1, §2.1(분산 프로토콜), §3.2–3.4.
- Zhang et al., *Deep Mutual Learning* (2017) — codistillation과 거의 동일한 알고리즘이 독립적으로 제안된 사례. Anil et al.은 이를 인정하면서도 DML이 보고한 online distillation의 품질 우위는 teacher 체크포인트 과적합 artifact였다고 반박하고, 자신들의 기여를 "품질"이 아니라 **지연에 강한 분산 학습 알고리즘**으로 위치시킨다. 이 위치 설정 자체가 DINO(표현 학습)와의 목적 차이를 잘 보여준다.
- Tarvainen & Valpola, *Mean teachers are better role models* (Mean Teacher) — DINO의 EMA teacher 해석의 뿌리.
- Grill et al., *BYOL* — DINO가 직접 영감을 받았다고 밝힌 방법. 다만 BYOL은 predictor가 붕괴 방지에 필수인 반면 DINO는 predictor 없이 centering + sharpening으로 해결한다.
