# DINO의 weight decay와 temperature 스케줄

## 한 줄 정답

| 하이퍼파라미터 | 값 | 스케줄 |
|---|---|---|
| weight decay | $0.04 \rightarrow 0.4$ | cosine (증가) |
| student temperature $\tau_s$ | $0.1$ | 고정 |
| teacher temperature $\tau_t$ | $0.04 \rightarrow 0.07$ | 첫 30 epoch 선형 warmup, 이후 0.07 고정 |

논문 §3.2 Implementation details 원문:

> "After this warmup, we decay the learning rate with a cosine schedule. **The weight decay also follows a cosine schedule from 0.04 to 0.4.** The temperature $\tau_s$ is set to 0.1 while we use a **linear warm-up for $\tau_t$ from 0.04 to 0.07 during the first 30 epochs.**"

(참고로 함께 외워두면 좋은 값: optimizer는 AdamW, batch size 1024, lr은 첫 10 epoch linear warmup 후 $lr = 0.0005 \times \text{batchsize}/256$에서 cosine decay, EMA momentum $\lambda$는 $0.996 \rightarrow 1$ cosine.)

---

## 0. 배경: temperature가 DINO에서 하는 일

![DINO 구조: student/teacher softmax와 centering](fig-1.jpeg)

DINO의 손실은 teacher 분포 $P_t$를 타깃으로 한 cross-entropy다.

$$P_s(x)^{(i)} = \frac{\exp\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}, \qquad \min_{\theta_s} H\big(P_t(x), P_s(x)\big)$$

teacher 쪽도 같은 형태이되 centering 후 $\tau_t$로 softmax를 취한다 (의사코드: `t = softmax((t - C) / tpt, dim=1)  # center + sharpen`).

DINO에는 contrastive loss도, predictor도, queue도 없다. 붕괴(collapse)를 막는 장치는 **centering + sharpening 두 개뿐**이고, 그중 sharpening의 강도를 조절하는 유일한 손잡이가 $\tau_t$다. 그래서 $\tau_t$는 "그냥 튜닝된 상수"가 아니라 **학습 안정성의 안전장치**이며, 스케줄이 붙는 이유도 여기에 있다.

- $\tau$가 **작다** → 분포가 뾰족해짐(sharpening). 극단적으로 $\tau \rightarrow 0$이면 `argmax`, 즉 one-hot 하드 타깃.
- $\tau$가 **크다** → 분포가 평평해짐. 극단적으로는 균일 분포.

---

## 1. 왜 $\tau_t$에 warmup이 필요한가

### 붕괴의 두 얼굴과 sharpening의 역할

![Fig. 7 붕괴 연구: target entropy와 KL divergence](fig-2.jpeg)

논문 §5.3은 cross-entropy를 다음처럼 분해해 붕괴를 진단한다.

$$H(P_t, P_s) = h(P_t) + D_{KL}(P_t \,\|\, P_s)$$

$D_{KL} \rightarrow 0$이면 출력이 상수, 즉 붕괴다. 그런데 붕괴는 두 종류다.

- **centering만 있고 sharpening이 없으면**: 엔트로피 $h \rightarrow -\log(1/K)$, 즉 균일 분포로 붕괴.
- **sharpening만 있고 centering이 없으면**: 엔트로피 $h \rightarrow 0$, 즉 한 차원이 지배하는 one-hot 붕괴.

centering은 "균일 쪽으로", sharpening은 "뾰족한 쪽으로" 미는 서로 반대 방향의 힘이고, 둘의 **균형**만이 붕괴를 막는다(Fig. 7의 주황색 곡선만 KL이 0으로 가지 않는다). $\tau_t$는 이 줄다리기에서 sharpening 쪽 힘의 세기다.

### 부록 D "Sharpening" ablation 수치

ViT-S/16, 100 epoch, $k$-NN top-1:

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | 0.08 | **0.04→0.07 (warmup)** |
|---|---|---|---|---|---|---|
| $k$-NN top-1 | 43.9 | 66.7 | **69.6** | 68.7 | **0.1** | **69.7** |

읽는 법:

- **너무 높으면 ($\tau_t = 0.08$) → 완전 붕괴, 0.1%.** 논문 원문: "a temperature lower than 0.06 is required to avoid collapse. When the temperature is higher than 0.06, the training loss consistently converges to $\ln(K)$." 손실이 $\ln(K)$로 수렴한다는 것은 타깃이 균일 분포가 되었다는 뜻이다 — 즉 sharpening이 약해서 centering이 이겨버린 균일 붕괴.
- **너무 낮으면 ($\tau_t = 0$) → 붕괴는 안 하지만 43.9%로 크게 나쁨.** $\tau \rightarrow 0$은 `argmax`, one-hot 하드 타깃이다. 0.02도 66.7%로 손해다. 즉 낮은 $\tau_t$의 대가는 붕괴가 아니라 **성능 저하**(하드 타깃의 정보 손실, 초기 타깃 오류를 그대로 확신 있게 학습).
- **고정 최적값은 0.04 (69.6), 그런데 warmup 0.04→0.07이 69.7로 그보다 낫다.**

### 그래서 warmup의 논리

핵심은 논문의 이 문장이다.

> "we have observed that using higher temperature than 0.06 does not collapse **if we start the training from a smaller value and increase it during the first epochs.**"

즉 0.07은 **처음부터 쓰면 붕괴하지만, 0.04에서 출발해 30 epoch에 걸쳐 올리면 안전하게 쓸 수 있는 값**이다. 정리하면:

1. **학습 초기** — teacher는 아직 랜덤에 가깝고 그 출력은 의미 없는 타깃이다. 이때 필요한 것은 붕괴를 막을 만큼 충분한 sharpening 압력이므로 낮은 값(0.04)에서 시작한다. 동시에 0(하드 argmax)까지 가지는 않으므로, 아직 신뢰할 수 없는 타깃에 student가 one-hot으로 과도하게 확신을 갖고 맞추는 것은 피한다.
2. **학습 후기** — teacher(momentum encoder)가 student보다 꾸준히 좋아지고 타깃이 신뢰할 만해지면, 더 높은 $\tau_t = 0.07$의 **부드러운 타깃**이 유리하다. 부드러운 분포는 클래스/프로토타입 간 유사도 정보(soft label의 "dark knowledge")를 더 많이 담고, centering과의 균형점도 이때는 무너지지 않는다.
3. 반대 순서는 성립하지 않는다. 0.07로 시작하면 sharpening이 약해 즉시 균일 붕괴로 빠지고, 한 번 붕괴하면 되돌아오지 못한다. 그래서 **"안전한 쪽(낮은 $\tau_t$)에서 시작해 유리한 쪽(높은 $\tau_t$)으로 옮겨가는"** 단방향 warmup이 된다.

> 카드 답에 적힌 "초기에는 더 부드러운 타깃으로 시작해 점차 sharpening을 강화"라는 직관은 방향이 반대로 읽히기 쉬우니 주의. 실제 논문 스케줄은 **초기에 더 뾰족(0.04) → 후기에 더 부드럽게(0.07)** 이고, 붕괴 위험이 큰 쪽은 "덜 뾰족한" 쪽이다. warmup의 목적은 "붕괴 안전 구간에서 출발해, 붕괴하지 않고 도달할 수 없었을 더 좋은 값으로 이동하는 것"이다.

---

## 2. 왜 $\tau_s = 0.1 > \tau_t$인 비대칭이 중요한가

$\tau_s = 0.1$은 학습 내내 고정이고, $\tau_t$는 0.04~0.07이므로 **항상 $\tau_t < \tau_s$**, 즉 **teacher 분포가 student 분포보다 언제나 더 뾰족하다.**

이것이 DINO를 "self-distillation"으로 만드는 장치다.

- teacher와 student는 **완전히 같은 아키텍처**이고(predictor 없음), teacher 가중치는 student의 EMA다. 만약 $\tau_t = \tau_s$라면 두 분포는 통계적으로 대칭이어서, "누가 누구를 따라가야 하는가"라는 방향성이 없다 — 손실이 그냥 자기 자신을 맞추는 자명한 해(모든 출력을 같게 만드는 붕괴)로 흐르기 쉽다.
- $\tau_t$를 낮춰 teacher를 더 뾰족하게 만들면, teacher 출력은 student 출력의 **"확신을 높인 버전"**이 된다. cross-entropy $H(P_t, P_s)$의 그래디언트는 student를 이 더 뾰족한 타깃 쪽으로 밀고, 이는 지식 증류/의사 라벨링에서 쓰는 entropy minimization(sharpened target을 향한 self-training)과 같은 효과다.
- 즉 비대칭이 **정보의 흐름 방향(teacher → student)을 만들고**, 동시에 앞 절의 sharpening 압력(균일 붕괴 방지)을 제공한다. teacher가 student보다 덜 뾰족하다면(만약 $\tau_t > \tau_s$) student는 자기보다 흐릿한 타깃을 향해 학습하게 되고, 이는 곧장 균일 붕괴다 — 위 표의 $\tau_t = 0.08$ 행이 그 실험적 증거다.

이 방향성은 결과로도 확인된다. 논문 Fig. 6(left)에서 momentum teacher는 학습 내내 student보다 $k$-NN 정확도가 높다. 논문은 이를 Polyak–Ruppert averaging에 의한 상시 앙상블로 해석하며, "teacher가 항상 더 나은 타깃을 준다 → student가 개선된다 → teacher는 student의 EMA이므로 다시 개선된다"는 부트스트랩 고리를 설명한다. $\tau_t < \tau_s$ 비대칭은 이 고리에 필요한 "teacher가 더 확신에 차 있다"는 조건을 온도 수준에서 보장한다.

---

## 3. 왜 weight decay는 0.04 → 0.4로 **증가**하는 cosine인가

lr은 warmup 후 cosine으로 **감소**하는데 weight decay는 반대로 **증가**한다는 점이 헷갈리기 쉬운 부분이다. (DINO 구현의 `weight_decay=0.04`, `weight_decay_end=0.4`가 이 스케줄이다. optimizer는 AdamW이므로 weight decay가 gradient와 분리(decoupled)되어 lr 스케줄과 독립적으로 걸린다.)

의도는 **규제의 시간적 배분**이다.

- **초기 (wd = 0.04, 약한 규제)** — 아직 아무 표현도 학습되지 않은 단계다. 강한 weight decay는 가중치를 원점으로 끌어당겨 head의 $K$차원 출력이 균일해지도록 압박하는데, 이는 앞서 본 균일 붕괴를 돕는 방향이다. 초기에는 규제를 풀어 네트워크가 자유롭게 특징을 형성하고 prototype들이 서로 분화되도록 둔다. lr warmup(첫 10 epoch)과 $\tau_t$ warmup(첫 30 epoch)이 걸린 구간과도 겹쳐, "초기에는 조심스럽게 표현을 세운다"는 하나의 설계 철학으로 읽힌다.
- **후기 (wd = 0.4, 강한 규제)** — 표현이 자리를 잡은 뒤에는 lr이 cosine으로 작아져 모델이 수렴 국면에 들어간다. 이때 weight decay를 10배로 키워 가중치 norm을 눌러주면, 학습 데이터의 augmentation 특유의 잡음에 과적합하는 것을 억제하고 더 평탄하고 일반화 잘 되는 해로 수렴한다. SSL 표현의 품질은 linear probe / $k$-NN 같은 **downstream 전이**로 측정되므로, 후반의 강한 규제가 직접적으로 평가 지표를 개선한다.
- 즉 **탐색(exploration) → 정규화(consolidation)**의 커리큘럼이다. 0.4는 supervised ImageNet 학습(보통 1e-4 ~ 0.05)에 비하면 매우 큰 값인데, DINO는 라벨이 없어 과적합을 막아줄 명시적 신호가 없고 multi-crop으로 view 수가 많아 head가 augmentation 인공물을 외우기 쉬우므로, 이 정도의 강한 규제를 감당할 수 있고 또 필요로 한다.

참고로 부록 E의 BYOL 비교 실험에서 weight decay 스윕은 $\{0.02, 0.05, 0.1\}$로 훨씬 작은 범위를 썼다 — DINO의 0.4는 그 자체로 이례적으로 큰 설정임을 보여준다.

---

## 4. 요약: 세 스케줄의 공통 논리

| 구간 | lr | weight decay | $\tau_t$ | 의도 |
|---|---|---|---|---|
| 0~10 ep | linear warmup ↑ | 0.04 (약) | 0.04 (뾰족, 안전) | 붕괴 없이 표현을 세운다 |
| ~30 ep | cosine decay ↓ | 증가 중 | 0.04→0.07 선형 ↑ | 타깃이 신뢰할 만해지면 부드럽게 |
| 이후 | 계속 ↓ | → 0.4 (강) | 0.07 고정 | 규제를 걸어 일반화 |

- $\tau_s = 0.1$만 처음부터 끝까지 고정 → 항상 $\tau_t < \tau_s$ (teacher가 더 뾰족) 유지.
- 세 스케줄 모두 "초기에는 자유롭게 / 후기에는 조이거나 부드럽게"라는 같은 방향의 설계이며, 근거 수치는 §3.2(스케줄 값)와 부록 D Sharpening 표(0.08 → 0.1% 붕괴, 0.04 → 69.6, warmup → 69.7)에 있다.

---

## 출처

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294v2 — §3.1 Avoiding collapse, §3.2 Implementation details, §5.3 Avoiding collapse (Fig. 7), Appendix D Additional Ablations (Online centering / Sharpening).
