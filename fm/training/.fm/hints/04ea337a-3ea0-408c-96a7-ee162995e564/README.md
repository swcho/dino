# teacher momentum $m$ 의 스케줄과 그 이유

> **Q.** teacher momentum $m$의 스케줄과 그 이유는?
> **A.** $0.996 \to 1.0$ 으로 cosine 증가한다. $m \to 1$이면 교사가 사실상 얼어붙어 타겟이 고정되므로 후반 학습이 안정된다.

---

## 1. 코드 위치

DINO의 교사는 역전파를 받지 않는다. 오직 학생 파라미터의 **지수이동평균(EMA)** 으로만 갱신된다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,
\qquad m: 0.996 \nearrow 1.0
$$

`main_dino.py`

```python
# L250 — 학습 시작 전에 iteration 길이 배열을 통째로 만든다
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                           args.epochs, len(data_loader))
...
# L348 — train_one_epoch 안, optimizer.step() 다음(12번째 단계)
with torch.no_grad():
    m = momentum_schedule[it]                      # momentum parameter
    for param_q, param_k in zip(student.module.parameters(),
                                teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

인자 정의:

```
--momentum_teacher  default=0.996
  "Base EMA parameter for teacher update. The value is increased to 1 during
   training with cosine schedule. We recommend setting a higher value with small
   batches: for example use 0.9995 with batch size of 256."
```

핵심 관찰 두 가지.

- **스케줄러는 상태가 없다.** `cosine_scheduler` 는 학습 *전에* 길이 `epochs × niter_per_ep` 짜리 numpy 배열을 만들고 루프는 `schedule[it]` 로 조회만 한다. 그래서 체크포인트에서 resume해도 스케줄이 자동으로 정확히 이어진다 (lr / wd / momentum 4종 모두 동일한 구조 — 노트북 §8).
- **momentum 스케줄에는 warmup이 없다.** `cosine_scheduler(0.996, 1, ...)` 호출에 `warmup_epochs` 를 주지 않으므로 기본 0이고, 첫 iteration부터 정확히 $m=0.996$ 이다. (lr만 `warmup_epochs=10` 을 받는다.)

---

## 2. 스케줄의 형태 — 왜 cosine인가

`utils.cosine_scheduler` (warmup 없는 경우):

$$
m_t = m_{\text{final}} + \tfrac{1}{2}\big(m_{\text{base}} - m_{\text{final}}\big)
\Big(1 + \cos\tfrac{\pi t}{T}\Big),
\qquad m_{\text{base}}=0.996,\ m_{\text{final}}=1.0
$$

$m_{\text{base}} < m_{\text{final}}$ 이므로 코사인 곡선이 **뒤집혀** 단조 증가 곡선이 된다. 모양의 성질:

- $t=0$ 근방에서 도함수가 0 → **초반엔 거의 변하지 않는다.** $m$ 이 한동안 $0.996$ 부근에 머무른다.
- $t=T$ 근방에서도 도함수가 0 → **끝에서 1에 부드럽게 도달한다.** 마지막에 계단처럼 급변하지 않는다.
- 변화의 대부분은 학습 중반에 몰려 있다.

선형 스케줄이었다면 매 스텝 $1-m$ 이 일정 비율로 줄어들어 초반부터 교사가 굳기 시작한다. cosine은 "초반 = 따라오기, 후반 = 얼기"라는 두 국면을 자연스럽게 만들어 준다.

노트북 §8은 lr / wd / momentum / teacher temp 4종을 한 줄에 나란히 그려서 momentum만 유일하게 **증가하는 곡선**(weight decay와 함께)임을 보여준다.

| 스케줄 | 시작 → 끝 | 방향 | 왜 |
|---|---|---|---|
| learning rate | $0 \to \texttt{lr} \to 10^{-6}$ | warmup 후 감소 | 표준 |
| weight decay | $0.04 \to 0.4$ | **증가** | 초기엔 자유 탐색, 후반에 표현 압축 |
| **teacher momentum $m$** | $\mathbf{0.996 \to 1.0}$ | **증가** | **교사를 점점 얼려 타겟을 안정화** |
| teacher temp $\tau_t$ | $0.04 \to 0.07$ | linear (warmup만) | 초기 고온은 불안정 |

---

## 3. $m$ 이 실제로 뜻하는 것: 유효 평균 구간 $1/(1-m)$

$\theta_t \leftarrow m\theta_t + (1-m)\theta_s$ 를 펼치면

$$
\theta_t^{(T)} = (1-m)\sum_{k\ge 0} m^{k}\,\theta_s^{(T-k)}
$$

즉 교사는 **학생 궤적의 지수가중 평균**이다. 가중치가 $1/e$ 로 떨어지는 시점이

$$
\tau_{\text{eff}} = \frac{1}{1-m}\ \text{ iteration}
$$

이고, 이것이 "교사는 최근 몇 step의 학생을 평균한 모델인가"에 대한 답이다. 동시에 $1-m$ 은 **한 step에 교사가 학생 쪽으로 이동하는 비율**이다.

| $m$ | 한 step 이동량 $1-m$ | $\tau_{\text{eff}}=1/(1-m)$ | epoch 환산 (niter = 1251) |
|---|---|---|---|
| $0.9$ | 10 % | 10 iter | 0.008 |
| $0.99$ | 1 % | 100 iter | 0.08 |
| $\mathbf{0.996}$ (시작) | **0.4 %** | **250 iter** | **0.20** |
| $0.999$ | 0.1 % | 1 000 iter | 0.80 |
| $0.9999$ | 0.01 % | 10 000 iter | 8.0 |
| $0.99999$ | 0.001 % | 100 000 iter | 80 |
| $1.0$ (끝) | $0$ | $\infty$ | $\infty$ — 완전 동결 |

**$0.996 \to 1$ 스케줄은 곧 "평균 구간을 250 step에서 $\infty$ 로 늘리는" 스케줄이다.**

100 epoch × 1251 iter 설정에서 실제 배열 값을 찍어보면:

| epoch | $m_t$ | $1-m_t$ | $\tau_{\text{eff}}$ (iter) | $\tau_{\text{eff}}$ (epoch) |
|---|---|---|---|---|
| 0 | 0.996000 | $4.0\times10^{-3}$ | 250 | 0.20 |
| 25 | 0.996586 | $3.4\times10^{-3}$ | 293 | 0.23 |
| 50 | 0.998000 | $2.0\times10^{-3}$ | 500 | 0.40 |
| 75 | 0.999414 | $5.9\times10^{-4}$ | 1 707 | 1.4 |
| 90 | 0.999902 | $9.8\times10^{-5}$ | 10 216 | 8.2 |
| 99 | 0.999999 | $9.9\times10^{-7}$ | 1 013 295 | 810 |
| 마지막 iter | $1-6.3\times10^{-13}$ | $6.3\times10^{-13}$ | $1.6\times10^{12}$ | $10^{9}$ |

초반 25 epoch 동안은 $\tau_{\text{eff}}$ 가 250 → 293 iter로 거의 그대로다(cosine의 평평한 구간). 반면 마지막 10 epoch에서 폭발적으로 늘어난다. 학습 끝에서는 $1-m \approx 6\times10^{-13}$ 로 **float32 machine epsilon($\approx 1.2\times10^{-7}$)보다 작아져 in-place 갱신이 수치적으로 no-op** 이 된다 — 교사는 비유가 아니라 문자 그대로 얼어붙는다.

---

## 4. 왜 초기엔 $m$ 이 (상대적으로) 작아야 하나

초기 학습에서 학생은 매우 빠르게 좋아진다. random init에서 시작한 교사가 만드는 타겟은 처음엔 거의 쓸모없는 신호다.

- **교사가 학생을 따라와야 한다.** $m$ 이 너무 크면(예: 처음부터 $0.9999$) 교사는 8 epoch 전의 학생을 평균한 모델이 된다. 학생이 이미 훨씬 좋아진 상태에서 **낡은(stale) 타겟**을 계속 맞추게 되어 학습이 지연되거나 정체된다.
- $m=0.996$ 은 $\tau_{\text{eff}} = 250$ iter $\approx 0.2$ epoch — "약 5분의 1 epoch 전의 학생"이라는 아주 짧은 지연이다. 교사는 학생의 개선을 거의 실시간으로 흡수하되, 개별 SGD step의 노이즈만 걸러낸다.
- 동시에 **너무 작으면 안 된다.** $m$ 이 작으면 교사 $\approx$ 학생이 되어, 학생이 자기 자신을 타겟으로 삼는 **자기 참조(self-referential)** 구조가 된다. 이 경우 손실은 자명하게 0으로 갈 수 있고 표현은 붕괴한다. 노트북 §14 표에도 `momentum_teacher` 항목에 "작으면 타겟 요동 → 붕괴"라고 적혀 있다.

즉 $0.996$ 은 "학생을 따라올 만큼 빠르지만, 학생과 구별될 만큼은 느린" 지점이다.

---

## 5. 왜 후반엔 $m \to 1$ 인가

후반부에는 학생의 개선 속도가 느려지고 lr도 $10^{-6}$ 까지 떨어진다. 이때 필요한 것은 "추종"이 아니라 "고정"이다.

1. **타겟 고정 → 수렴.** DINO의 손실은 움직이는 목표를 쫓는 문제다. 목표 자체가 계속 움직이면 학생은 결코 수렴하지 않는다. $m\to1$ 이면 타겟 분포 $P_t$ 가 고정되고, 학생은 잘 정의된 고정된 목표로 수렴할 수 있다.
2. **노이즈 억제.** $\tau_{\text{eff}}$ 가 커질수록 교사는 더 많은 학생 스냅샷을 평균한다. 마지막 epoch의 교사는 사실상 **수천~수만 step에 걸친 학생 가중치의 앙상블(Polyak averaging)** 이다. 가중치 평균은 SGD 노이즈를 지우고 더 평평한 해로 이끄는 것으로 알려져 있고, DINO에서 실제로 교사가 학생보다 계속 성능이 좋은 이유이기도 하다.
3. **양성 피드백 루프 차단.** 교사가 학생을 빠르게 따라가면 "학생의 편향 → 교사의 편향 → 더 강한 학생의 편향"이라는 되먹임이 생긴다. $m\to1$ 은 후반에 이 루프의 이득을 0으로 만들어, centering / sharpening이 잡아주지 못하는 느린 붕괴(slow collapse)까지 막는다.
4. **평가와의 정합.** DINO는 최종적으로 **교사** 백본을 배포한다(`teacher` 체크포인트가 k-NN/linear에서 더 좋다). 마지막에 얼어붙은 교사는 마지막 몇십 epoch의 앙상블이므로 그대로 쓰기 좋은 산출물이다.

> 주의: DINO 사전학습에는 **검증이 전혀 없다**(노트북 §11). 조기 종료도 best 체크포인트 선택도 못 한다. 마지막 상태가 곧 결과물이므로, 마지막에 타겟을 얼려서 안정적으로 착지시키는 것이 그만큼 중요하다.

---

## 6. 출처: BYOL에서 그대로 가져온 설계

이 스케줄은 DINO의 발명이 아니라 **BYOL(Grill et al., 2020)** 의 target network 갱신 규칙을 그대로 채택한 것이다. BYOL:

$$
\xi \leftarrow \tau\xi + (1-\tau)\theta,
\qquad
\tau = 1 - (1-\tau_{\text{base}})\cdot\frac{\cos(\pi k/K)+1}{2},
\quad \tau_{\text{base}} = 0.996
$$

$k$ = 현재 step, $K$ = 전체 step. 이 식은 `utils.cosine_scheduler(0.996, 1, ...)` 와 **정확히 동일**하다. DINO 논문 §3.1 "Teacher network"도 그대로 적는다: *"$\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, with $\lambda$ following a cosine schedule from 0.996 to 1 during training."*

### DINO 논문의 근거 (Figure 6, right)

DINO는 momentum 값 자체를 sweep하기보다, **교사를 어떻게 만들 것인가**를 비교한다 (ImageNet k-NN top-1, ViT-S/16):

| 교사 구성 방식 | k-NN top-1 |
|---|---|
| 학생 복사 (student copy, $m=0$) | **0.1** — 완전 붕괴 |
| 직전 iteration 학생 | **0.1** — 완전 붕괴 |
| 직전 epoch 학생 | 66.6 |
| **momentum encoder (EMA)** | **72.8** |

이 표가 §4의 두 방향을 동시에 말해준다. $m=0$(= 학생 복사)이나 $m$ 이 사실상 0인 "직전 iteration"은 자기 참조가 되어 **붕괴(0.1 %)** 한다. 반대로 "직전 epoch" 처럼 지연을 크게 주면 붕괴는 면하지만 타겟이 낡아 66.6 %에 그친다. 부드러운 EMA가 두 실패를 모두 피해 72.8 %를 낸다.

> **흔한 혼동 주의.** DINO 논문 Appendix D에 나오는 $m \in \{0,\ 0.9,\ 0.99,\ 0.999\}$ → k-NN $\{69.1,\ 69.7,\ 69.4,\ 0.1\}$ 표는 **teacher momentum이 아니라 centering의 smoothing 파라미터 $m_c$**(`DINOLoss(center_momentum=0.9)`) 에 대한 ablation이다. 여기서는 반대로 $m_c$ 가 **너무 크면**(0.999) center 갱신이 너무 느려 배치 편향을 따라가지 못하고 붕괴한다. 두 momentum은 이름만 같고 역할이 정반대이니 섞지 말 것.
>
> | | teacher momentum $m$ | center momentum $m_c$ |
> |---|---|---|
> | 대상 | 교사 **가중치** $\theta_t$ | 로짓 **중심** $c$ |
> | 기본값 | 0.996 → 1 (cosine 증가) | 0.9 (고정) |
> | 너무 작으면 | 교사≈학생 → 자기참조 붕괴 | center가 노이즈를 따라감 |
> | 너무 크면 | 타겟이 낡음 (단, $\to1$ 은 후반엔 의도된 것) | 편향 추적 실패 → 붕괴 |

---

## 7. "작은 배치에서는 0.9995로 올려라"의 의미

help 텍스트의 권장은 **step 수와 유효 구간의 관계** 때문이다.

$\tau_{\text{eff}} = 1/(1-m)$ 의 단위는 **iteration** 이지 epoch나 이미지 수가 아니다. 배치를 1024 → 256으로 줄이면 같은 epoch가 **4배 많은 iteration** 이 된다. $m$ 을 그대로 두면:

| 배치 | 1 epoch iter 수 | $m=0.996$ 일 때 $\tau_{\text{eff}}$ | 그 구간이 담는 이미지 수 |
|---|---|---|---|
| 1024 (DINO 기본) | 1 251 | 250 iter = 0.20 epoch | 약 256 k 장 |
| 256 | 5 005 | 250 iter = **0.05 epoch** | 약 64 k 장 |

즉 배치를 줄이면 교사가 평균하는 **데이터 양이 4배로 줄어** 타겟이 그만큼 요동친다. 이를 보정하려면 "$\tau_{\text{eff}} \times B$ 를 일정하게", 즉

$$
1-m \ \propto\ B
$$

로 두어야 한다. $B: 1024 \to 256$ 이면 $1-m: 0.004 \to 0.001$, 곧 $m = 0.999$. 실제 권장값 $0.9995$ 는 그보다 한 단계 더 보수적인데, 작은 배치는 그래디언트 자체도 더 시끄럽기 때문에 평활을 더 주는 쪽이 안전하다는 경험적 조정이다. **방향(배치가 작으면 $m$ 을 올린다)이 요점이다.**

같은 맥락에서 lr에는 별도로 linear scaling rule이 적용된다:
$\texttt{lr}_{\text{eff}} = 0.0005 \times \dfrac{\texttt{batch\_size\_per\_gpu} \times \texttt{world\_size}}{256}$.

---

## 8. 노트북 실측

### §9 — EMA가 정말 $1/(1-m)$ 시정수를 갖는지

학생 파라미터를 1.0으로 고정하고 교사를 0.0에서 출발시켜 $m=0.996$ 으로 EMA만 반복하면:

| EMA step | 교사 값 |
|---|---|
| 1 | 0.0040 |
| 10 | 0.0393 |
| 100 | 0.3302 |
| **250 $=1/(1-m)$** | **0.6329** $\approx 1-1/e = 0.6321$ |
| 500 | 0.8652 |
| 1000 | 0.9818 |
| 1500 | 0.9976 |

정확히 지수 수렴이고, $1/(1-m)$ 이 $1/e$-folding time이라는 해석이 수치로 확인된다.

### §10 — 실제 한 iteration에서의 이동량

12단계 해부의 마지막 단계(EMA 갱신) 전후로 $\max|\theta_s - \theta_t|$ 를 찍으면, $m=0.996$ 에서 교사는 그 간극의 **0.4 %만** 좁힌다.

```python
with torch.no_grad():                                      # 12) EMA teacher 갱신
    m = mo_s[gi]
    d0 = max((ps - pt).abs().max().item() for ps, pt in zip(...))
    for pq, pk in zip(student.parameters(), teacher.parameters()):
        pk.data.mul_(m).add_((1 - m) * pq.detach().data)
    d1 = max((ps - pt).abs().max().item() for ps, pt in zip(...))
# EMA 전 max|θs-θt| → 후 (m=0.996 이라 교사는 아주 조금만 따라감)
```

한 step에 0.4 %는 "거의 안 움직인다"처럼 보이지만, 250 step이면 63 %, 1000 step이면 98 %를 따라잡는다. 시간 척도를 iteration이 아니라 $1/(1-m)$ 단위로 봐야 한다.

---

## 9. $m$ 을 잘못 주면

| 설정 | 무슨 일이 생기나 |
|---|---|
| $m = 0$ (= 학생 복사) | 교사 = 학생. 손실이 자기 자신에 대한 cross-entropy가 되어 자명해로 붕괴. 논문에서 k-NN 0.1 % |
| $m$ 너무 작음 (예: 0.9) | 교사가 학생 노이즈를 거의 그대로 반영 → 타겟 요동 → 자기참조 되먹임 → 붕괴 위험 |
| $m$ 처음부터 너무 큼 (예: 0.9999 고정) | 타겟이 낡아 초기 학습 지연·정체. "직전 epoch 교사"(66.6 %)와 같은 실패 모드 |
| $m$ 을 1까지 올리지 않음 (예: 0.996 고정) | 후반에도 타겟이 계속 흔들려 수렴이 덜 깔끔. 앙상블 효과도 못 얻음 |
| **$0.996 \to 1$ cosine** | 초반 추종 + 후반 동결. DINO/BYOL 기본 |

> 노트북 §14의 하이퍼파라미터 표: `momentum_teacher` = `0.996 → 1`, 역할 = **타겟 안정성**, 잘못 주면 = **작으면 타겟 요동 → 붕괴**.

---

## 10. 한 장 요약

```
        ┌──────────── EMA (m: 0.996 ↗ 1.0, cosine) ────────────┐
        │                                                       │
   teacher(g1,g2) ──▶ centering(-c) + sharpening(τt) + detach ──▶ 타겟
        ▲                                                       │
        │                                                       ▼
   student(전부) ──────────────────── DINOLoss ─────── AdamW ───┘

  초반  m=0.996  →  τ_eff = 250 iter (0.2 epoch)  →  교사가 학생을 "따라온다"
  후반  m→1.0    →  τ_eff → ∞                     →  교사가 "얼어붙는다"
```

- **스케줄**: `utils.cosine_scheduler(0.996, 1, epochs, niter)` — warmup 없음, 단조 증가, 양 끝이 평평한 cosine.
- **왜 증가하나**: $1-m$ 은 한 step 추종률, $1/(1-m)$ 은 유효 평균 구간. 초반엔 짧게(빠르게 좋아지는 학생을 따라와야 하고, 낡은 타겟은 학습을 막는다), 후반엔 무한대로(타겟이 고정되어야 학생이 수렴하고, 긴 앙상블이 노이즈를 지운다).
- **출처**: BYOL의 target network 스케줄을 그대로 채택. DINO Figure 6(right)에서 momentum 교사 72.8 % vs 학생 복사/직전 iteration 0.1 %(붕괴), 직전 epoch 66.6 %.
- **배치가 작으면**: iteration 수가 늘어 같은 $m$ 이 더 짧은 데이터 구간을 뜻하므로 $1-m \propto B$ 로 낮춘다 → `--momentum_teacher 0.9995` (batch 256).
