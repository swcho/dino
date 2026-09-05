# 단일 프로토타입 collapse — 증상과 막는 장치

**Q.** DINO에서 단일 프로토타입 collapse의 증상과 막는 장치는?

**A.** $P_t$가 항상 같은 one-hot이 되고 $H(P_t) \to 0$ 이 된다. **centering**($z_t - c$)이 이를 막는다.

---

## 1. 왜 collapse가 "이득"인가 — 손실 분해

DINO의 손실은 교사 분포 $P_t$와 학생 분포 $P_s$의 교차엔트로피다. 이건 이렇게 쪼개진다.

$$
H\big(P_t, P_s\big) \;=\; \underbrace{H(P_t)}_{\text{교사 분포의 엔트로피}} \;+\; \underbrace{D_{\mathrm{KL}}\big(P_t \,\|\, P_s\big)}_{\text{두 view 정렬}}
$$

우리가 원하는 학습은 **두 번째 항**을 줄이는 것이다 — 같은 이미지의 서로 다른 crop이 같은 분포로 매핑되도록. 그런데 최적화기는 목적함수를 낮추면 그만이지 "어느 항을 낮추라"는 지시를 받지 않는다.

**첫 번째 항 $H(P_t)$를 0으로 만들어버리면** 아무것도 배우지 않고도 손실이 뚝 떨어진다. 이게 지름길이고, 이 지름길이 바로 단일 프로토타입 collapse다.

---

## 2. 단일 프로토타입 collapse의 정의

교사 head의 출력 차원은 $K$개의 **프로토타입**(DINO 기본 `out_dim` $K = 65536$, 워크스루에서는 $K = 4096$)이다. 정상이라면 입력 이미지마다 다른 프로토타입 쪽으로 확률질량이 쏠려야 한다.

단일 프로토타입 collapse는 **입력과 무관하게 항상 같은 프로토타입 $k^*$가 argmax를 차지하는** 상태다.

$$
\forall x:\quad P_t(\cdot\mid x) \;\to\; e_{k^*}, \qquad H(P_t) \;\to\; 0
$$

여기서 $e_{k^*}$는 $k^*$번째 성분만 1인 one-hot 벡터다. "one-dimensional collapse"라고도 부르는데, 표현 전체가 사실상 1차원(항상 같은 답)으로 무너졌다는 뜻이다.

### 증상 체크리스트

| 진단량 | 정상 | 단일 프로토타입 collapse |
|---|---|---|
| 교사 엔트로피 $H(P_t)$ | $0 \ll H(P_t) < \log K$ | $\to 0$ (급락) |
| 교사 top-1 확률 $\max_k P_t(k)$ | $1/K$보다 크지만 1에서 멂 | $\to 1$ (급등) |
| 배치 내 argmax 다양성 | 배치 크기에 가깝다 | $\to 1$ (전부 같은 $k^*$) |
| loss | $\log K$ 근처에서 천천히 | **가장 빠르게 내려간다** |
| center 노름 $\lVert c\rVert_2$ | 안정 | (centering이 없으면 관측 불가) |

**핵심 함정**: loss가 제일 잘 내려가는 설정이 바로 붕괴한 설정이다. 워크스루 §11이 지적하듯 `main_dino.py`의 사전학습 루프에는 검증이 전혀 없고 loss/lr/wd만 로깅한다. **loss만 보고 있으면 붕괴를 "학습 잘 됨"으로 오독한다.**

---

## 3. 왜 이 방향으로 밀리는가 — 자기강화 루프

두 가지가 겹친다.

**(a) sharpening이 뾰족함을 강화한다.** 교사 온도 $\tau_t = 0.04$는 학생 온도 $\tau_s = 0.1$보다 낮다.

$$
P_t^{(u)}(k) = \frac{\exp\!\big((z_t^{(u)}(k) - c_k)/\tau_t\big)}{\sum_j \exp\!\big((z_t^{(u)}(j) - c_j)/\tau_t\big)}
$$

$\tau_t$가 작을수록 logit의 작은 차이가 크게 증폭된다. 어떤 프로토타입이 조금이라도 평균적으로 높은 logit을 받으면, 낮은 온도가 그 우위를 거의 확정적인 one-hot으로 부풀린다.

**(b) EMA 교사가 학생의 편향을 되먹인다.** 교사는 학생의 exponential moving average다($m: 0.996 \to 1.0$). 학생이 프로토타입 $k^*$를 편애하기 시작하면 → 교사도 $k^*$를 편애하게 되고 → 교사가 만든 타겟이 다시 학생에게 $k^*$를 학습시킨다. **rich-get-richer** 자기강화 루프다. 한 번 기울면 되돌아올 힘이 손실함수 안에 없다.

---

## 4. centering이 막는 원리

교사 logit에서 **배치 평균의 EMA** $c$를 뺀다.

$$
c \;\leftarrow\; m_c\, c \;+\; (1 - m_c)\,\frac{1}{B\cdot W}\sum_{i=1}^{B\cdot W} z_t(i),
\qquad m_c = 0.9
$$

($W$는 world_size. 그래서 `main_dino.py`의 `update_center` 안에 `dist.all_reduce`가 있고, 프로세스 그룹 없이는 `DINOLoss`가 돌지 않는다.)

직관: $c_k$는 "프로토타입 $k$가 평균적으로 받는 기본 점수"다. 이걸 빼면 **모든 프로토타입의 평균 점수가 0으로 정규화**되어, 어떤 프로토타입도 구조적 우위를 갖지 못한다. 남는 것은 "이 입력이 평균보다 얼마나 더/덜 $k$스러운가"라는 **상대적 신호**뿐이다.

여기서 미묘하지만 결정적인 점: softmax는 **모든 성분에 같은 상수**를 더하거나 빼면 불변이지만, **성분마다 다른 상수**($c_k$가 $k$마다 다르다)를 빼면 분포가 실제로 바뀐다. centering이 무의미한 연산이 아닌 이유가 이것이다.

$$
\mathrm{softmax}(z + a\mathbf{1}) = \mathrm{softmax}(z) \quad\text{이지만}\quad \mathrm{softmax}(z - c) \neq \mathrm{softmax}(z)\ \ (c \not\propto \mathbf{1})
$$

### 코드에서 (`main_dino.py` `DINOLoss`)

```python
self.register_buffer("center", torch.zeros(1, out_dim))
...
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)
...
self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

`center`는 파라미터가 아니라 **buffer**이고 `update_center`는 `@torch.no_grad()`다 — gradient가 흐르지 않는, 통계량 기반의 보정이다. 교사 쪽에는 `.detach()`가 걸려 있어 gradient는 학생으로만 흐른다.

---

## 5. 실측 1 — §7 실험 B (합성 logit)

프로토타입 0에만 인위적으로 `bias = 2.0`을 주입해 "구조적으로 유리한 프로토타입"을 만들고, centering만 켜고/끄며 300 step을 돌린 결과다($K = 512$, $B = 64$, $\tau_t = 0.04$, $m_c = 0.9$).

| 설정 | 프로토타입 0이 argmax인 비율 (마지막 50 step 평균) |
|---|---|
| centering 없음 | **0.819** — 배치의 82%를 한 프로토타입이 독식 |
| centering 있음 | **0.003** — uniform 기대값 $1/K = 0.00195$에 근접 |

그리고 학습된 center가 그 bias를 그대로 흡수한다: $c_0 \approx 2.0$(주입한 bias와 일치), 나머지 성분 평균 $\approx 0$. **center가 편향을 그대로 빨아들여 빼준다**는 게 눈으로 확인된다.

### 중요한 단서 — centering은 엔트로피를 올리지 않는다

같은 실험의 패널 C를 보면, **$\tau_t = 0.04$인 한 $H(P_t)$는 centering 유무와 무관하게 둘 다 낮다.**

- **centering**: "어떤 프로토타입이 뽑히나"의 **균형**을 담당한다 (배치 전체에 걸친 분포).
- **sharpening**: "얼마나 확신하나"를 담당한다 (개별 샘플의 뾰족함).

즉 centering은 $P_t$를 uniform 쪽으로 "부드럽게" 만드는 장치가 아니다. **개별 샘플은 여전히 뾰족하되, 그 뾰족함이 향하는 대상이 샘플마다 흩어지도록** 만드는 장치다. 이래서 두 장치는 서로를 대체할 수 없다.

---

## 6. 실측 2 — §11 ablation (실제 ViT 미니 학습)

같은 학습 루프를 세 설정으로 3 epoch 돌린 결과($K = 4096$, $\log K \approx 8.317$, batch 8 → 교사 출력 16행). `centering 제거`는 `dl_.update_center = lambda *a, **k: None`으로 center를 0에 고정해 구현했다.

| 설정 | loss 처음→끝 | $H(P_t)$ 처음→끝 | top-1 처음→끝 | argmax 다양성 |
|---|---|---|---|---|
| DINO (center + sharpen) | $\log K$ 근처 유지 | $\log K$보다 확실히 낮은 값에서 안정 | $1/K$보다 크되 1에서 멂 | 유지 |
| **centering 제거** | **8.076 → 6.628** | **7.15 → 5.86** | **0.020 → 0.192** | **9 → 5** |
| sharpening 제거 ($\tau_t = \tau_s$) | $\log K \approx 8.32$에서 꼼짝 안 함 | $\to \log K$ | $\to 1/K$ | — |

읽는 법:

- **centering 제거가 세 설정 중 loss를 가장 많이 내렸다** (8.076 → 6.628). 그런데 그 감소분은 $D_{\mathrm{KL}}(P_t\|P_s)$를 줄여서 얻은 게 아니라 $H(P_t)$를 7.15 → 5.86으로 **깎아서** 얻은 것이다. 손실 분해의 첫 항이 깎인 만큼 그대로 loss가 내려갔다.
- top-1이 0.020 → 0.192로 **약 10배** 뛰었고, 16행 배치의 argmax 다양성이 9 → 5로 줄었다. 세 지표가 **동시에** 붕괴 방향을 가리킨다.
- 다만 이건 수백 step짜리 미니 실험이라 $H(P_t)$가 0까지 내려가진 않았다. 방향과 속도가 명백할 뿐, 실제 학습을 오래 돌리면 $\to 0$, top-1 $\to 1$, 다양성 $\to 1$로 간다.

---

## 7. 진단 방법 — 한 문장

> **loss는 내려가는데 $H(P_t)$가 급락하고 top-1이 급등하고 argmax 다양성이 줄어드는 패턴이 동시에 나타나면 단일 프로토타입 collapse다.**

세 지표가 함께 움직여야 한다. loss만 보면 놓치고, 엔트로피만 보면 uniform collapse와 헷갈린다. 사전학습에는 검증이 없으므로(§14 함정 4) 이 진단량들을 직접 로깅하는 수밖에 없다. 최소한 매 iteration:

```python
p_t = F.softmax((teacher_output.float() - dino_loss.center) / teacher_temp, dim=-1)
H_t  = (-(p_t * p_t.clamp_min(1e-12).log()).sum(-1)).mean()   # → 0 이면 위험
top1 = p_t.max(-1).values.mean()                              # → 1 이면 위험
uniq = p_t.argmax(-1).unique().numel()                        # → 1 이면 위험
cnorm = dino_loss.center.norm()                               # 발산하면 위험
```

---

## 8. uniform collapse와의 대비

| | **단일 프로토타입 collapse** | **uniform collapse** |
|---|---|---|
| 최종 분포 | $P_t \to e_{k^*}$ (항상 같은 one-hot) | $P_t \to 1/K$ (모든 입력이 같은 flat 분포) |
| $H(P_t)$ | $\to 0$ | $\to \log K$ |
| top-1 확률 | $\to 1$ | $\to 1/K$ |
| argmax 다양성 | $\to 1$ | 무의미 (거의 균등, 노이즈로 결정) |
| loss 거동 | **급격히 내려간다** (지름길) | $\log K$에서 **꼼짝 안 한다** (평탄면) |
| 무엇이 사라지나 | 입력별 구분이 사라짐 (다 같은 답) | 확신이 사라짐 (gradient 소멸) |
| **막는 장치** | **centering** ($z_t - c$) | **sharpening** ($\tau_t < \tau_s$) |
| 미는 방향 | centering이 uniform 쪽으로 민다 | sharpening이 one-hot 쪽으로 민다 |

두 장치는 **서로 반대 방향으로 민다.** 하나만 있으면 그 반대쪽 붕괴로 간다. DINO의 건강한 상태는 두 붕괴 영역 사이에 **매달려 있는(hanging)** 균형이다 — 논문 Sec. 5.3 / Fig. 5의 요지이자, §11에서 loss가 $\log K$ 근처에 머무는데도 $H(P_t)$는 $\log K$보다 확실히 낮은 이유다.

기억하기: **centering = "누가 뽑히나"의 균형, sharpening = "얼마나 확신하나"의 세기.**

---

## 9. `center_momentum` = 0.9 한 줄

$m_c = 0.9$는 center의 EMA 반감기를 정한다 — 대략 최근 10 step 규모의 배치 평균을 추적한다. **너무 크면**(1에 가까우면) 편향이 옮겨갈 때 center가 따라가지 못해 collapse를 놓치고, **너무 작으면** center가 배치 노이즈에 흔들려 타겟이 요동친다. 참고로 세 번째 보조 장치인 `freeze_last_layer`(첫 1 epoch 동안 마지막 층의 gradient를 버림)도 초기 노이즈로 프로토타입이 흔들리며 한쪽으로 기우는 것을 막는 역할을 한다.

---

## 출처

- `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` §6 (DINOLoss 수식), §7 (실험 B), §11 (ablation 실측), §14 (하이퍼파라미터·함정)
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — `DINOLoss.forward` / `DINOLoss.update_center`
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), Sec. 5.3 "Avoiding collapse" / Fig. 5
