# Algorithm 1 (DINO 의사코드)의 손실 계산

## 핵심 한 줄

두 증강 뷰 $x_1, x_2$ 에 대해 **교차·대칭** 크로스엔트로피를 계산한다.

$$\mathcal{L} = \tfrac{1}{2}H(t_1, s_2) + \tfrac{1}{2}H(t_2, s_1)$$

즉 **한 뷰의 teacher 출력**을 목표로 **다른 뷰의 student 출력**을 맞춘다. 여기서 $H(a,b) = -a\log b$ 이고, $t_i$ 에는 stop-gradient가 걸려 있어 그래디언트는 student 쪽으로만 흐른다.

---

## 논문의 Algorithm 1 원문

> **Algorithm 1** DINO PyTorch pseudocode w/o multi-crop.

```python
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader:                      # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x)   # random views

    s1, s2 = gs(x1), gs(x2)           # student output n-by-K
    t1, t2 = gt(x1), gt(x2)           # teacher output n-by-K

    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward()                   # back-propagate

    # student, teacher and center updates
    update(gs)                        # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach()                    # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```

(에셋 마크다운에서는 OCR 때문에 momentum 변수 `l`이 숫자 `1`처럼 보이는데, 원 논문의 `gt.params = l*gt.params + (1-l)*gs.params`가 맞다.)

---

## 한 줄씩 해설

| 줄 | 하는 일 |
|---|---|
| `gt.params = gs.params` | teacher를 student와 **동일 초기화**. DINO는 predictor가 없어 두 네트워크 구조 $g = h \circ f$ 가 완전히 같다. 사전학습된 teacher가 주어지는 일반 knowledge distillation과 달리, teacher는 student의 과거 상태로부터 만들어진다. |
| `x1, x2 = augment(x), augment(x)` | 같은 이미지 $x$ 에서 **서로 다른 random view 두 개**를 생성. BYOL 계열 증강(color jitter, Gaussian blur, solarization) + multi-crop. multi-crop 없는 버전이므로 view는 2개뿐. |
| `s1, s2 = gs(x1), gs(x2)` | student가 두 뷰 모두를 forward → $n \times K$ 로짓. |
| `t1, t2 = gt(x1), gt(x2)` | teacher도 두 뷰 모두를 forward. multi-crop을 쓰면 여기에 **global view만** 들어간다(local-to-global). |
| `loss = H(t1, s2)/2 + H(t2, s1)/2` | **핵심**: 교차 항 두 개의 평균. $t_1$ 은 $x_1$ 의 teacher 출력, $s_2$ 는 $x_2$ 의 student 출력 — 인덱스가 어긋나 있다는 점이 전부다. |
| `loss.backward()` | 역전파. `H` 안의 `t.detach()` 덕분에 그래디언트는 **student 경로로만** 흐른다(Figure 2의 `sg`). |
| `update(gs) # SGD` | student 파라미터 $\theta_s$ 만 경사하강으로 갱신(실제 구현은 AdamW). teacher는 옵티마이저에 등록되지 않는다. |
| `gt.params = l*gt.params + (1-l)*gs.params` | teacher는 학습되지 않고 student의 **EMA(momentum encoder)**: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$, $\lambda$ 는 0.996 → 1 코사인 스케줄. Polyak–Ruppert 평균처럼 작동해 teacher가 student보다 항상 좋은 타깃을 준다. |
| `C = m*C + (1-m)*cat([t1,t2]).mean(dim=0)` | center의 **EMA 갱신**: $c \leftarrow mc + (1-m)\frac{1}{B}\sum_i g_{\theta_t}(x_i)$ (식 (4)). 이번 스텝의 teacher 출력 두 묶음을 이어붙여 배치 평균을 낸다. 1차 배치 통계만 쓰므로 배치 크기 의존성이 낮다. |
| `H`: `t = t.detach()` | stop-gradient. teacher는 상수 타깃. |
| `H`: `s = softmax(s / tps)` | student는 온도 $\tau_s = 0.1$ 로 softmax. |
| `H`: `t = softmax((t - C) / tpt)` | teacher는 **centering(−C) + sharpening(낮은 $\tau_t$, 0.04→0.07)**. centering은 한 차원 독주를 막지만 균등분포 붕괴를 유도하고, sharpening은 그 반대 — 둘을 함께 써야 collapse가 방지된다. |
| `H`: `return -(t*log(s)).sum(1).mean()` | $K$ 차원 크로스엔트로피를 배치 평균. |

![DINO 자기증류 구조 (Figure 2)](fig-1.jpeg)

Figure 2가 위 코드와 정확히 대응한다. 왼쪽 아래에서 이미지 하나가 두 갈래 random transformation으로 갈라져 각각 student($\theta_s$)와 teacher($\theta_t$)로 들어가고, teacher 쪽에만 **centering → softmax(sharpen) → `sg`(stop-gradient)** 가 붙는다. 두 출력이 만나는 지점이 `loss` 박스이고, teacher 파라미터로 향하는 **`ema` 화살표**가 `gt.params = l*gt.params + (1-l)*gs.params` 줄이다. 그림은 "한 쌍의 뷰에 대한 한쪽 방향"만 그린 것이고, 실제 코드는 이 그림을 좌우로 뒤집은 대칭 항을 하나 더 더한다.

---

## 왜 **교차**인가 ($t_1 \leftrightarrow s_2$, $t_2 \leftrightarrow s_1$)

- **같은 뷰끼리 맞추면 학습 신호가 자명해진다.** teacher는 student의 EMA라 파라미터가 거의 같다. 같은 입력 $x_1$ 을 넣으면 $t_1 \approx s_1$ 이므로 $H(t_1, s_1)$ 은 이미 거의 최소값이고 그래디언트가 유의미한 정보를 주지 못한다. "자기 자신을 복사하라"는 문제는 아무 표현도 학습시키지 않는다.
- **서로 다른 뷰를 맞춰야 증강 불변(augmentation-invariant) 표현이 생긴다.** $x_1$ 과 $x_2$ 는 crop 위치·색·블러가 다르다. $s_2$ 가 $t_1$ 을 예측하도록 강제하면, 네트워크는 "crop과 색이 달라도 같은 이미지면 같은 분포"라는 제약을 만족해야 한다. 이것이 유일한 학습 신호다.
- multi-crop 설정에서 이 교차는 **local-to-global 대응**으로 확장된다. $96^2$ local crop을 본 student가 $224^2$ global crop을 본 teacher의 출력을 예측해야 하므로, 부분에서 전체 맥락을 추론하는 능력이 강제된다.
- 실제 구현도 동일 뷰 항을 명시적으로 건너뛴다 — `main_dino.py`의 `DINOLoss.forward`에 `if v == iq: continue  # we skip cases where student and teacher operate on the same view`.

## 왜 **대칭**으로 두 항을 평균하는가

- 한쪽 방향만 쓰면($H(t_1,s_2)$ 만) 뷰의 역할이 **비대칭**해진다. $x_1$ 은 영원히 타깃 제공자, $x_2$ 는 영원히 예측자가 되어 같은 forward 비용(뷰 2개 × 네트워크 2개 = 4회)을 쓰고도 **손실 항은 절반**만 얻는다. 이미 계산해 둔 $t_2, s_1$ 이 그냥 버려진다.
- 대칭화하면 두 뷰가 동등하게 타깃이자 예측 대상이 되어 그래디언트 분산이 줄고 학습이 안정된다(SimCLR·BYOL·SimSiam의 symmetrized loss와 같은 이유).
- `/2`는 항 개수로 나누는 **정규화**다. 항이 늘어도 손실 스케일(≈ 학습률의 실효 크기)이 변하지 않게 한다. 코드에서는 `total_loss /= n_loss_terms`로 일반화되어 있다.

## §3.1 식 (3)과의 관계

논문의 일반 손실은

$$\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big) \qquad (3)$$

- 바깥 합: **teacher는 global view만** 본다.
- 안쪽 합: student는 $V$ 의 **모든** 뷰를 보되 $x' \neq x$ — 이것이 코드의 "동일 뷰 건너뛰기"다.

Algorithm 1은 multi-crop이 없는 경우, 즉 $V = \{x_1^g, x_2^g\}$ 인 **식 (3)의 특수한 경우**다. 대입하면 $x = x_1$ 일 때 $x' = x_2$ 만 남고, $x = x_2$ 일 때 $x' = x_1$ 만 남아

$$H(P_t(x_1), P_s(x_2)) + H(P_t(x_2), P_s(x_1))$$

이 되고, 항 수 2로 나눈 것이 코드의 `H(t1,s2)/2 + H(t2,s1)/2`이다. 기본 설정처럼 global 2개 + local $L$ 개를 쓰면 항은 $2(1 + L)$ 개가 되고 나누는 수도 그만큼 커진다(예: local 8개 → 18개 항). 논문 표현대로 "이 손실은 일반적이며 뷰가 2개뿐일 때도 쓸 수 있다".

---

## 자주 헷갈리는 점

- **$H$ 는 대칭이 아니다.** $H(t,s) = -\sum t\log s$ 에서 첫 인자는 타깃(teacher), 둘째는 예측(student). `H(s2, t1)`로 쓰면 완전히 다른 손실이 된다. 손실 **전체**가 대칭인 것이지 $H$ 자체가 대칭인 게 아니다.
- **teacher는 손실로 학습되지 않는다.** `detach()` + EMA 갱신뿐이다. codistillation과 달리 teacher가 student로부터 증류받지 않고 student의 평균일 뿐이다.
- **center $C$ 는 손실 계산 시점의 값을 쓰고, 갱신은 그 후에 한다.** 코드 순서상 `H`가 이전 스텝의 $C$ 를 사용한 뒤 마지막 줄에서 EMA로 갱신된다.
