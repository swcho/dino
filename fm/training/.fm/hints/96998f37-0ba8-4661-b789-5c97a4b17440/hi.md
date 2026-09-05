# `cosine_scheduler`의 수식 — 삼각함수 그래프 변형으로 읽기

## 0. 목표 수식 먼저 보기

DINO의 `utils.cosine_scheduler`가 만드는 값 $v_t$ (여기서 $t$는 **iteration 번호**)는 두 구간으로 나뉜다.

$$
v_t =
\begin{cases}
\dfrac{t}{T_w}\, v_{\text{base}} & t < T_w \quad \text{(linear warmup)} \\[10pt]
v_{\text{final}} + \dfrac{1}{2}\big(v_{\text{base}} - v_{\text{final}}\big)
\left(1 + \cos\dfrac{\pi (t - T_w)}{T - T_w}\right) & t \ge T_w
\end{cases}
$$

- $T$ : 전체 iteration 수 $=$ `epochs` $\times$ `niter_per_ep`
- $T_w$ : warmup iteration 수 $=$ `warmup_epochs` $\times$ `niter_per_ep`
- $v_{\text{base}}$ : warmup이 끝났을 때 도달하는 값 (코사인 구간의 **시작값**)
- $v_{\text{final}}$ : 학습이 끝났을 때의 값 (코사인 구간의 **끝값**)

아래 절부터, 이 아랫줄 식이 고등학교에서 배운 $y=\cos x$ 그래프를 **네 번 변형**해서 만들어진 것임을 하나씩 쌓아 올린다.

---

## 1. 재료: $\cos\theta$를 $[0,\pi]$에서만 잘라 쓴다

$y = \cos\theta$는 주기 $2\pi$의 진동함수지만, **정의역을 $\theta \in [0,\pi]$로 제한**하면 성질이 확 달라진다.

| $\theta$ | $0$ | $\dfrac{\pi}{4}$ | $\dfrac{\pi}{2}$ | $\dfrac{3\pi}{4}$ | $\pi$ |
|---|---|---|---|---|---|
| $\cos\theta$ | $1$ | $\approx 0.707$ | $0$ | $\approx -0.707$ | $-1$ |

이 구간에서 $\cos\theta$는

- **단조 감소**한다 ($\cos'\theta = -\sin\theta \le 0$, $\theta\in[0,\pi]$에서 $\sin\theta \ge 0$이므로),
- $1$에서 출발해 $-1$로 끝나며,
- 양 끝 $\theta = 0, \pi$에서 접선의 기울기가 $0$이다 ($-\sin 0 = 0$, $-\sin\pi = 0$),
- 한가운데 $\theta = \pi/2$에서 기울기가 $-\sin(\pi/2) = -1$로 **가장 가파르다**.

즉 "완만하게 출발 → 중간에 급하게 → 완만하게 착지"하는 **S자 곡선**이다. 스케줄러가 원하는 모양이 정확히 이것이다.

---

## 2. 변형 ①: 치역을 $[-1,1]$에서 $[0,1]$로 — 세로 축소 + 평행이동

$\cos\theta$의 값은 $[-1,1]$인데, 우리가 원하는 건 "$100\%$에서 $0\%$로 줄어드는 비율"이다. 고교 그래프 변형 그대로:

$$
u(\theta) = \frac{1 + \cos\theta}{2}
$$

- $+1$ : 그래프를 위로 1만큼 평행이동 → 치역 $[0,2]$
- $\div 2$ : 세로로 $\tfrac12$배 축소(진폭 $1 \to \tfrac12$) → 치역 $[0,1]$

확인하면 $u(0) = \frac{1+1}{2} = 1$, $u(\pi) = \frac{1-1}{2} = 0$. **1에서 0으로 부드럽게 떨어지는 비율 함수**를 얻었다.

> 참고: $u(\theta) = \dfrac{1+\cos\theta}{2} = \cos^2\dfrac{\theta}{2}$ (반각 공식). 항상 $0$ 이상임이 이 형태에서 바로 보인다.

---

## 3. 변형 ②: 비율을 두 값 사이로 늘리기 — 내분점(선형보간)

이제 $u \in [0,1]$이라는 "비율"이 있으니, 이걸 $v_{\text{final}}$과 $v_{\text{base}}$ 사이의 값으로 옮긴다. 수직선 위에서 두 점 $A = v_{\text{final}}$, $B = v_{\text{base}}$를 잡고, $B$쪽 가중치가 $u$인 점은

$$
v = (1-u)\,v_{\text{final}} + u\,v_{\text{base}} = v_{\text{final}} + u\,(v_{\text{base}} - v_{\text{final}})
$$

이것이 **선형보간(linear interpolation)** 이자 고교 기하의 **내분점** 공식이다. $u=1$이면 $v = v_{\text{base}}$, $u=0$이면 $v = v_{\text{final}}$.

여기에 2절의 $u$를 대입하면

$$
v = v_{\text{final}} + \frac{1}{2}\big(v_{\text{base}} - v_{\text{final}}\big)\big(1 + \cos\theta\big)
$$

목표 수식의 아랫줄이 거의 완성됐다. 남은 건 $\theta$가 무엇인지다.

**중요**: 이 식은 $v_{\text{base}} > v_{\text{final}}$(감소)든 $v_{\text{base}} < v_{\text{final}}$(증가)든 그대로 성립한다. $v_{\text{base}} - v_{\text{final}}$이 음수면 곡선이 위아래로 뒤집힐 뿐, "부드럽게 출발해서 부드럽게 착지"하는 성질은 유지된다. DINO가 lr(감소), wd(증가), momentum(증가)를 **같은 함수 하나**로 처리할 수 있는 이유다.

---

## 4. 변형 ③: 시간축 매핑 — $[T_w, T] \to [0,\pi]$

우리에게 주어진 건 $\theta$가 아니라 iteration 번호 $t$다. $t$가 $T_w$일 때 $\theta = 0$, $t$가 $T$일 때 $\theta = \pi$가 되도록 **일차함수로 사상**한다.

두 점 $(T_w, 0)$, $(T, \pi)$를 지나는 직선의 기울기는 $\dfrac{\pi - 0}{T - T_w}$이므로

$$
\theta(t) = \frac{\pi}{T - T_w}\,(t - T_w) = \pi \cdot \frac{t - T_w}{T - T_w}
$$

여기서 $\dfrac{t - T_w}{T - T_w}$는 "코사인 구간의 진행률"($0 \to 1$)이고, 거기에 $\pi$를 곱해 각도로 바꾼 것이다. 이것을 3절 식에 넣으면 목표 수식이 완성된다.

$$
\boxed{\;v_t = v_{\text{final}} + \frac{1}{2}\big(v_{\text{base}} - v_{\text{final}}\big)
\left(1 + \cos\frac{\pi(t - T_w)}{T - T_w}\right)\;}
$$

---

## 5. 변형 ④(윗줄): warmup은 그냥 1차함수

$t < T_w$ 구간은 원점을 지나는 직선이다.

$$
v_t = \frac{v_{\text{base}}}{T_w}\, t
$$

- 기울기 $\dfrac{v_{\text{base}}}{T_w}$, $y$절편 $0$인 **정비례 함수**.
- $t=0$에서 $0$, $t=T_w$에서 $v_{\text{base}}$ — 코사인 구간의 시작값과 정확히 이어진다.

DINO 구현에는 `start_warmup_value` 인자가 있는데, 이것이 $0$이 아니면 절편이 생겨 일반 1차함수가 된다.

$$
v_t = v_{\text{start}} + \frac{v_{\text{base}} - v_{\text{start}}}{T_w}\,t
$$

(구현은 `np.linspace(start_warmup_value, base_value, warmup_iters)` — 즉 $T_w$개의 점을 양 끝 포함해 균등 분할하므로, 실제 분모는 $T_w - 1$이다. 개념식과 미세하게 다르지만 $T_w$가 수천 이상이라 무시할 수준이다.)

**왜 warmup이 필요한가**: 학습 초기에 파라미터는 랜덤 초기값이고, Adam류 옵티마이저의 2차 모멘트 추정도 아직 부정확하다. 이때 큰 lr을 주면 발산하기 쉬우므로 $0$에서 서서히 올린다.

---

## 6. 경계값 확인 — 식이 정말 맞는지

수식을 외우는 것보다 **양 끝을 대입해 보는 습관**이 훨씬 안전하다.

**$t = T_w$일 때** (warmup이 막 끝난 순간):
$$
\cos\frac{\pi \cdot 0}{T - T_w} = \cos 0 = 1
\;\Longrightarrow\;
v = v_{\text{final}} + \frac{1}{2}(v_{\text{base}} - v_{\text{final}})(1+1) = v_{\text{base}} \;\checkmark
$$

**$t = T$일 때** (학습 끝):
$$
\cos\frac{\pi(T - T_w)}{T - T_w} = \cos\pi = -1
\;\Longrightarrow\;
v = v_{\text{final}} + \frac{1}{2}(v_{\text{base}} - v_{\text{final}})(1-1) = v_{\text{final}} \;\checkmark
$$

**중간점 $t = \dfrac{T_w + T}{2}$**:
$$
\cos\frac{\pi}{2} = 0 \;\Longrightarrow\; v = \frac{v_{\text{base}} + v_{\text{final}}}{2}
$$
정확히 두 값의 산술평균. 코사인 스케줄은 **"절반 시점에 절반 값"** 이라는 깔끔한 성질을 가진다.

---

## 7. 왜 하필 코사인인가 — 미분해 보자

$t \ge T_w$ 구간을 $t$로 미분한다. 합성함수 미분법($\frac{d}{dt}\cos\theta(t) = -\sin\theta(t)\cdot\theta'(t)$)을 쓰고, $\theta'(t) = \dfrac{\pi}{T-T_w}$이므로

$$
\frac{dv}{dt}
= \frac{1}{2}(v_{\text{base}} - v_{\text{final}}) \cdot \left(-\sin\theta\right) \cdot \frac{\pi}{T - T_w}
= -\,\frac{\pi\,(v_{\text{base}} - v_{\text{final}})}{2\,(T - T_w)}\,\sin\frac{\pi(t - T_w)}{T - T_w}
$$

이 도함수에서 세 가지가 읽힌다.

1. **$t = T_w$에서 $\dfrac{dv}{dt} = 0$** ($\sin 0 = 0$) — warmup에서 코사인으로 넘어갈 때 값이 급격히 꺾이지 않는다.
2. **$t = T$에서 $\dfrac{dv}{dt} = 0$** ($\sin\pi = 0$) — **부드러운 착지(soft landing)**. 학습 마지막 순간에 lr이 흔들리지 않고 정착하므로, 최적점 근처에서 파라미터가 안정적으로 수렴한다. 이것이 step decay(계단식 감소)나 linear decay(끝에서 기울기가 $-$상수로 유지)보다 코사인이 선호되는 핵심 이유다.
3. **중간 $t = \dfrac{T_w+T}{2}$에서 $|\sin| = 1$로 최대** — 값이 가장 빠르게 변한다. 즉 "탐색은 오래 하고, 감소는 중반에 몰아서 하고, 마무리는 천천히"라는 배분이 저절로 나온다.

최대 변화율은
$$
\left|\frac{dv}{dt}\right|_{\max} = \frac{\pi\,|v_{\text{base}} - v_{\text{final}}|}{2\,(T - T_w)}
$$
로, 전체 변화폭을 구간 길이로 나눈 평균 변화율 $\dfrac{|v_{\text{base}}-v_{\text{final}}|}{T-T_w}$의 $\dfrac{\pi}{2} \approx 1.571$배다.

---

## 8. 하나의 식, 네 개의 스케줄 (DINO 실제 값)

`main_dino.py`는 이 **같은 함수**에 인자만 바꿔 넣어 세 개(+ teacher temp는 linear)의 스케줄을 만든다.

| 스케줄 | $v_{\text{base}}$ | $v_{\text{final}}$ | $T_w$ (epoch) | 부호 $v_{\text{base}}-v_{\text{final}}$ | 방향 | 이유 |
|---|---|---|---|---|---|---|
| learning rate | $0.0005 \times \frac{\text{bs}\times\text{ws}}{256}$ | $10^{-6}$ | $10$ | $+$ | 감소 | 표준. warmup 후 수렴 |
| weight decay | $0.04$ | $0.4$ | $0$ | $-$ | **증가** | 초반엔 자유롭게 탐색, 후반에 표현을 압축 |
| teacher momentum $m$ | $0.996$ | $1.0$ | $0$ | $-$ | 증가 | 교사를 점점 얼려 타겟을 안정화 |
| teacher temp $\tau_t$ | $0.04$ | $0.07$ | — | — | 증가 | (코사인 아님, 앞 30 epoch linear 후 상수) |

lr에 곱해진 $\dfrac{\text{batch\_size} \times \text{world\_size}}{256}$은 **linear scaling rule** — 배치가 $k$배 커지면 lr도 $k$배로.

**부호가 음수인 경우 다시 확인**: wd는 $v_{\text{base}}-v_{\text{final}} = 0.04-0.4 = -0.36$. $t=T_w=0$에서 $0.4 + \frac12(-0.36)(2) = 0.4 - 0.36 = 0.04$ $\checkmark$, $t=T$에서 $0.4 + 0 = 0.4$ $\checkmark$. 위로 볼록하게 시작해 아래로 볼록하게 끝나는, 2절 곡선을 상하 반전한 모양이다.

---

## 9. 구현상의 두 가지 잔가지

실제 코드(`/home/sungwoo/projects/swcho/dino/utils.py`)는 이렇다.

```python
warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
iters = np.arange(epochs * niter_per_ep - warmup_iters)
schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
schedule = np.concatenate((warmup_schedule, schedule))
assert len(schedule) == epochs * niter_per_ep
```

1. **분모가 $T-T_w$이지 $T-T_w-1$이 아니다.** `iters`는 $0, 1, \dots, N-1$ ($N = T - T_w$)인데 `len(iters)` $= N$으로 나눈다. 따라서 마지막 원소의 각도는 $\pi\frac{N-1}{N} < \pi$라서 값이 정확히 $v_{\text{final}}$이 아니라 **아주 살짝 못 미친다**. $N$이 수만이면 오차는 무시할 수준이며, "다음 스텝이 있었다면 $v_{\text{final}}$"이라는 반열린구간 해석으로 보면 오히려 자연스럽다.
2. **`warmup_epochs > epochs`이면 assert에서 죽는다.** `iters` 길이가 음수가 되어 `np.arange`가 빈 배열을 만들고, 전체 길이가 $T$에 못 미친다. 짧은 스모크 테스트에서는 `--warmup_epochs 0`을 반드시 줘야 한다.

또 하나: 스케줄이 **미리 계산된 배열**이라 스케줄러에 상태가 없다. 루프에서는 `schedule[it]`로 조회만 하므로 학습 재개(resume) 시 iteration 번호만 맞으면 값이 자동으로 정확하다.

---

## 10. 한 줄 요약

$\cos\theta$를 $[0,\pi]$에서 잘라 $\frac{1+\cos\theta}{2}$로 $[0,1]$ 비율로 만들고 → $v_{\text{final}} + (\cdot)(v_{\text{base}}-v_{\text{final}})$로 원하는 두 값 사이를 내분하고 → $\theta = \pi\frac{t-T_w}{T-T_w}$로 시간축을 각도축에 사상한 것. 앞의 warmup은 원점을 지나는 1차함수. 양 끝에서 도함수가 $0$이라 부드럽게 출발하고 부드럽게 착지한다.
