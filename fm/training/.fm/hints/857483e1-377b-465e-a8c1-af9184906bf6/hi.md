# `nn.utils.weight_norm` 은 가중치를 어떻게 재매개화하는가?

## 0. 한 줄 요약

행렬 $W$ 의 **각 행** $w_k$ 를

$$
w_k \;=\; g_k \,\frac{v_k}{\lVert v_k \rVert}
$$

처럼 **"길이 $g_k$" × "단위 방향벡터 $v_k/\lVert v_k\rVert$"** 로 쪼갠다.
학습되는 실제 파라미터는 더 이상 $w_k$ 가 아니라 $(g_k,\; v_k)$ 두 개다.

---

## 1. 출발점: 벡터를 "크기 × 방향"으로 쓰기 (극좌표 발상)

고등학교 기하에서 평면벡터 $\vec{a} = (3, 4)$ 를 이렇게 쓸 수 있다는 걸 배웠다.

$$
\vec{a} = 5 \cdot \left(\tfrac{3}{5},\ \tfrac{4}{5}\right)
= \lVert \vec{a}\rVert \cdot \hat{a}
$$

여기서

- $\lVert \vec a\rVert = \sqrt{3^2+4^2} = 5$ : **크기(스칼라)**
- $\hat a = \vec a / \lVert \vec a \rVert$ : 길이가 정확히 1인 **단위 방향벡터**

즉 벡터 하나는 "얼마나 긴가"와 "어느 쪽을 보는가" 두 정보로 완전히 나뉜다.
극좌표 $(r,\theta)$ 로 점을 표현하는 것과 같은 발상이다.

이 분해를 **학습 파라미터 수준에서** 그대로 하겠다는 것이 weight normalization
(Salimans & Kingma, 2016)의 전부다.

---

## 2. 행렬 $W$ 의 각 "행"이 벡터 하나다

$n$ 차원 입력 $u$ 를 받아 $K$ 차원 출력을 내는 선형층 $z = Wu$ 를 보자.
$W$ 는 $K \times n$ 행렬이고, 출력의 $k$ 번째 성분은

$$
z_k \;=\; w_k^{\top} u \;=\; \sum_{j=1}^{n} W_{kj}\, u_j
$$

여기서 $w_k$ 는 $W$ 의 **$k$ 번째 행**을 세로로 세운 $n$ 차원 벡터다.

핵심: **행 하나가 곧 벡터 하나**이고, 그 벡터는 "이 뉴런이 좋아하는 입력 패턴"
— DINO 용어로는 **프로토타입(prototype)** — 을 가리킨다.
$z_k$ 는 입력 $u$ 가 그 프로토타입과 얼마나 닮았는지를 재는 내적이다.

내적을 크기·방향으로 풀어 쓰면 고교 공식 그대로

$$
z_k = \lVert w_k\rVert\, \lVert u \rVert \cos\theta_k
$$

즉 $z_k$ 안에는 **"방향이 얼마나 맞는가($\cos\theta_k$)"** 와
**"이 행의 길이가 얼마나 큰가($\lVert w_k\rVert$)"** 가 곱해져 섞여 있다.
두 가지가 한 덩어리 $w_k$ 안에 뭉쳐 있으니, gradient가 하나를 고치면 다른 하나도 딸려 움직인다.

---

## 3. 재매개화: 파라미터를 $(g_k, v_k)$ 로 바꾸기

그래서 $w_k$ 를 직접 학습하지 말고, 새 파라미터 두 개를 두자.

$$
w_k \;=\; g_k\,\frac{v_k}{\lVert v_k\rVert}
\qquad
\begin{cases}
g_k \in \mathbb{R} & \text{스칼라 — 행의 길이} \\[2pt]
v_k \in \mathbb{R}^{n} & \text{벡터 — 행의 방향}
\end{cases}
$$

바로 확인되는 사실:

$$
\lVert w_k \rVert
= \left\lVert g_k \frac{v_k}{\lVert v_k\rVert}\right\rVert
= |g_k| \cdot \underbrace{\frac{\lVert v_k\rVert}{\lVert v_k\rVert}}_{=1}
= |g_k|
$$

**행의 길이는 오직 $g_k$ 가 결정한다.** $v_k$ 가 아무리 커지거나 작아져도
$w_k$ 의 길이는 안 변한다 — $v_k$ 는 **방향만** 담당하기 때문이다.
($v_k$ 를 2배로 늘려도 $v_k/\lVert v_k\rVert$ 는 그대로다. 이걸
"$w_k$ 는 $v_k$ 의 스케일에 대해 불변(scale-invariant)"이라고 한다.)

> 왜 $v_k$ 를 굳이 벡터 그대로 두고 단위벡터로 강제하지 않는가?
> "길이 1"이라는 제약이 걸린 파라미터는 경사하강법으로 직접 다루기 까다롭다.
> 대신 자유로운 $v_k$ 를 쓰고 나눗셈으로 정규화하면, 제약 없는 보통의 SGD/AdamW를
> 그대로 쓰면서도 결과적으로 방향/크기가 분리된다.

---

## 4. 왜 분리하면 학습에 유리한가 — gradient가 "수직 성분만" 남는다

체인 룰로 $L$ 을 $v_k$ 에 대해 미분하면 (첨자 $k$ 는 잠시 생략, $\hat v = v/\lVert v\rVert$)

$$
\nabla_{v} L
\;=\; \frac{g}{\lVert v\rVert}\Big(I - \hat v\,\hat v^{\top}\Big)\,\nabla_{w} L
$$

무섭게 생겼지만 고교 개념으로 완전히 읽힌다.

**(a) $\hat v \hat v^{\top} a$ 는 "$a$ 를 $\hat v$ 방향으로 정사영한 것"이다.**
정사영 공식을 떠올리자. 벡터 $a$ 의 $\hat v$ 방향 성분은

$$
(\text{$a$의 $\hat v$ 성분}) = (a\cdot \hat v)\,\hat v = \hat v\,(\hat v^{\top}a) = (\hat v \hat v^{\top})a
$$

**(b) 따라서 $(I - \hat v\hat v^{\top})a = a - (\text{$\hat v$ 방향 성분})$ 은 "$a$ 에서 $\hat v$ 방향 성분을 빼고 남은 것"** 이다.
어떤 벡터 $a$ 를 "$\hat v$ 와 나란한 성분 + $\hat v$ 와 수직인 성분"으로 분해했을 때,
나란한 성분을 지우고 **수직 성분만** 남긴 것이다. 실제로 이 결과와 $\hat v$ 의 내적은

$$
\hat v^{\top}\big(a - (\hat v^{\top}a)\hat v\big)
= \hat v^{\top}a - (\hat v^{\top}a)\underbrace{\hat v^{\top}\hat v}_{=1} = 0
$$

**정확히 0** — 즉 $\nabla_v L \perp v$ 가 항상 성립한다.

**(c) 의미.** $v$ 의 gradient는 $v$ 에 늘 수직이므로, 경사하강 한 스텝
$v \leftarrow v - \eta\,\nabla_v L$ 은 $v$ 를 **길게/짧게 하는 데 전혀 쓰이지 않고
오직 방향을 돌리는 데만 쓰인다**. (피타고라스로
$\lVert v - \eta\nabla_v L\rVert^2 = \lVert v\rVert^2 + \eta^2\lVert \nabla_v L\rVert^2 \ge \lVert v\rVert^2$
이니 $\lVert v\rVert$ 는 오히려 단조 증가하지만, 그 값은 $w$ 에 아무 영향이 없다.)

크기 쪽은 따로 깔끔한 식이 나온다.

$$
\frac{\partial L}{\partial g} = \hat v^{\top} \nabla_w L
$$

즉 **$g$ 는 gradient의 "나란한 성분", $v$ 는 "수직 성분"** 을 각각 받아간다.
$w$ 를 통째로 학습할 때는 이 둘이 한 벡터에 섞여 서로 간섭했지만, 재매개화 뒤에는
서로 다른 파라미터가 나눠 맡는다. 게다가 계수 $\dfrac{g}{\lVert v\rVert}$ 가
$\lVert v\rVert$ 가 커질수록 스텝을 자동으로 줄여줘서(일종의 자기 조절 학습률)
학습이 안정된다. 이것이 논문이 말하는 "최적화 지형의 조건수 개선" 효과다.

---

## 5. PyTorch에서 실제로 벌어지는 일

```python
lin = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
```

이 한 줄이 하는 일:

1. 원래 파라미터 `lin.weight` 를 **파라미터 목록에서 제거**한다.
2. 대신 새 파라미터 두 개를 등록한다.
   - `weight_g` : shape `(out_dim, 1)` — 행마다 스칼라 $g_k$ 하나
   - `weight_v` : shape `(out_dim, in_dim)` — 초기값은 원래 `weight` 그대로
   - 초기 `weight_g[k] = ` 원래 $\lVert w_k \rVert$ 로 잡아서, 시작 시점의 $W$ 는 변하지 않는다.
3. forward 직전 훅에서 매번 `weight` 를 **다시 계산**한다.
   ```python
   weight = weight_g * weight_v / weight_v.norm(dim=1, keepdim=True)
   ```
   그러니 `lin.weight` 는 이제 **파라미터가 아니라 파생된 값(버퍼/속성)** 이다.
   옵티마이저는 `weight_g`, `weight_v` 만 업데이트한다.

> 참고: PyTorch 2.1+ 에서는 `torch.nn.utils.parametrizations.weight_norm` 이
> 권장 API이고, 이쪽은 이름이 `parametrizations.weight.original0`(=$g$),
> `original1`(=$v$)로 바뀐다. DINO 코드는 옛 API를 써서 `weight_g`/`weight_v` 이름이 그대로 보인다.

---

## 6. DINO는 왜 $g_k = 1$ 로 고정하는가

`vision_transformer.py` 의 `DINOHead`:

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False   # g 를 학습에서 제외
```

$g_k = 1$ 을 넣고 얼려버리면 모든 행의 길이가 **정확히 1** 이 된다:
$\lVert w_k \rVert = |g_k| = 1$. 그리고 DINOHead는 직전에 입력도 L2 정규화한다
($\tilde u = \mathrm{MLP}(y)/\lVert \mathrm{MLP}(y)\rVert$, 즉 $\lVert \tilde u\rVert = 1$).
그러면 로짓은

$$
z_k = w_k^{\top}\tilde u
= \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert}
= \underbrace{\lVert w_k\rVert}_{=1}\underbrace{\lVert \tilde u\rVert}_{=1}\cos\theta_k
= \cos\angle(v_k,\ \tilde u) \;\in\; [-1,\,1]
$$

**로짓이 곧 $K$ 개 프로토타입 방향과의 코사인 유사도**이고, 구조적으로 $[-1,1]$ 에 갇힌다.

무엇을 얻는가:

- **스케일 폭주 차단.** $g_k$ 가 자유롭다면 특정 프로토타입 하나가 노름을 키워
  softmax를 독식할 수 있다. 길이를 1로 못 박으면 "누가 더 큰가" 경쟁이 사라지고
  **"누가 방향이 더 맞는가"** 만 남는다. 이것이 DINO 붕괴 방지 장치의 0번째 요소다.
- **온도가 온전히 온도 역할을 한다.** 로짓 범위가 $[-1,1]$ 로 고정이니
  $z/\tau$ 의 날카로움을 $\tau$(student 0.1 / teacher 0.04→0.07)가 단독으로 결정한다.
- **학습 대상이 명확해진다.** 마지막 층에서 배우는 것은 오직 $v_k$ 의 **방향** — 즉
  $K$ 개 프로토타입을 단위 구면 위 어디에 배치할지다.

한편 `main_dino.py` 의 `cancel_gradients_last_layer` 가 첫 1 epoch 동안 버리는 gradient는
이제 `head.last_layer.weight_v.grad` 다 (`weight_g` 는 애초에 `requires_grad=False` 라 grad가 없다).
노트북에서 그걸 이름으로 직접 꺼내 보는 이유가 이것이다.

---

## 7. 정리 표

| | 재매개화 전 | 재매개화 후 |
|---|---|---|
| 학습 파라미터 | $w_k \in \mathbb{R}^{n}$ | $g_k \in \mathbb{R}$, $v_k \in \mathbb{R}^{n}$ |
| 행의 길이 | $\lVert w_k\rVert$ (섞여 있음) | $\lvert g_k \rvert$ 만으로 결정 |
| gradient | $\nabla_w L$ 한 덩어리 | 나란한 성분 → $g$, 수직 성분 → $v$ |
| $v$ 스케일 | — | $w$ 에 영향 없음 (불변) |
| DINO 설정 | — | $g_k \equiv 1$ 고정 → 로짓 = 코사인 유사도 |
