# `weight_norm` — 가중치를 "크기 × 방향"으로 쪼개기

> **문제** `weight_norm`은 가중치를 어떻게 분해하는가?
>
> **답** 각 행을 $w_k = g_k \dfrac{v_k}{\lVert v_k \rVert}$ 로 크기 $g_k$ 와 방향으로 나눈다. DINO는 $g_k = 1$ 로 채우고 `norm_last_layer=True` 면 학습에서 제외한다.

이 답 한 줄에는 고교 기하 하나(벡터의 크기·방향 분해)와 대학 수준 아이디어 하나(재매개화가 최적화를 돕는다)가 겹쳐 있다. 아래에서 고교 개념부터 쌓아 올린다.

---

## ① 출발점: 벡터는 이미 "크기 × 방향"이다

고교 기하에서 배운 그대로다. 영벡터가 아닌 벡터 $\vec v$ 에 대해

$$
\hat v = \frac{\vec v}{\lvert \vec v \rvert}
$$

는 **단위벡터**($\lvert \hat v \rvert = 1$)이고, 원래 벡터는 언제나

$$
\vec v = \underbrace{\lvert \vec v \rvert}_{\text{크기(스칼라)}} \cdot \underbrace{\hat v}_{\text{방향(단위벡터)}}
$$

로 쓸 수 있다. 예를 들어 $\vec v = (3, 4)$ 라면 $\lvert \vec v \rvert = \sqrt{3^2+4^2} = 5$, $\hat v = (0.6,\, 0.8)$, 그리고 $(3,4) = 5 \cdot (0.6, 0.8)$.

`weight_norm`의 아이디어는 여기서 한 발짝만 더 나간 것이다. **크기를 원래 벡터의 길이 $\lvert \vec v\rvert$ 로 두지 말고, 별도의 자유로운 숫자 $g$ 로 바꿔 끼우자**:

$$
\vec w = g \cdot \hat v = g\,\frac{\vec v}{\lvert \vec v\rvert}
$$

이제 $\vec w$ 의 길이는 정확히 $\lvert g \rvert$ 다($\hat v$ 의 길이가 1이므로). 즉 $g$ 는 "길이 손잡이", $\vec v$ 는 "방향 손잡이"가 되고, 두 손잡이가 서로 간섭하지 않는다. $\vec v$ 를 2배로 늘려도 $\hat v$ 는 그대로이므로 $\vec w$ 는 전혀 변하지 않는다.

---

## ② 왜 "행"인가: 선형층의 각 행은 벡터다

신경망의 선형층(`nn.Linear`, bias 없음)은 행렬 곱

$$
z = W u, \qquad W \in \mathbb{R}^{K \times d},\ u \in \mathbb{R}^{d}
$$

이다. 행렬 곱의 정의를 풀어 쓰면, 출력의 $k$ 번째 성분은 $W$ 의 **$k$ 번째 행** $w_k \in \mathbb{R}^d$ 와 입력 $u$ 의 내적이다.

$$
z_k = \sum_{j=1}^{d} W_{kj} u_j = w_k \cdot u = w_k^{\top} u
$$

$$
W = \begin{pmatrix} \text{---}\ w_1^{\top}\ \text{---} \\ \text{---}\ w_2^{\top}\ \text{---} \\ \vdots \\ \text{---}\ w_K^{\top}\ \text{---}\end{pmatrix}
\;\Longrightarrow\;
Wu = \begin{pmatrix} w_1 \cdot u \\ w_2 \cdot u \\ \vdots \\ w_K \cdot u\end{pmatrix}
$$

그러니까 **$K \times d$ 행렬은 $d$차원 벡터 $K$ 개의 묶음**으로 보는 게 맞다. 출력 뉴런 하나마다 벡터 하나가 대응하고, 그 벡터와 입력의 내적이 그 뉴런의 값(로짓)이다. DINO에서는 이 $K$ 개의 행을 **프로토타입(prototype)** 이라 부른다 — "이런 방향의 특징이면 $k$ 번 클러스터"라고 주장하는 대표 벡터들이다.

행이 각각 벡터이므로, ①의 분해를 **행마다 따로** 적용하는 것이 자연스럽다. 이것이 답에 등장하는 아래첨자 $k$ 의 정체다.

$$
\boxed{\;w_k = g_k \frac{v_k}{\lVert v_k \rVert}, \qquad k = 1, \dots, K\;}
$$

$K$ 개의 방향 벡터 $v_1,\dots,v_K$ (합쳐서 행렬 $V \in \mathbb{R}^{K\times d}$)와 $K$ 개의 스칼라 $g_1,\dots,g_K$ (합쳐서 열벡터 $g \in \mathbb{R}^{K \times 1}$)가 학습 파라미터가 되고, 실제 쓰이는 $W$ 는 이 둘로부터 **매번 계산**된다.

> 파라미터 개수는 $Kd \to Kd + K$ 로 늘지만 표현할 수 있는 $W$ 의 집합은 그대로다(단, $v_k = 0$ 은 제외). 자유도가 하나 남는(중복인) 재매개화다 — $v_k$ 를 상수배해도 $w_k$ 가 안 바뀌므로.

---

## ③ 재매개화가 왜 최적화에 도움이 되는가

이 기법의 원 논문은 Salimans & Kingma, *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks* (NeurIPS 2016)다. 핵심 동기는 이렇다.

**보통 선형층의 곤란함.** 경사하강법은 $w \leftarrow w - \eta\, \nabla_w L$ 로 갱신한다. 그런데 $w$ 안에는 "크기"와 "방향" 정보가 뒤엉켜 있다. 어떤 층의 가중치 노름이 커지면 그 층의 출력이 커지고, 역전파로 흐르는 기울기의 크기도 함께 달라진다. 결과적으로 **하나의 학습률 $\eta$ 가 "방향을 얼마나 돌릴지"와 "크기를 얼마나 바꿀지"를 동시에 담당**하게 되고, 층마다·행마다 적절한 $\eta$ 가 달라져 튜닝이 까다로워진다.

**분해하면 어떻게 되는가.** $w = g\,v/\lVert v\rVert$ 로 두고 연쇄법칙을 적용하면(미적분: 합성함수의 미분) 두 손잡이의 기울기가 이렇게 갈린다.

$$
\nabla_g L = \nabla_w L \cdot \hat v,
\qquad
\nabla_v L = \frac{g}{\lVert v \rVert}\Big(\nabla_w L - (\nabla_w L \cdot \hat v)\,\hat v\Big)
$$

<details>
<summary>심화: 두 식이 나오는 과정</summary>

$g$ 에 대해서는 $w = g\hat v$ 에서 $\partial w/\partial g = \hat v$ 이므로 $\nabla_g L = \nabla_w L \cdot \hat v$ 로 바로 나온다.

$v$ 에 대해서는 $\lVert v\rVert = \sqrt{v\cdot v}$ 를 미분해 $\nabla_v \lVert v\rVert = \hat v$ 를 쓰고, $w = g\,v\,\lVert v\rVert^{-1}$ 에 곱의 미분법을 적용하면
$$
\frac{\partial w_i}{\partial v_j} = \frac{g}{\lVert v\rVert}\left(\delta_{ij} - \hat v_i \hat v_j\right)
$$
가 되어 위 식이 된다. 괄호 안의 $\left(I - \hat v\hat v^{\top}\right)$ 는 고교 기하에서 배운 **정사영의 여(餘)** — 벡터에서 $\hat v$ 방향 성분을 빼는 연산이다.
</details>

여기서 두 가지가 읽힌다.

1. **$\nabla_v L$ 은 항상 $v$ 에 수직이다.** 실제로 $\nabla_v L \cdot \hat v = \frac{g}{\lVert v\rVert}\big((\nabla_w L\cdot\hat v) - (\nabla_w L\cdot\hat v)\cdot 1\big) = 0$. 즉 $v$ 갱신은 순수하게 **방향만 회전**시킨다. 크기 변경은 $g$ 가 전담한다. 두 역할이 깔끔히 분리됐다.
2. **$\nabla_v L$ 에 $1/\lVert v\rVert$ 가 곱해져 있다.** $v$ 에 수직인 방향으로 스텝을 밟으면 피타고라스 정리에 의해 $\lVert v\rVert$ 는 항상 조금 **늘어난다**. 그런데 $\lVert v\rVert$ 가 늘면 다음 스텝의 기울기가 작아진다 — **노름 증가가 자동 학습률 감쇠처럼 작동**해서 스스로 안정화된다. (논문의 주장 중 하나가 이것이다.)

즉 `weight_norm`은 "성능을 올리는 층"이 아니라 **같은 함수를 더 학습하기 쉬운 좌표계로 다시 쓴 것**이다. 배치 통계를 쓰지 않으므로 BatchNorm과 달리 배치 크기·분산 환경에 의존하지 않는다는 점도 장점이다.

---

## ④ $g_k = 1$ 로 고정하면: 로짓이 코사인 유사도가 된다

DINO는 이 손잡이를 **일부러 잠근다**. 모든 $g_k = 1$ 이면

$$
w_k = 1 \cdot \frac{v_k}{\lVert v_k\rVert} = \hat v_k
$$

즉 **$W$ 의 모든 행이 단위벡터**가 된다. 출력은

$$
z_k = w_k^{\top} \tilde u = \hat v_k \cdot \tilde u
$$

이제 고교에서 배운 내적의 기하적 정의를 그대로 쓴다.

$$
\vec a \cdot \vec b = \lvert \vec a\rvert\,\lvert \vec b\rvert \cos\theta
$$

$\lvert \hat v_k \rvert = 1$ 이고, DINO는 `forward`에서 입력도 L2 정규화해서 넣으므로($\lVert \tilde u\rVert = 1$, `nn.functional.normalize(x, dim=-1, p=2)`) 두 길이가 모두 1이다. 따라서

$$
\boxed{\;z_k = 1 \cdot 1 \cdot \cos\theta_k = \cos\angle(v_k,\ \tilde u) \in [-1,\, 1]\;}
$$

로짓이 **프로토타입 방향과 특징 방향 사이 각도의 코사인** 그 자체다. 크기 정보는 양쪽에서 모두 제거됐고 순수한 "방향 유사도"만 남는다. 기하적으로는, 특징과 프로토타입 모두 **단위 초구(hypersphere) 표면**에 올라앉고 로짓은 그 위의 각거리를 재는 것이다.

---

## ⑤ 왜 DINO는 굳이 고정하는가 — 붕괴(collapse) 방지

DINO는 정답 레이블 없이 학습한다. 학생 네트워크가 교사 네트워크의 출력 분포를 따라가는데, 그 분포는 위 로짓에 소프트맥스를 씌워 만든다.

$$
p_k = \frac{\exp(z_k / \tau)}{\sum_{j} \exp(z_j/\tau)}
$$

만약 $g_k$ 가 자유롭게 학습된다면, 어떤 프로토타입 하나가 **방향은 그대로 두고 $g_k$ 만 키워서** 자기 로짓을 얼마든지 크게 만들 수 있다. 소프트맥스는 상대적 크기만 보므로, 노름을 키운 그 프로토타입이 모든 입력에 대해 확률을 독식하고 — 모든 이미지가 같은 클러스터로 배정되는 **붕괴**로 직행한다. 게다가 DINO의 학생 온도는 $\tau = 0.1$ 로 작아서 로짓 차이가 10배로 증폭돼 이 경로가 더 위험하다.

$g_k = 1$ 로 고정하면 $z_k \in [-1,1]$ 이라는 상한이 **구조적으로** 걸린다. 최적화가 아무리 애써도 노름을 키워 로짓을 독식하는 길이 애초에 존재하지 않는다. 워크스루가 이를 "붕괴 방지의 0번째 장치"라고 부르는 이유다(centering·sharpening보다 앞서, 구조 자체로 막는다는 뜻).

프로토타입들은 이제 노름 경쟁이 아니라 **초구 표면에서 서로 다른 방향을 차지하려는 경쟁**만 하게 된다.

---

## ⑥ 손으로 계산해 보기 ($K=2$, $d=2$)

방향 파라미터를 이렇게 두자.

$$
V = \begin{pmatrix} 3 & 4 \\ 0 & 1 \end{pmatrix}, \qquad
v_1 = (3,4),\quad v_2 = (0,1)
$$

입력 특징은 이미 정규화된 $\tilde u = (0, 1)$ (길이 1).

**단계 1 — 각 행의 크기와 방향**

| $k$ | $v_k$ | $\lVert v_k\rVert$ | $\hat v_k = v_k/\lVert v_k\rVert$ | $g_k$ (DINO) | $w_k = g_k\hat v_k$ |
|---|---|---|---|---|---|
| 1 | $(3,\,4)$ | $\sqrt{9+16}=5$ | $(0.6,\,0.8)$ | $1$ | $(0.6,\,0.8)$ |
| 2 | $(0,\,1)$ | $\sqrt{0+1}=1$ | $(0,\,1)$ | $1$ | $(0,\,1)$ |

**단계 2 — 로짓: 정규화 전(그냥 `nn.Linear`, $W=V$) vs 정규화 후($W$ 의 행이 단위벡터)**

| $k$ | 정규화 전 $v_k\cdot\tilde u$ | 정규화 후 $w_k\cdot\tilde u = \cos\theta_k$ | 각도 $\theta_k$ |
|---|---|---|---|
| 1 | $3\cdot 0 + 4\cdot 1 = \mathbf{4}$ | $0.6\cdot 0 + 0.8 \cdot 1 = \mathbf{0.8}$ | $\arccos 0.8 \approx 36.87^\circ$ |
| 2 | $0\cdot 0 + 1\cdot 1 = \mathbf{1}$ | $0\cdot 0 + 1\cdot 1 = \mathbf{1.0}$ | $0^\circ$ |

**여기가 핵심이다.** 방향만 보면 프로토타입 2가 입력과 **완벽히 일치**한다($\theta_2 = 0^\circ$). 그런데 정규화 전에는 프로토타입 1이 로짓 4로 이긴다 — 이유는 순전히 $\lVert v_1\rVert = 5$ 가 $\lVert v_2\rVert=1$ 보다 5배 크기 때문이다. 정규화 후에는 순위가 바로잡혀 프로토타입 2가 이긴다.

**단계 3 — 소프트맥스($\tau = 0.1$)까지 가면 차이가 극단적이다**

| | $z/\tau$ | $p_1$ | $p_2$ |
|---|---|---|---|
| 정규화 전 | $(40,\ 10)$ | $\approx 1.0000$ | $\approx 9.4\times 10^{-14}$ |
| 정규화 후 ($g=1$) | $(8,\ 10)$ | $0.1192$ | $0.8808$ |

정규화 전에는 노름이 큰 프로토타입 1이 확률을 100% 독식한다 — 붕괴가 이런 식으로 시작된다. 정규화 후에는 $\lvert z_1 - z_2\rvert \le 2$ 라는 상한 덕분에 확률비도 최대 $e^{2/\tau}=e^{20}$ 로 제한되어, 다른 프로토타입에도 기울기가 계속 흐른다.

> 위 수치는 `torch 2.4.0`에서 그대로 재현된다: `weight_v`에 $V$ 를 넣고 `weight_g`를 1로 채운 뒤 $(0,1)$ 을 통과시키면 `tensor([[0.8000, 1.0000]])` 이 나온다.

---

## PyTorch에서의 실제 모양

`nn.utils.weight_norm(module, name='weight', dim=0)` 은 다음을 한다.

1. `module.weight` 를 `Parameter` 목록에서 **제거**한다.
2. 대신 `weight_g` (모양 $(K, 1)$ — 행마다 스칼라 하나)와 `weight_v` (모양 $(K, d)$ — 원래 가중치와 동일)를 `Parameter` 로 등록한다. 초기값은 원래 가중치의 노름과 원래 가중치 자체다.
3. **forward pre-hook** 을 걸어, `forward`가 불릴 때마다 `module.weight = weight_g * weight_v / ||weight_v||` 를 다시 계산한다(평범한 속성으로 대입). 그래서 `weight`는 파라미터가 아니라 **매 forward마다 만들어지는 중간 결과**이고, 역전파는 자동으로 `weight_g`·`weight_v` 까지 흘러간다.

`dim=0` 이 "행마다"의 정체다. `nn.Linear`의 가중치는 $(\text{out\_features}, \text{in\_features})$ 이므로 0번 축을 남기고 나머지를 노름 내면 정확히 **출력 뉴런(= 프로토타입)별 노름**이 된다.

DINO의 코드(`/home/sungwoo/projects/swcho/dino/vision_transformer.py`, `DINOHead.__init__`)는 이렇게 쓴다.

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

세 줄이 각각 ②③④⑤에 대응한다.

- 1행: $256 \times K$ 선형층에 `weight_norm` 을 걸어 행별 $(g_k, v_k)$ 분해를 만든다. `bias=False` 도 중요하다 — bias가 있으면 $z_k = \cos\theta_k + b_k$ 가 되어 $[-1,1]$ 상한이 깨진다.
- 2행: `weight_g` 를 전부 1로 덮어쓴다(초기값은 원래 노름이었으므로 반드시 필요). 이 순간 모든 프로토타입이 단위벡터가 된다.
- 3행: `requires_grad = False` — 기울기 계산 대상에서 빼서 **영구히 1로 잠근다**. `norm_last_layer` 의 기본값은 `True`.

그리고 `forward`가 입력 쪽 정규화를 담당한다.

```python
def forward(self, x):
    x = self.mlp(x)
    x = nn.functional.normalize(x, dim=-1, p=2)   # ‖ũ‖ = 1
    x = self.last_layer(x)                        # z_k = cos θ_k
    return x
```

**세부 사항 두 개.**

- `norm_last_layer=False` 를 주면 `weight_g` 가 학습된다. DINO 저자들은 **큰 배치를 쓰는 convnet(ResNet)에서만** 풀라고 권고한다. ViT에서는 위에서 본 노름 독식 경로가 열려 학습이 불안정해진다.
- `torch` 최신 버전에서 `nn.utils.weight_norm` 은 `torch.nn.utils.parametrizations.weight_norm` 에 밀려 **deprecated** 다(위 실행에서도 `FutureWarning` 이 뜬다). 새 API는 같은 수학을 `parametrize` 프레임워크로 구현하며 파라미터 이름이 `parametrizations.weight.original0`(=$g$)·`original1`(=$v$)로 바뀐다. DINO 코드는 구 API를 쓰므로 워크스루도 그 경고를 억제한다(`vision_transformer_walkthrough.py` 상단의 `warnings.filterwarnings`).

---

## 한 줄 요약

| 개념 | 식 | 역할 |
|---|---|---|
| 고교 벡터 분해 | $\vec v = \lvert\vec v\rvert\,\hat v$ | 크기와 방향은 분리 가능 |
| 행 = 벡터 | $z_k = w_k \cdot u$ | 출력 뉴런 하나 ↔ 프로토타입 하나 |
| `weight_norm` | $w_k = g_k\,v_k/\lVert v_k\rVert$ | 크기($g_k$)와 방향($v_k$)을 **따로 학습** → 최적화 안정화 |
| DINO의 선택 | $g_k \equiv 1$, `requires_grad=False` | 모든 행을 단위벡터로 잠금 |
| 결과 | $z_k = \cos\theta_k \in [-1,1]$ | 로짓 = 코사인 유사도, 노름 독식형 붕괴 원천 차단 |
