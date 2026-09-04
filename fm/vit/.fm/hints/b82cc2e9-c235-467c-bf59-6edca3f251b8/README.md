# `DINOHead` 로짓이 코사인 유사도가 되는 유도

## 0. 결론 먼저

`DINOHead`가 뱉는 $K$차원 벡터 $z$의 각 성분은 **가중치가 아니라 각도**다.

$$
z_k \;=\; \cos\angle\!\left(v_k,\ \tilde u\right)\ \in\ [-1,\,1],
\qquad k = 1,\dots,K
$$

여기서 $\tilde u$는 MLP 출력을 L2 정규화한 벡터, $v_k$는 마지막 층의 $k$번째 행(프로토타입)이다.
즉 로짓 벡터는 "입력 표현이 $K$개 프로토타입 방향과 각각 얼마나 정렬됐는가"의 목록이다.

---

## 1. 코드 근거 두 조각

### (a) `forward` — 입력을 단위 초구로 투영

DINO 저장소 `vision_transformer.py`의 `DINOHead.forward`:

```python
def forward(self, x):
    x = self.mlp(x)                                  # (B, in_dim) → (B, 256)
    x = nn.functional.normalize(x, dim=-1, p=2)      # ★ 하이퍼구 S^255 로 투영
    x = self.last_layer(x)                           # (B, 256) → (B, K)
    return x
```

두 번째 줄이 핵심이다. `bottleneck_dim=256` 차원 벡터 $u = \mathrm{MLP}(y)$를

$$
\tilde u \;=\; \frac{u}{\lVert u \rVert_2}
\quad\Longrightarrow\quad
\lVert \tilde u \rVert_2 = 1
$$

로 만들어 **크기 정보를 버리고 방향만 남긴다**. 이후 `last_layer`가 보는 입력은 항상 단위 벡터다.

### (b) `__init__` — 프로토타입의 크기를 1로 고정

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
self.last_layer.weight_g.data.fill_(1)
if norm_last_layer:
    self.last_layer.weight_g.requires_grad = False
```

`nn.utils.weight_norm`은 가중치 행렬의 각 행을 **크기(gain) $\times$ 방향**으로 재파라미터화한다.

$$
w_k \;=\; g_k \,\frac{v_k}{\lVert v_k \rVert_2}
$$

- `weight_v` $= v_k$ — 학습되는 방향 파라미터 (`shape = (K, 256)`)
- `weight_g` $= g_k$ — 행별 스칼라 크기 (`shape = (K, 1)`)

DINO는 `weight_g.data.fill_(1)`로 **모든 $g_k = 1$** 로 채우고, 기본값 `norm_last_layer=True`이면
`requires_grad = False`로 학습에서 아예 제외한다. 결과적으로 실효 가중치 행 $w_k$는
**영구히 단위 벡터**다.

$$
\lVert w_k \rVert_2 = g_k \cdot \frac{\lVert v_k\rVert}{\lVert v_k\rVert} = g_k = 1
$$

또 `bias=False`라서 로짓에 상수 항조차 더해지지 않는다.

---

## 2. 유도 세 줄

두 조각을 합치면 유도는 세 줄로 끝난다.

**1) 선형층 정의** — bias가 없으니 로짓은 순수 내적이다.

$$
z_k \;=\; w_k^{\top} \tilde u
$$

**2) `weight_norm` + $g_k=1$ 대입** — 프로토타입이 단위 벡터로 치환된다.

$$
z_k \;=\; \left(1 \cdot \frac{v_k}{\lVert v_k\rVert}\right)^{\!\top} \tilde u
\;=\; \frac{v_k^{\top}\tilde u}{\lVert v_k\rVert}
$$

**3) 내적의 기하 정의 적용** — $a^\top b = \lVert a\rVert\lVert b\rVert\cos\theta$ 이고 $\lVert\tilde u\rVert = 1$ 이므로
분모의 $\lVert v_k \rVert$ 가 정확히 약분된다.

$$
z_k \;=\; \frac{\lVert v_k\rVert \,\lVert \tilde u\rVert \cos\angle(v_k,\tilde u)}{\lVert v_k\rVert}
\;=\; \cos\angle(v_k,\ \tilde u) \;\in\; [-1,\,1]
$$

**"둘 다 단위 벡터 → 내적 = 코사인"** 이 전부다. 한쪽이라도 정규화가 빠지면 성립하지 않는다.

### 행렬 형태

배치 전체를 한 번에 보면 $\tilde U \in \mathbb{R}^{B\times 256}$, $\hat V \in \mathbb{R}^{K \times 256}$ (행 정규화) 에 대해

$$
Z \;=\; \tilde U\,\hat V^{\top} \;\in\; \mathbb{R}^{B\times K}
$$

즉 `last_layer`의 GEMM 한 번이 곧 $B \times K$ 코사인 유사도 표다. 워크스루의 검증 셀이
바로 이걸 확인한다.

```python
protos = F.normalize(head.last_layer.weight_v, dim=-1, p=2)   # (K, 256)
cos = un @ protos.t()
assert torch.allclose(cos, z, atol=1e-4)      # 로짓 == 코사인 ✔
assert z.abs().max() <= 1.0 + 1e-4            # 범위도 [-1,1] ✔
```

---

## 3. 결과: 로짓이 $[-1,1]$ 에 갇힌다

이건 초기화 시점의 우연이 아니라 **구조적 제약**이다. 학습이 $v_k$를 어떻게 움직여도
$\lVert v_k\rVert$ 는 약분되어 사라지므로, 어떤 프로토타입도 노름을 키워 로짓을 키울 수 없다.
표현력은 오직 **방향**에만 있다.

### 온도 $\tau$ 와의 상호작용

DINO 손실은 로짓에 softmax를 씌워 분포로 만든다 (`main_dino.py` `DINOLoss.forward`):

```python
student_out = student_output / self.student_temp                 # student_temp = 0.1
teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)   # temp = 0.04 ~ 0.07
```

로짓 범위가 $[-1,1]$ 이면 **softmax를 그대로 씌울 때 분포가 거의 균등해진다.** 최대·최소
로짓 차이가 2뿐이므로 확률비의 상한이 $e^{2} \approx 7.4$ 인데, $K = 65536$ 개 중 7.4배 차이는
사실상 평평한 분포다. 학습 신호(교차 엔트로피 그래디언트)가 죽는다.

온도로 나누는 것이 이 스케일 문제를 해결한다.

| | $\tau$ | 나눈 뒤 로짓 범위 | 확률비 상한 |
|---|---|---|---|
| softmax 직접 | $1$ | $[-1, 1]$ | $e^{2}\approx 7.4$ |
| student | $0.1$ | $[-10, 10]$ | $e^{20}\approx 4.9\times10^{8}$ |
| teacher | $0.04$ | $[-25, 25]$ | $e^{50}\approx 5\times10^{21}$ |

- **student $\tau_s = 0.1$** → $\pm 1/0.1 = \pm 10$. 유의미하게 뾰족하지만 여전히 부드러운 분포.
- **teacher $\tau_t = 0.04\sim0.07$** → 훨씬 더 뾰족(sharpening). 학생보다 자신감 있는 타깃을
  만들어 학생이 따라갈 방향을 준다. 워밍업 스케줄(`warmup_teacher_temp` $0.04$ → `teacher_temp`)로
  초기에 너무 날카로운 타깃이 나오지 않게 조절한다.

역으로 읽으면 **온도는 "코사인 스케일을 로짓 스케일로 환산하는 계수"** 다. 헤드가 로짓 범위를
구조적으로 고정해 줬기 때문에, $\tau$ 값이 학습 내내 의미가 변하지 않는 안정된 하이퍼파라미터가
된다. 노름이 자유로운 일반 분류기라면 로짓 스케일이 학습 중 표류해서 고정 $\tau$가 의미를 잃는다.

---

## 4. 프로토타입 $v_k$ 를 "클러스터 중심 방향"으로 읽기

$z_k$ 가 코사인이면 $\arg\max_k z_k$ 는 **$\tilde u$ 에 각도상 가장 가까운 $v_k$** 다. 이건 단위
초구 위의 최근접 중심 할당, 즉 **구면 k-means의 소프트 버전**이다.

- $\hat v_k = v_k/\lVert v_k\rVert$ : 초구 $S^{255}$ 위의 한 점 = $k$번째 클러스터 중심 방향
- softmax$(z/\tau)$ : 하드 할당 대신 **소프트 클러스터 할당 분포**
- 학습 목표: 같은 이미지의 두 뷰가 같은 소프트 할당을 갖게 하기 (뷰 불변 클러스터링)

따라서 DINO 학습은 "라벨 없이 $K$개 클러스터에 소프트 할당하되, 그 할당이 크롭·색·스케일
변화에 불변이도록" 표현을 학습하는 문제로 읽힌다. `bottleneck_dim=256` 이 이 초구의 차원이고,
$K$ 개 중심이 그 안에 흩어져 있다.

### out_dim $K = 65536$ 의 의미

- $K$ 는 클래스 수가 아니라 **코드북(어휘) 크기**다. ImageNet의 1000 클래스와 무관하게
  $2^{16} = 65536$ 을 쓴다.
- 크게 잡는 이유: 클러스터가 많으면 각 프로토타입이 미세한 시각적 개념(자세, 부위, 텍스처)에
  특화될 수 있고, 표현이 더 세밀해진다. 논문의 ablation에서 $K$ 를 키우면 성능이 오르다
  포화된다.
- 비용: 마지막 층은 $256 \times K$ 다. $K = 65536$ 이면 $16.8\text{M}$ 파라미터 —
  ViT-S backbone($21.7\text{M}$)에 맞먹는 크기가 헤드 한 층에 들어간다. 그래도 **학습이 끝나면
  헤드는 버리고 backbone만 쓴다.**
- 그래디언트/메모리 관점에선 배치당 $B\times K$ 로짓 행렬이 실체화되므로 $K$ 가 크롭 수와
  곱해져 메모리를 먹는다.

---

## 5. 붕괴 방지 관점 (요점만 — 별도 카드 있음)

로짓의 $[-1,1]$ 고정은 붕괴 방지의 **"0번째 장치"** 다.

- 노름을 키워 특정 프로토타입이 로짓을 독식하는 경로가 **원천 차단**된다. 붕괴가 일어난다면
  방향 정렬을 통해서만 가능하고, 이는 centering이 감시할 수 있는 형태다.
- 실제 주 장치는 **centering(teacher 로짓에서 EMA 평균 $c$ 를 빼 균등화 압력)** 과
  **sharpening(teacher $\tau_t$ 로 뾰족하게, 균등 붕괴 방지)** 의 균형이다. 이 둘은 서로 반대
  방향으로 작용해 서로를 견제한다.
- 헤드가 스케일을 잠가 두었기 때문에 centering의 "빼기"가 스케일 표류 없이 일관된 의미를 갖는다.

### `norm_last_layer=False` 로 풀면?

```python
head2 = DINOHead(in_dim=D, out_dim=4096, norm_last_layer=False)
# → weight_g.requires_grad == True
```

$g_k$ 가 학습되면 $z_k = g_k\cos\angle(v_k,\tilde u) \in [-g_k, g_k]$ 로 **범위가 풀린다.**
코사인 해석은 방향 부분에 남지만 스케일 보장이 사라져 $\tau$ 의 의미도 흔들린다. DINO는
이걸 "convnet + 큰 배치일 때만 풀어 보라"고 권고하고, **ViT에서는 불안정**하다고 명시한다.

---

## 6. 한 줄 정리

> `weight_norm`의 $g_k$ 를 1로 못박아 프로토타입을 단위 벡터로 만들고, 입력도 L2 정규화해
> 단위 벡터로 만들면, bias 없는 선형층의 내적은 정확히 $\cos\angle(v_k,\tilde u)$ 다.
> 로짓은 $[-1,1]$ 에 갇히고, 그 좁은 범위를 온도 $\tau$ 로 나눠 분포를 뾰족하게 만든다.
