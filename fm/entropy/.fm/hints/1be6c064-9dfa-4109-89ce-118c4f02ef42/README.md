# 원-핫 붕괴에서 $h(q)$와 배치 평균 분포의 엔트로피

## 0. 한 줄 답

**둘 다 0으로 간다.** 모든 입력이 같은 차원 하나로 몰리므로 개별 분포도 뾰족하고($h=0$), 그것들의 평균도 여전히 같은 원-핫이라 평균 분포의 엔트로피도 0이다.

하지만 이 카드의 진짜 값어치는 "둘 다 0"이라는 사실 자체가 아니라, **두 양이 항상 한쪽 방향의 부등식으로 묶여 있고 그 간격이 정확히 상호정보량이라는 것**이다. 그걸 알고 나면 원-핫 붕괴, 균등 붕괴, 건강한 상태가 하나의 그림으로 정리된다.

---

## 1. 두 양을 기호로 정의한다

teacher가 입력 $x$에 대해 낸 확률분포를 $q(x) \in \Delta^{K-1}$라 하자. 엔트로피는 노트북 1절 그대로

$$h(p) = -\sum_{i=1}^{K} p_i \log p_i, \qquad 0 \le h(p) \le \log K.$$

여기서 **두 가지 다른 것**을 잴 수 있다.

$$\boxed{\;h_{\text{each}} \;=\; \mathbb{E}_x\!\left[h\big(q(x)\big)\right]\;}
\qquad\qquad
\boxed{\;h_{\text{mean}} \;=\; h\!\left(\mathbb{E}_x\big[q(x)\big]\right) \;=\; h(\bar q)\;}$$

- $h_{\text{each}}$ — "점 하나하나가 **확신하는가**". 각 분포의 엔트로피를 먼저 재고, 그 다음 평균.
- $h_{\text{mean}}$ — "배치 전체가 **차원을 골고루 쓰는가**". 분포들을 먼저 평균 내고($\bar q = \frac{1}{N}\sum_n q(x_n)$), 그 다음 엔트로피.

코드로는 노트북 5절의 두 줄이 전부다:

```python
hist["h_each"].append(entropy(q).mean().item())   # E_x[ h(q(x)) ]
hist["h_mean"].append(entropy(q.mean(0)).item())  # h( E_x[q(x)] )
```

`entropy(...)`와 `.mean(...)`의 **순서만 뒤바뀐 것**인데 재는 대상이 전혀 다르다. `q`가 `(N, K)`일 때 `.mean(0)`은 배치 축을 접고, `entropy`는 클래스 축을 접는다. 어느 축을 먼저 접느냐가 "개별 확신"과 "전체 사용량"을 가른다.

---

## 2. 핵심 부등식: $h_{\text{mean}} \ge h_{\text{each}}$ (항상)

### 증명

엔트로피 $h(\cdot)$는 확률 심플렉스 위에서 **오목(concave)** 함수다. 실제로 $h(p) = \sum_i \phi(p_i)$이고 $\phi(t) = -t\log t$는 $\phi''(t) = -1/t < 0$이므로 각 항이 강오목, 따라서 합도 강오목이다.

오목 함수에 대한 Jensen 부등식은 "평균의 함수 $\ge$ 함수의 평균"이므로, $q(x)$를 랜덤 벡터로 보면

$$h\big(\mathbb{E}_x[q(x)]\big) \;\ge\; \mathbb{E}_x\big[h(q(x))\big]
\qquad\Longleftrightarrow\qquad
h_{\text{mean}} \;\ge\; h_{\text{each}}.$$

**섞으면 엔트로피는 절대 줄지 않는다.** 서로 다른 뾰족한 분포들을 평균 내면 반드시 더 퍼진 분포가 나오기 때문이다.

### 등호 조건 — 이게 곧 붕괴다

$\phi$가 **강**오목이므로 Jensen의 등호는 $q(x)$가 확률 1로 상수일 때만 성립한다:

$$h_{\text{mean}} = h_{\text{each}}
\quad\Longleftrightarrow\quad
q(x) = q^\star \ \ \text{for all } x.$$

즉 **모든 입력이 같은 출력 분포를 낸다** — 정확히 붕괴의 정의다. 그러니 이렇게 읽어야 한다:

> 두 엔트로피가 **같아지는 것 자체가 붕괴 신호**다. 어느 값이 크고 작은지가 아니라, **둘이 붙는지**를 봐야 한다.

노트북 5절이 "`h_each`와 `h_mean`은 **짝으로** 봐야 한다"고 말하는 이유가 이것이다. 둘 다 낮아서 붙으면 원-핫 붕괴, 둘 다 $\log K$ 근처에서 붙으면 균등 붕괴다.

---

## 3. 그 간격은 무엇인가 — $h_{\text{mean}} - h_{\text{each}} = I(X;Z)$

Jensen은 "간격이 0 이상"이라고만 말한다. 그 간격의 **정체**를 알면 그림이 완성된다.

### 유도

확률변수 두 개를 세운다.

- $X$: 배치에서 균등하게 뽑은 입력 (한 점 $x_n$이 확률 $1/N$).
- $Z \in \{1,\dots,K\}$: **코드**(prototype 인덱스). 조건부 분포를 $\Pr[Z = i \mid X = x] = q_i(x)$로 둔다. 즉 $q(x)$를 조건부 분포로 해석한다.

그러면 정의에 의해 곧바로

$$H(Z \mid X) \;=\; \mathbb{E}_x\big[h(q(x))\big] \;=\; h_{\text{each}},$$

$$\Pr[Z=i] = \mathbb{E}_x[q_i(x)] = \bar q_i
\quad\Longrightarrow\quad
H(Z) \;=\; h(\bar q) \;=\; h_{\text{mean}}.$$

상호정보량의 정의 $I(X;Z) = H(Z) - H(Z\mid X)$를 그대로 대입하면

$$\boxed{\;h_{\text{mean}} - h_{\text{each}} \;=\; H(Z) - H(Z\mid X) \;=\; I(X;Z)\;\ge\;0.\;}$$

Jensen 부등식이 사실은 "상호정보량은 음수가 될 수 없다"의 다른 얼굴이었다.

### 결론 — 우리가 최대화하고 싶은 것은 바로 이 간격이다

$I(X;Z)$는 **"코드 $Z$를 보면 입력 $X$에 대해 몇 nat을 알 수 있는가"**이다. 자기지도학습이 원하는 것 자체다.

- $I = 0$ $\Leftrightarrow$ 코드가 입력에 대해 아무것도 말해주지 않는다 $\Leftrightarrow$ **붕괴**.
- $I$가 크다 $\Leftrightarrow$ 개별 분포는 뾰족한데($h_{\text{each}}$ 낮음) 점마다 다른 차원을 고른다($h_{\text{mean}}$ 높음) $\Leftrightarrow$ **건강**.

그리고 여기서 4절의 수수께끼가 풀린다. **loss는 $I(X;Z)$를 전혀 보지 않는다.** 노트북 4절이 "원-핫 붕괴 loss = 0 = 건강한 원-핫 loss"를 보여준 것은, loss가 두 crop의 일치만 재고 $X \to Z$의 정보 흐름은 재지 않기 때문이다. 그래서 `h_each`/`h_mean`을 **따로 로깅**해야 한다.

### 원-핫 붕괴와 균등 붕괴는 "간격 0"의 두 가지 방식일 뿐

$I = 0$이 되는 방법이 여러 가지라는 것이 두 붕괴가 반대 방향으로 보이는 이유다.

$$\underbrace{h_{\text{each}} = h_{\text{mean}} = 0}_{\text{원-핫 붕괴}}
\qquad\text{vs}\qquad
\underbrace{h_{\text{each}} = h_{\text{mean}} = \log K}_{\text{균등 붕괴}}$$

둘 다 $q(x)$가 $x$에 무관한 상수라는 점에서 **완전히 같은 병**이다. 다만 그 상수가 $e_j$냐 $\mathbf{1}/K$냐가 다를 뿐. 방지 장치가 두 개 필요한 이유는 붕괴가 두 종류여서가 아니라, **간격을 0으로 만드는 경로가 양쪽에 있어서** 둘 다 막아야 하기 때문이다.

---

## 4. 원-핫 붕괴에서 두 값이 0인 이유 — 단계별로

전제: 어떤 고정된 차원 $j$에 대해 **모든** 입력이 $q(x) = e_j$ (예: 노트북 4절의 `const[:, 2] = 50.0` → 온도 0.04 softmax → 사실상 $e_2$).

**1단계 — 개별 엔트로피가 0.**
$$h(e_j) = -\sum_i (e_j)_i \log (e_j)_i = -1\cdot\log 1 - \sum_{i \ne j} 0 \cdot \log 0 = 0.$$
($0\log 0 = 0$ 규약. 코드에서는 `p.clamp_min(1e-12).log()`가 이를 대신한다.)

**2단계 — 따라서 $h_{\text{each}} = 0$.**
$$h_{\text{each}} = \mathbb{E}_x[h(e_j)] = \mathbb{E}_x[0] = 0.$$

**3단계 — 평균 분포도 여전히 같은 원-핫.**
$$\bar q = \frac{1}{N}\sum_{n=1}^{N} q(x_n) = \frac{1}{N}\sum_{n=1}^{N} e_j = e_j.$$
같은 것을 아무리 많이 평균 내도 그대로다. **섞을 다양성이 없다.**

**4단계 — 그러므로 $h_{\text{mean}} = 0$.**
$$h_{\text{mean}} = h(\bar q) = h(e_j) = 0.$$

**5단계 — 간격도 0.**
$$I(X;Z) = h_{\text{mean}} - h_{\text{each}} = 0 - 0 = 0.$$

$h_{\text{mean}} \ge h_{\text{each}} \ge 0$이고 $h_{\text{mean}} = 0$이므로 사실 **3단계만 확인하면 나머지는 부등식에 끼여서 자동**이다. $h_{\text{mean}} = 0$은 두 값 중 더 강한 조건이다.

> **대조 — 건강한 원-핫.** 노트북 4절의 `healthy` 케이스처럼 입력마다 *다른* 원-핫을 내면 1·2단계는 똑같이 $h_{\text{each}} = 0$인데, 3단계가 무너진다: $\bar q$가 여러 차원에 퍼지므로 $h_{\text{mean}} > 0$. **$h_{\text{each}} = 0$만으로는 붕괴를 판정할 수 없다** — 반드시 $h_{\text{mean}}$을 같이 봐야 한다.

---

## 5. 세 상태 정리표

$K = 32$ (노트북 5절 `out_dim=32`), 데이터는 6개 클러스터. $\log K = 3.47$, $\log 6 = 1.79$.

| 상태 | $q(x)$ | $h_{\text{each}}$ | $h_{\text{mean}}$ | 간격 $I(X;Z)$ | loss |
|---|---|---|---|---|---|
| **원-핫 붕괴** | 모든 $x$ → 같은 $e_j$ | $\to 0$ | $\to 0$ | **0** | $\downarrow 0$ (완벽해 보임!) |
| **균등 붕괴** | 모든 $x$ → $\mathbf{1}/K$ | $\to \log K = 3.47$ | $\to \log K = 3.47$ | **0** | $\to \log K$ |
| **건강** | $x$마다 다른, 뾰족한 분포 | 낮음 ($\approx 0.6$) | 높음 ($\approx 2.4$) | **크다 ($\approx 1.8$)** | 중간 ($\approx 0.8$) |

**한눈에 보이는 것**: 마지막 열(loss)은 세 상태를 구분하지 못한다 — 원-핫 붕괴의 loss가 오히려 가장 낮다. 구분해 주는 것은 **간격 열 하나뿐**이고, 건강한 상태만 그 값이 크다.

### 노트북 실측치가 이론값과 맞는다

5절 `both = DINO (center on, temp 0.04)` 설정의 최종 수치:

$$h_{\text{each}} \approx 0.6, \qquad h_{\text{mean}} \approx 2.4
\qquad\Longrightarrow\qquad
I(X;Z) \approx 2.4 - 0.6 = 1.8.$$

한편 6개 클러스터를 **완벽히 1:1로 구분**할 때의 이론값은, $X$의 클러스터 라벨이 균등하고 코드가 그 라벨을 결정적으로 결정하므로

$$I(X;Z) = H(Z) = \log 6 = 1.7918.$$

$1.8 \approx \log 6$ — **정확히 맞는다.** 이것이 아름다운 이유는, 노트북이 따로 보고한 `top_share = 1/6`과 `code_acc = 1.0`이라는 두 사실을 **정보량 한 개의 숫자가 동시에 확인**해 주기 때문이다. 6개 코드가 균등하게 쓰이고($H(Z) = \log 6$) 코드가 클러스터를 결정한다($H(Z\mid \text{cluster}) \approx 0$)는 것이, 간격이 $\log 6$이라는 한 줄에 다 들어 있다.

### 수치 확인 (python3)

```python
import math, numpy as np
def ent(q): return -(q*np.log(np.clip(q,1e-30,None))).sum(-1)
K, N = 32, 768
y = np.arange(N) % 6                       # 6개 클러스터

def report(name, q):
    he, hm = ent(q).mean(), ent(q.mean(0))
    print(f"{name:24s} h_each={he:6.3f}  h_mean={hm:6.3f}  gap={hm-he:6.3f}")

q = np.zeros((N,K)); q[:,2] = 1.0                      ; report("one-hot collapse", q)
q = np.full((N,K), 1/K)                                ; report("uniform collapse", q)
q = np.zeros((N,K)); q[np.arange(N), y] = 1.0          ; report("healthy, hard 6 codes", q)
a = 0.1                                                 # 살짝 부드럽게
q = np.full((N,K), a/K); q[np.arange(N), y] += 1-a     ; report("healthy, soft (a=0.1)", q)
print(f"log K = {math.log(K):.4f},  log 6 = {math.log(6):.4f}")
```

실행 결과:

```
one-hot collapse         h_each= 0.000  h_mean=-0.000  gap=-0.000
uniform collapse         h_each= 3.466  h_mean= 3.466  gap= 0.000
healthy, hard 6 codes    h_each= 0.000  h_mean= 1.792  gap= 1.792   ← = log 6
healthy, soft (a=0.1)    h_each= 0.651  h_mean= 2.193  gap= 1.542
log K = 3.4657,  log 6 = 1.7918
```

- 두 붕괴 모두 `gap = 0` — **엔트로피 값은 정반대인데 간격은 똑같이 0**이다. (원-핫 줄의 `-0.000`은 `clip(...,1e-30)` 때문에 생긴 $-1\cdot\log(1) \to$ 부호 있는 0일 뿐, 값은 0이다.)
- `hard 6 codes`의 간격이 정확히 $\log 6 = 1.7918$. 위에서 유도한 이론값 그대로다.
- $a = 0.1$로 부드럽게 하면 $h_{\text{each}} = 0.65$, $h_{\text{mean}} = 2.19$ — 노트북 실측 $(0.6,\ 2.4)$와 같은 자리에 온다.

**Jensen 부등식 자체도 확인:**

```python
rng = np.random.default_rng(0); bad, gaps = 0, []
for _ in range(20000):
    z = rng.gumbel(size=(8,5)) * rng.uniform(0.1, 3.0)
    q = np.exp(z); q /= q.sum(-1, keepdims=True)
    g = ent(q.mean(0)) - ent(q).mean(); gaps.append(g)
    if g < -1e-12: bad += 1
print(bad, min(gaps), max(gaps))
# → 0  0.00294  1.424
```

랜덤 배치 20000개 중 **위반 0건**, 최소 간격도 양수다. $h_{\text{mean}} \ge h_{\text{each}}$는 우연이 아니라 항등적으로 성립한다.

---

## 6. 어느 장치가 어느 값을 미는가

이제 DINO의 두 장치를 간격의 언어로 다시 읽을 수 있다.

$$I(X;Z) \;=\; \underbrace{h_{\text{mean}}}_{\text{centering이 } \uparrow} \;-\; \underbrace{h_{\text{each}}}_{\text{sharpening이 } \downarrow}$$

| 장치 | 무엇을 하나 | 직접 미는 값 | 방향 | 막는 붕괴 |
|---|---|---|---|---|
| **sharpening** (teacher 온도 $\tau_t = 0.04$) | 로짓을 낮은 온도로 나눠 분포를 뾰족하게 | $h_{\text{each}}$ | $\downarrow$ | 균등 붕괴 |
| **centering** (출력 평균 EMA를 빼기) | 특정 차원이 배치를 독점하지 못하게 | $h_{\text{mean}}$ | $\uparrow$ | 원-핫 붕괴 |

**두 장치는 각각 간격 공식의 한 항씩을, 정확히 간격을 벌리는 부호로 민다.** 빼는 항을 내리고 더하는 항을 올리니, 둘을 함께 켜면 $I(X;Z)$가 양쪽에서 밀려 커진다. loss가 $I$를 전혀 보지 않는데도 DINO가 붕괴하지 않는 이유가 여기 있다 — **손실 함수가 아니라 이 두 장치가 $I$를 담당한다.**

노트북 5절의 네 설정이 이 표를 그대로 실증한다:

- **none** (둘 다 없음): $h_{\text{each}} \approx h_{\text{mean}} \approx 2.9$ — **두 값이 붙었다.** 간격 $\approx 0$, `code_acc`는 chance 근처. 등호 조건 = 붕괴가 그대로 확인된다.
- **center only**: $h_{\text{mean}}$은 지켰지만 $h_{\text{each}} \to \log K$ — 빼는 항이 따라 올라가 간격이 안 벌어진다. 균등 쪽 붕괴.
- **sharp only**: $h_{\text{each}} = 0$으로 완벽히 내렸는데 $h_{\text{mean}} \approx 1.2$뿐 — 더하는 항이 낮아 간격 $\approx 1.2 < \log 6$. 6개 클러스터가 ~4개 코드로 뭉치고 한 코드가 절반을 먹는다. **한 항만 밀어서는 안 된다**는 증거.
- **both = DINO**: $0.6$ / $2.4$, 간격 $\approx 1.8 = \log 6$ — 6개 코드가 6개 클러스터에 1:1.

---

## 7. 외울 것

1. $h_{\text{each}} = \mathbb{E}_x[h(q(x))]$, $h_{\text{mean}} = h(\mathbb{E}_x[q(x)])$ — `entropy(q).mean()` vs `entropy(q.mean(0))`, **축 순서만 다르다**.
2. **항상** $h_{\text{mean}} \ge h_{\text{each}}$ (엔트로피의 오목성 + Jensen). 등호 $\Leftrightarrow$ $q(x)$가 $x$에 무관 $\Leftrightarrow$ **붕괴**.
3. 간격 $h_{\text{mean}} - h_{\text{each}} = I(X;Z)$. **우리가 키우고 싶은 것은 이 간격**이고, 붕괴란 간격이 0이 되는 일이다.
4. **원-핫 붕괴: 둘 다 0** (모두 같은 $e_j$ → 개별 0, 평균도 $e_j$ → 평균 분포도 0, 간격 0). 균등 붕괴: 둘 다 $\log K$, 역시 간격 0.
5. 노트북 DINO 실측 $2.4 - 0.6 = 1.8 \approx \log 6 = 1.79$ — 6개 클러스터를 완벽히 구분할 때의 이론값과 일치.
6. sharpening은 $h_{\text{each}}\downarrow$, centering은 $h_{\text{mean}}\uparrow$ — **두 장치가 정확히 간격을 벌린다.**
