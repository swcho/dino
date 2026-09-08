# `entropy(p)`의 `clamp_min(1e-12)` — 왜 필요한가

## 카드가 가리키는 코드

노트북 1절에 나오는 세 줄짜리 헬퍼다.

```python
def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)

def cross_entropy(q, p):
    return -(q * p.clamp_min(1e-12).log()).sum(-1)

def kl(q, p):
    return (q * (q.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1)
```

세 함수 모두 `.log()` 앞에 `clamp_min(1e-12)`가 붙어 있다. 이유는 하나다:
**`log(0) = -inf`, 그리고 `0 * -inf = nan`**.

---

## 1. 수학에서는 $0\log 0 = 0$인데 왜 코드는 `nan`이 되나

엔트로피 정의에는 관례가 하나 붙어 있다.

$$h(p) = -\sum_i p_i \log p_i, \qquad 0 \log 0 := 0$$

이건 임의의 약속이 아니라 극한값이다. $x \to 0^+$일 때

$$\lim_{x\to 0^+} x \log x = \lim_{x\to 0^+} \frac{\log x}{1/x}
\;\overset{\text{L'Hôpital}}{=}\; \lim_{x\to 0^+} \frac{1/x}{-1/x^2} = \lim_{x\to 0^+} (-x) = 0$$

즉 $x$가 0으로 가는 속도가 $\log x$가 $-\infty$로 가는 속도보다 훨씬 빠르다. 그래서 원-핫 분포의 엔트로피는 잘 정의된 $0$이다.

**그런데 부동소수점은 극한을 모른다.** IEEE 754는 두 피연산자를 각각 계산한 뒤 곱할 뿐이다.

| 단계 | 값 |
|---|---|
| `p_i` | `0.0` |
| `p_i.log()` | `-inf` |
| `p_i * p_i.log()` | `0.0 * -inf` = **`nan`** |

`0 * inf`는 IEEE 754가 명시적으로 "invalid operation"으로 규정해 `nan`을 내도록 정한 케이스다. 어떤 값을 줘야 할지 정보가 없기 때문이다($0 \cdot \infty$는 부정형이라 문맥에 따라 답이 다르다).

그리고 `nan`은 전염된다. `sum()` 한 번이면 벡터 전체 결과가 `nan`이 되고, 이후 로깅·플롯·비교가 전부 오염된다.

노트북에서 바로 이 상황이 벌어진다. 1절 첫 실행에서 `one_hot`을 넣는데:

```python
one_hot = F.one_hot(torch.tensor(3), K).float()   # [0,0,0,1,0,0,0,0]
```

clamp 없이 계산하면:

```
naive one_hot: nan
clamped      : -0.0     ← 원하는 값
```

즉 카드가 다루는 첫 예제부터 clamp 없이는 못 돌아간다.

---

## 2. `clamp_min`이 정확히 하는 일

`x.clamp_min(t)`는 원소별 $\max(x, t)$다. `torch.clamp(x, min=t)`와 같다.

```python
p                     # [0.0, 1.0]
p.clamp_min(1e-12)    # [1e-12, 1.0]
p.clamp_min(1e-12).log()   # [-27.631, 0.0]     ← -inf 대신 유한한 큰 음수
```

여기서 **가장 중요한 디테일**: clamp는 `.log()`의 인자에만 걸리고, 바깥에 곱해지는 `p`는 원래 값 그대로다.

```python
-(p * p.clamp_min(1e-12).log()).sum(-1)
#   ↑ 원본 p          ↑ clamp된 p
```

그래서 $p_i = 0$인 항의 기여는

$$-\,0 \times \log(10^{-12}) = -\,0 \times (-27.631) = 0$$

**정확히 0**이 된다. 수학적 관례 $0\log 0 = 0$을 부동소수점 산술 위에서 그대로 재현한 것이다. 근사가 아니라 정답이다.

만약 실수로 `-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)`처럼 양쪽 다 clamp했다면 항마다 $10^{-12}\times 27.6 \approx 2.8\times10^{-11}$의 가짜 엔트로피가 더해진다. 지금 형태가 옳다.

### 부호 주의

위 계산의 결과는 `-0.0`(음의 영)이다. IEEE 754에서 `-0.0 == 0.0`은 `True`이고 출력 포맷에서 `-0.000`으로 보일 뿐이라 실질 문제는 없다. 노트북 1절 출력에 `h=-0.000`이 찍히면 그게 이 이유다.

---

## 3. 왜 하필 $10^{-12}$인가

세 가지 조건을 동시에 만족해야 한다.

**(a) `log`가 유한해야 한다** — 어떤 양수든 만족.

**(b) 오차가 무시 가능해야 한다.** $0 < p_i < 10^{-12}$인 항에서만 오차가 생기는데, 그 크기는

$$\bigl|\,p_i\log p_i - p_i \log 10^{-12}\,\bigr| \;\le\; p_i \cdot |\log p_i| \;<\; 10^{-12} \times 27.63 \approx 2.8\times 10^{-11}$$

DINO의 $K = 65536$ 차원이 전부 최악의 경우여도 총 오차는 $65536 \times 2.8\times10^{-11} \approx 1.8\times10^{-6}$이다. 실제로는 확률의 합이 1이라 그런 상황 자체가 불가능하므로 훨씬 작다. 반면 엔트로피의 스케일은 $\log 65536 = 11.09$ — **오차가 유효숫자 밖**이다.

실제로 재보면:

```
p = [1 - 1e-13, 1e-13]  (float64)
  true    h = 3.0934e-12
  clamped h = 2.8631e-12
  diff      = 2.3e-13     ← 절대값 자체가 이미 무의미한 영역
```

**(c) float32에서 표현 가능하고, 그 로그가 극단적이지 않아야 한다.**
float32의 최소 정규수는 $1.18\times10^{-38}$, 최소 비정규수는 $1.4\times10^{-45}$. $10^{-12}$는 여유롭게 정규수 범위 안이고 `log` 결과 $-27.63$도 평범한 수다.

$10^{-45}$처럼 극단적으로 작게 잡으면 비정규수 영역이라 정밀도가 무너지고 `log`도 $-103$까지 커진다. 반대로 $10^{-7}$처럼 크게 잡으면 실제로 의미 있는 작은 확률까지 잘려 나간다.

$10^{-12}$는 **"float32의 유효 정밀도($\approx 10^{-7}$ 상대오차)보다는 한참 아래이면서, 비정규수 영역보다는 한참 위"** 인 넉넉한 중간 지점이다. 관례적으로 널리 쓰이는 값이고, $10^{-9}$나 $10^{-10}$을 써도 결론은 같다. **정확한 값이 중요한 게 아니라 "0이 아닌 작은 양수"라는 점이 중요하다.**

---

## 4. DINO에서 왜 이게 실제로 문제가 되나 — softmax underflow

"확률이 정확히 0이 될 일이 있나? softmax 출력은 항상 양수 아닌가?" — **수학적으로는 그렇지만 float32에서는 아니다.**

softmax는 최댓값을 빼고 계산한다(수치 안정화):

$$p_i = \frac{\exp\bigl((z_i - z_{\max})/\tau\bigr)}{\sum_j \exp\bigl((z_j - z_{\max})/\tau\bigr)}$$

float32에서 `exp(x)`는 $x \lesssim -87.3$부터 비정규수로 내려가고 $x \lesssim -103.9$에서 **정확히 0**이 된다.

DINO의 teacher 온도는 `teacher_temp=0.04`다. 로짓 차이 $\Delta z$가 지수부에서 $\Delta z / 0.04 = 25\Delta z$로 **25배 증폭**된다.

| 로짓 차이 $\Delta z$ | $-\Delta z/0.04$ | float32 결과 |
|---|---|---|
| 3.5 | -87.5 | $\approx 10^{-38}$ (비정규수 진입) |
| 4.0 | -100 | $3.8\times10^{-44}$ (비정규수, 정밀도 손실) |
| 5.0 | -125 | **정확히 `0.0`** |

실제 측정:

```python
z = torch.tensor([0., -3.5, -4., -5., -6.])
torch.softmax(z / 0.04, -1)
# tensor([1.0000e+00, 9.9824e-39, 3.7835e-44, 0.0000e+00, 0.0000e+00])
#                                             └──────── 진짜 0 ────────┘
```

로짓 차이 **5**면 끝이다. 65536차원 출력에서 로짓 스프레드가 5 이상인 건 학습 초반부터 흔하다. 게다가 노트북 4절은 아예 `const[:, 2] = 50.0`처럼 극단적 로짓을 만들어 자명해를 시연한다 — 여기서 나오는 teacher 분포는 한 자리를 뺀 나머지가 전부 정확히 0이다.

그리고 이 카드가 다루는 `entropy`는 **정확히 그 분포에 대해 호출된다**. 5절 학습 루프:

```python
hist["h_each"].append(entropy(q).mean().item())   # q = softmax(logits/0.04)
hist["h_mean"].append(entropy(q.mean(0)).item())
```

clamp가 없으면 `h_each`와 `h_mean`이 학습 도중 `nan`으로 바뀌고, 그래프는 그 지점부터 끊긴다. **하필 원-핫 붕괴가 진행될수록 (= 가장 관찰하고 싶은 순간) 확실히 터진다.** 붕괴 진단 지표가 붕괴 때문에 죽는 셈이라, clamp는 편의가 아니라 필수다.

> bf16/fp16을 쓰면 훨씬 심하다. fp16의 최소 정규수는 $6.1\times10^{-5}$라 `exp(-10)` 수준에서 이미 0으로 내려앉는다. AMP 환경에서는 loss/지표 계산을 float32로 승격시키는 게 표준이다.

---

## 5. 대안들과의 비교

### (a) `torch.xlogy(p, p)` — 같은 목적의 전용 연산

`torch.xlogy(x, y)`는 $x\log y$를 계산하되 **$x = 0$이면 $y$가 뭐든 0을 반환**하도록 특수 처리된 함수다.

```python
-torch.xlogy(p, p).sum(-1)     # p=[0,1] → -0.  ✅ 값은 정확
```

값 계산만 보면 clamp보다 깨끗하다(오차 $2.8\times10^{-11}$조차 없다). 하지만 **역전파에서 문제가 있다**:

```python
q = torch.tensor([0.0, 1.0], requires_grad=True)
(-torch.xlogy(q, q).sum()).backward()
# q.grad = tensor([nan, -1.])     ← nan
```

$\partial_x (x\log x) = \log x + 1$이라 $x=0$에서 발산하기 때문이다. 순전파는 살고 역전파가 죽는다.

### (b) `torch.special.entr(p)` — 엔트로피 항 전용

$-x\log x$를 통째로 계산하며 $x=0 \Rightarrow 0$, $x<0 \Rightarrow -\infty$로 정의된 함수다.

```python
torch.special.entr(p).sum(-1)   # p=[0,1] → 0.  ✅ 부호도 이미 반영됨
```

값만 필요한 진단 지표라면 **가장 정확하고 의도가 명확한 선택**이다. 다만
- 이것도 gradient는 `inf`가 나온다 (`tensor([inf, -1.])`).
- 엔트로피 전용이라 `cross_entropy(q, p)`나 `kl(q, p)`처럼 **서로 다른 두 분포**를 다루는 함수에는 쓸 수 없다. 노트북은 세 함수를 같은 패턴으로 맞춰 놓았기 때문에 clamp 쪽이 일관적이다.
- 이름이 덜 익숙해서 교육용 노트북에서는 "무슨 일이 일어나는지" 덜 드러난다.

### (c) `F.log_softmax` — 애초에 log-space에서 계산하기 (**실전 정답**)

**로짓이 손에 있다면 이게 최선이다.** softmax를 취한 뒤 로그를 씌우는 대신, 처음부터 로그 확률을 계산한다.

$$\log p_i = \frac{z_i}{\tau} - \log\sum_j \exp\Bigl(\frac{z_j}{\tau}\Bigr)$$

`log_softmax`는 내부적으로 log-sum-exp 트릭을 쓰므로 지수함수 언더플로가 **결과에 반영되지 않는다**. $p_i$가 $10^{-50}$이어도 $\log p_i \approx -115$라는 유한하고 정확한 값을 얻는다. 잘라낼 것도 없고 오차도 없다.

DINO 본체가 정확히 이렇게 한다 — `MiniDINOLoss.forward`(원본 `main_dino.py:380`):

```python
loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
#                      └── clamp_min 없음. 필요 없다 ──┘
```

**그래서 정리하면 역할 분담이 이렇다:**

| 상황 | 가진 것 | 방법 |
|---|---|---|
| **학습 loss** (`MiniDINOLoss`) | student **로짓** | `F.log_softmax` — 언더플로 자체가 없음 |
| **진단 지표** (`entropy`, `kl`) | 이미 계산된 **확률** $q$ | `clamp_min(1e-12).log()` — 0이 섞여 있을 수 있으므로 방어 |

`entropy`/`kl`이 확률 $q$를 받는 건 이유가 있다. `q`는 `teacher_dist()`가 `F.softmax(...)`로 만들어 반환한 값이고, `code_metrics`의 `argmax`, `q.mean(0)` 같은 다른 용도와 공유된다. 확률 형태가 필요하므로 log-space로 돌아갈 수 없고, 따라서 clamp가 필요하다.

### (d) `p + eps` (clamp 대신 더하기)

`(p + 1e-12).log()`도 흔히 보이지만 clamp보다 나쁘다:
- **모든** 원소에 편향을 준다 (clamp는 임계값 아래만 건드린다).
- 확률의 합이 $1 + K\varepsilon$이 되어 정규화가 미세하게 깨진다. $K=65536$이면 $6.6\times10^{-8}$ — 여기선 무해하지만 습관으로는 나쁘다.

---

## 6. 보너스: clamp가 gradient도 살린다

`clamp_min`은 임계값 아래 구간에서 **gradient를 0으로 흘린다**(상수 구간이므로). 그 결과:

```python
q = torch.tensor([1e-20, 1.0], requires_grad=True)
(-(q * q.clamp_min(1e-12).log()).sum()).backward()
# q.grad = tensor([27.6310, -1.0000])    ← 유한
```

$\partial_{p_i}\bigl[-p_i \log(\text{clamp})\bigr] = -\log(10^{-12}) = 27.63$ 로 딱 잘린다. 반면 naive 버전은 $-(\log p_i + 1)$이라 $p_i \to 0$에서 발산하고, `xlogy`/`entr`는 위에서 봤듯 `nan`/`inf`를 낸다.

노트북에서 `entropy`는 `@torch.no_grad()` 문맥의 지표로만 쓰이므로 이 성질을 활용하진 않지만, 엔트로피 정규화 항 같은 걸 loss에 넣게 되면 clamp 방식이 유일하게 안전한 선택이 된다.

---

## 한 줄 정리

수학의 $0\log 0 = 0$ 관례는 극한값이지만 IEEE 754는 `0.0 * -inf`를 `nan`으로 처리한다. `clamp_min(1e-12)`는 **`log`의 인자만** 아주 작은 양수로 잘라 `-inf`를 없애고, 바깥에 곱해지는 원본 $p$가 0이므로 결과는 **정확히 0** — 관례를 그대로 재현한다. DINO는 teacher 온도 0.04가 로짓 차이를 25배 증폭해 softmax 출력이 float32에서 **실제로 정확히 0이 되므로**, 붕괴 진단용 `entropy`/`kl`에는 이 방어가 필수다. 로짓을 쥐고 있는 학습 loss 쪽은 `F.log_softmax`로 log-space에서 계산해 문제 자체를 회피한다.
