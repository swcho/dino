# `trunc_normal_` 의 $a, b$ 는 몇 $\sigma$ 가 아니라 "절대 좌표"다

## 0. 결론 먼저

```python
trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.)
```

여기서 $a=-2$, $b=2$ 는 **"평균에서 $2\sigma$ 떨어진 곳"이 아니다.**
**수직선 위의 절대 위치 $-2$ 와 $+2$** 다. 그래서 `std` 를 바꾸면
같은 $a,b$ 라도 "몇 $\sigma$ 지점인가"가 완전히 달라진다.

이 카드는 고교 '확률과 통계'의 **정규분포 표준화** 하나로 완결된다.
아래에서 정규분포 → 표준화 → 누적분포함수 → 절단정규분포 → 실제 코드 순서로 쌓아 올린다.

---

## 1. 출발점: 정규분포와 표준화 (고교 확률과 통계)

확률변수 $X$ 가 평균 $\mu$, 표준편차 $\sigma$ 인 정규분포를 따를 때

$$X \sim N(\mu,\ \sigma^2)$$

라고 쓴다. 고교에서 정규분포표를 쓰려면 반드시 **표준화**를 먼저 했다.

$$Z = \frac{X-\mu}{\sigma} \sim N(0,1)$$

이 식이 이 카드의 **전부**다. $Z$ 는 "평균에서 몇 $\sigma$ 떨어져 있는가"를 재는 눈금이고,
$X$ 는 "실제 값이 얼마인가"를 재는 눈금이다. 둘은 **다른 자다.**

예를 들어 $X$ 가 키(cm)이고 $\mu=170,\ \sigma=6$ 이라면

- $X = 182$ (절대 좌표: 182cm)
- $Z = \dfrac{182-170}{6} = 2$ (상대 좌표: $2\sigma$)

"182"라고만 말하면 $\sigma$ 를 알아야 몇 $\sigma$ 인지 알 수 있다.
$\sigma$ 가 6이 아니라 0.12였다면 같은 $X=182$ 는 $Z=100$ 이 된다.
**$a=-2, b=2$ 가 겪는 일이 정확히 이것이다.**

---

## 2. 누적분포함수 $\Phi(z)$

표준정규분포의 확률밀도함수를

$$\phi(z) = \frac{1}{\sqrt{2\pi}}\,e^{-z^2/2}$$

라 하고, 이를 $-\infty$ 부터 $z$ 까지 적분한 것을 **누적분포함수(CDF)** 라 한다.

$$\Phi(z) = P(Z \le z) = \int_{-\infty}^{z} \phi(t)\,dt$$

고교 정규분포표가 주는 값 $P(0 \le Z \le z)$ 와는 $\Phi(z) = 0.5 + P(0\le Z\le z)$ 관계다.
구체값 몇 개를 기억해 두면 이 카드가 한눈에 보인다.

| $z$ | $\Phi(z)$ | 의미 |
|---|---|---|
| $-4$ | $3.17\times10^{-5}$ | 3만 개 중 1개 |
| $-3$ | $0.00135$ | 0.135% |
| $-2$ | $0.02275$ | 약 2.28% |
| $-1$ | $0.15866$ | 약 15.9% |
| $0$ | $0.5$ | 절반 |
| $-20$ | $\approx 2.8\times10^{-89}$ | 사실상 0 |
| $-100$ | $\approx 0$ (부동소수점에서 정확히 0) | 완전히 0 |

$\Phi(-2)\approx 0.0228$ 은 고교에서 외운 "$\pm2\sigma$ 안에 약 95%" 의 근거다.
양쪽 꼬리를 합치면 $2\times0.0228 = 0.0455$, 즉 약 4.55%가 밖에 있다.

---

## 3. 절단정규분포(truncated normal)란 무엇인가

**정의:** 정규분포에서 구간 $[a,b]$ **밖의 값은 버리고**, 남은 것만 다시 확률 총합 1로
**재정규화**한 분포.

유도는 고교 조건부확률 그대로다. $X \sim N(\mu,\sigma^2)$ 일 때 $a\le X\le b$ 라는 조건 아래

$$f(x \mid a \le X \le b) = \frac{f(x)}{P(a \le X \le b)} \quad (a \le x \le b)$$

분모를 표준화해서 계산하자. $\alpha = \dfrac{a-\mu}{\sigma}$, $\beta = \dfrac{b-\mu}{\sigma}$ 로 두면

$$P(a \le X \le b) = P(\alpha \le Z \le \beta) = \Phi(\beta) - \Phi(\alpha)$$

이고, $z=\frac{x-\mu}{\sigma}$ 로 치환하면 밀도는

$$f(x) = \frac{1}{\sigma}\cdot\frac{\phi(z)}{\Phi(\beta)-\Phi(\alpha)}, \qquad z = \frac{x-\mu}{\sigma}$$

핵심은 분모의 $\Phi(\beta)-\Phi(\alpha)$ 다. 이것이 **"잘리지 않고 살아남는 확률"** 이고,
$1 - (\Phi(\beta)-\Phi(\alpha))$ 가 **"잘려 나가는 확률"** 이다.

그리고 이 분모에 들어가는 것은 $a, b$ 자체가 아니라
**표준화된 $\alpha=\frac{a-\mu}{\sigma}$, $\beta=\frac{b-\mu}{\sigma}$** 다.
즉 $a,b$ 가 고정돼 있어도 **$\sigma$ 가 바뀌면 절단 위치가 바뀐다.**

---

## 4. 핵심: 같은 $a=-2$ 인데 $\sigma$ 에 따라 이렇게 달라진다

$\mu = 0$, $a=-2$, $b=+2$ 로 고정하고 `std` 만 바꿔 보자.

$$\alpha = \frac{a-\mu}{\sigma} = \frac{-2}{\sigma}, \qquad \beta = \frac{2}{\sigma}$$

| `std` $=\sigma$ | $\alpha = -2/\sigma$ | 경계가 몇 $\sigma$ 인가 | 잘리는 확률 $2\Phi(\alpha)$ | 실질 |
|---|---|---|---|---|
| **0.02** (DINO) | $-100$ | $\pm100\sigma$ | $\approx 0$ (정확히 0으로 계산됨) | **절단 없음** |
| 0.1 | $-20$ | $\pm20\sigma$ | $\approx 5.5\times10^{-89}$ | 절단 없음 |
| 0.5 | $-4$ | $\pm4\sigma$ | $6.3\times10^{-5}$ (0.006%) | 거의 없음 |
| **1.0** (기본값) | $-2$ | $\pm2\sigma$ | $0.0455$ (약 **4.6%**) | **실제로 잘림** |
| 2.0 | $-1$ | $\pm1\sigma$ | $0.3173$ (약 **31.7%**) | 심하게 잘림 |

읽는 법: $\sigma=0.02$ 일 때 "값이 $-2$ 보다 작을 확률"은 $P(Z < -100)$ 이다.
표준편차가 0.02인 분포에서 $-2$ 는 100 표준편차 밖이므로, 그런 값은 절대 나오지 않는다.
**즉 자를 것이 애초에 없다.**

반대로 $\sigma=1$ 이면 $-2$ 는 딱 $2\sigma$ 이므로, 고교에서 배운 대로
전체의 약 4.55%가 잘려 나가고 분포 모양이 실제로 바뀐다.

### 실측으로 확인

```
std=0.02 : 실측 std=0.0200   max|w|=0.1059  = 5.29 sigma   → 경계(±2)에 근접조차 못 함
std=1.0  : 실측 std=0.8777   max|w|=1.9999  = 2.00 sigma   → 정확히 경계에서 잘림
```

$\sigma=1$ 에서 실측 표준편차가 0.8777로 **1보다 작아진** 것을 보라.
꼬리를 잘라 냈으니 퍼짐이 줄어드는 것이 당연하다. 절단정규분포의 분산은

$$\mathrm{Var} = \sigma^2\left[1 + \frac{\alpha\phi(\alpha)-\beta\phi(\beta)}{\Phi(\beta)-\Phi(\alpha)} - \left(\frac{\phi(\alpha)-\phi(\beta)}{\Phi(\beta)-\Phi(\alpha)}\right)^2\right]$$

로 원래 $\sigma^2$ 보다 작다. 반면 $\sigma=0.02$ 에서는 실측이 정확히 0.0200 —
**절단이 전혀 개입하지 않았다는 증거**다.

---

## 5. 구현 근거: 실제 PyTorch 코드

DINO가 쓰는 구현은 `/home/sungwoo/projects/swcho/dino/utils.py` 의
`_no_grad_trunc_normal_` 이다(PyTorch 공식 구현을 그대로 복사한 것).

```python
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    ...
        l = norm_cdf((a - mean) / std)     # ← α = (a-μ)/σ  를 표준화해서 넣는다
        u = norm_cdf((b - mean) / std)     # ← β = (b-μ)/σ
```

`norm_cdf` 가 곧 $\Phi$ 다. 실제로

$$\Phi(x) = \frac{1}{2}\left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)$$

이고 이는 오차함수 $\operatorname{erf}(t)=\frac{2}{\sqrt\pi}\int_0^t e^{-s^2}ds$ 의 정의에서
$e^{-z^2/2}$ 적분을 치환하면 바로 나온다.

그리고 `(a - mean) / std` 라는 나눗셈 한 줄이 **이 카드의 모든 근거**다.
$a$ 는 그대로 쓰이지 않고, 반드시 `std` 로 나뉘어 들어간다.
따라서 `std=0.02` 면 `l = norm_cdf(-100) = 0.0`, `u = norm_cdf(100) = 1.0` 이 되어
$[l,u] = [0,1]$ — **절단 구간이 전체가 되므로 아무것도 잘리지 않는다.**

---

## 6. 나머지 코드: 역변환 샘플링(inverse transform sampling)

$\Phi$ 를 왜 계산했는지가 이어지는 코드에서 드러난다.

```python
        tensor.uniform_(2 * l - 1, 2 * u - 1)   # ① 균등분포 뽑기 (erfinv 입력 스케일로 이동)
        tensor.erfinv_()                        # ② Φ⁻¹ 적용
        tensor.mul_(std * math.sqrt(2.))        # ③ σ 스케일
        tensor.add_(mean)                       # ④ μ 이동
        tensor.clamp_(min=a, max=b)             # ⑤ 부동소수점 오차 보정용 안전망
```

### 원리를 고교 수준으로

$\Phi$ 는 **단조증가**하는 함수이고 치역이 $(0,1)$ 이다(미분값이 $\phi(z)>0$ 이므로).
단조증가 함수는 역함수 $\Phi^{-1}:(0,1)\to\mathbb{R}$ 를 가진다.

여기서 다음 사실이 성립한다.

> $U$ 가 $[0,1]$ 위 균등분포를 따르면, $Z = \Phi^{-1}(U)$ 는 표준정규분포를 따른다.

**증명(고교 수준):** $Z=\Phi^{-1}(U)$ 의 누적분포함수를 직접 구하면

$$P(Z \le z) = P(\Phi^{-1}(U) \le z) = P(U \le \Phi(z)) = \Phi(z)$$

마지막 등호는 $U$ 가 균등분포이므로 $P(U\le u)=u$ 라는 사실이다.
누적분포함수가 $\Phi$ 이니 $Z$ 는 표준정규분포다. $\blacksquare$

여기서 $U$ 를 $[0,1]$ 전체가 아니라 **$[l,u]$ 구간에서만** 뽑으면?
같은 계산으로 $P(Z\le z)$ 가 $\dfrac{\Phi(z)-l}{u-l}$ 이 되고, 이는 정확히
$[a,b]$ 로 절단된 정규분포의 CDF다. **§3에서 유도한 분모 $\Phi(\beta)-\Phi(\alpha)$ 가
바로 이 `u - l`** 이다.

### `erfinv` 는 $\Phi^{-1}$ 의 다른 표기

$\Phi(z)=\frac12(1+\operatorname{erf}(z/\sqrt2))$ 를 $z$ 에 대해 풀면

$$z = \sqrt{2}\,\operatorname{erf}^{-1}(2\Phi - 1)$$

그래서 코드는 $\Phi$ 값 $p\in[l,u]$ 를 $2p-1$ 로 옮겨(`2*l-1, 2*u-1`),
`erfinv_()` 를 적용한 뒤 $\sqrt2$ 를 곱한다(`mul_(std * sqrt(2))`).
즉 ①②③은 합쳐서 "$\mu + \sigma\,\Phi^{-1}(p)$" 한 줄과 같다.
`erfinv` 는 새로운 함수가 아니라 **$\Phi^{-1}$ 을 표준 라이브러리 함수로 쓰기 위한 표기 변환**일 뿐이다.

⑤ `clamp_` 는 부동소수점 오차로 $a$ 나 $b$ 를 미세하게 넘는 경우만 막는 안전망이다.
`std=1.0` 실측에서 `max|w| = 1.9999` 가 나온 것이 이 경계에 붙은 흔적이다.

---

## 7. 그래서 DINO에서는?

`vision_transformer.py` 의 초기화는 이렇다.

```python
trunc_normal_(self.pos_embed, std=.02)
trunc_normal_(self.cls_token, std=.02)

def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
```

`a`, `b` 는 넘기지 않았으므로 기본값 $-2, +2$ 가 쓰인다. §4의 표 첫 줄이 그것이다.

$$\alpha = \frac{-2 - 0}{0.02} = -100 \quad\Longrightarrow\quad \Phi(-100) = 0$$

**결론: DINO의 `trunc_normal_(std=.02)` 는 절단이 전혀 일어나지 않는,
사실상 그냥 $N(0,\,0.02^2)$ 정규분포 초기화다.**
`nn.init.normal_(m.weight, std=.02)` 와 통계적으로 구별되지 않는다.
timm에서 물려받은 관례라서 그렇고, 실질적 문제는 없다.

**절단이 의미를 갖는 것은 `std` 가 경계 $|a|,|b|$ 와 비슷한 규모일 때뿐이다.**
$|a|/\sigma$ 가 대략 2~3 정도로 떨어져야 꼬리가 실제로 잘린다.

### 흔히 하는 오해

> "절단정규분포로 초기화했으니 모든 가중치가 $\pm2\sigma$ 안에 있겠지"

**틀렸다.** 실측에서 `std=0.02` 일 때 `max|w| = 0.1059 = 5.29σ` 였다.
$5\sigma$ 를 넘는 값이 태연히 나온다. 20만 개를 뽑았으니
$P(|Z|>5)\approx 5.7\times10^{-7}$ 에 $2\times10^5$ 를 곱하면 기댓값 약 0.1개,
$|Z|>4.5$ 는 약 1.4개 — **잘리지 않은 순수 정규분포의 꼬리 그대로**다.

$\pm2\sigma$ 보장을 원한다면 $a, b$ 를 $\sigma$ 에 맞춰 직접 넘겨야 한다.

```python
trunc_normal_(m.weight, std=.02, a=-.04, b=.04)   # 이제야 ±2σ 절단
```

---

## 8. 한 줄 요약

$a, b$ 는 **절대값 경계**다. 코드가 `norm_cdf((a - mean) / std)` 로
반드시 `std` 로 나눠 표준화하므로, 같은 $a=-2$ 라도
$\sigma=1$ 이면 $2\sigma$(4.6% 절단), $\sigma=0.02$ 면 $100\sigma$(절단 0%)가 된다.
