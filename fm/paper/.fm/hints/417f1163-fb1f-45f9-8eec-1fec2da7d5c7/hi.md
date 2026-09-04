# DINO의 teacher 갱신 규칙과 $\lambda$ 스케줄

## 0. 한 줄 답

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$$

teacher의 가중치 $\theta_t$는 학습되지 않는다. student의 가중치 $\theta_s$를 **과거까지 포함해 평균낸 값**으로 매 스텝 조금씩 끌려간다. 그리고 $\lambda$는 고정값이 아니라 학습이 진행되면서 **0.996에서 1까지 cosine 곡선을 따라** 커진다.

이 글은 "왜 저 식이 평균인가", "왜 하필 0.996인가", "왜 1로 커지는가"를 고등학교 수학(수열, 등비급수, 삼각함수, 가중평균)만으로 쌓아 올린다.

---

## 1. 출발점: 고등학교의 가중평균

두 수 $a$, $b$를 섞을 때 우리는 이렇게 쓴다.

$$m = w\,a + (1-w)\,b, \qquad 0 \le w \le 1$$

계수의 합이 $w + (1-w) = 1$이므로 이건 **가중평균**이다. $w$가 1에 가까우면 $a$ 쪽에, 0에 가까우면 $b$ 쪽에 가깝다.

DINO의 갱신식을 다시 보자.

$$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$$

계수의 합이 $\lambda + (1-\lambda) = 1$. **똑같은 가중평균이다.** 다만 왼쪽의 $\theta_t$는 "갱신 후의 새 값", 오른쪽의 $\theta_t$는 "갱신 전의 옛 값"이라는 점이 다르다. 즉 이건 수열의 **점화식**이다.

스텝 번호 $n$을 붙여 정확히 쓰면

$$\theta_t^{(n)} = \lambda\,\theta_t^{(n-1)} + (1-\lambda)\,\theta_s^{(n)}$$

"새 teacher = (옛 teacher를 $\lambda$만큼) + (지금 student를 $1-\lambda$만큼)".

$\lambda = 0.996$이면 한 스텝에서 teacher는 **자기 자신을 99.6% 유지하고, 새 student 정보를 0.4%만 받아들인다.** 아주 굼뜨게 움직이는 셈이다.

> 비유: teacher는 student를 뒤따라가는 "무거운 수레"다. student가 이리저리 흔들려도 수레는 관성 때문에 부드러운 궤적만 그린다. 논문에서 이를 **momentum encoder**(관성 인코더)라 부르는 이유다.

---

## 2. 점화식을 펼치면: 지수적으로 감쇠하는 가중합

점화식은 반복 대입하면 일반항이 나온다. 고등학교에서 등비수열 점화식을 풀 때 하던 그대로 한다.

$$
\begin{aligned}
\theta_t^{(n)} &= \lambda\,\theta_t^{(n-1)} + (1-\lambda)\,\theta_s^{(n)}\\[2pt]
&= \lambda\Big[\lambda\,\theta_t^{(n-2)} + (1-\lambda)\,\theta_s^{(n-1)}\Big] + (1-\lambda)\,\theta_s^{(n)}\\[2pt]
&= (1-\lambda)\,\theta_s^{(n)} + (1-\lambda)\lambda\,\theta_s^{(n-1)} + \lambda^{2}\theta_t^{(n-2)}\\[2pt]
&= \cdots
\end{aligned}
$$

$n$번 끝까지 밀어내면

$$\boxed{\;\theta_t^{(n)} = (1-\lambda)\sum_{k=0}^{n-1}\lambda^{k}\,\theta_s^{(n-k)} \;+\; \lambda^{n}\theta_t^{(0)}\;}$$

여기서 읽어야 할 것이 세 가지다.

**(a) 계수의 합이 1에 수렴한다 — 진짜 "평균"이다.**

등비급수의 합 공식 $\sum_{k=0}^{n-1}\lambda^k = \dfrac{1-\lambda^n}{1-\lambda}$을 쓰면

$$(1-\lambda)\sum_{k=0}^{n-1}\lambda^{k} = (1-\lambda)\cdot\frac{1-\lambda^{n}}{1-\lambda} = 1-\lambda^{n}$$

여기에 초기항 계수 $\lambda^n$을 더하면 정확히 $1$. 즉 $\theta_t^{(n)}$은 $\theta_s^{(1)},\dots,\theta_s^{(n)}$과 $\theta_t^{(0)}$의 **가중평균**이며, $0<\lambda<1$이므로 $\lambda^n \to 0$, 초기값의 영향은 지수적으로 잊힌다.

**(b) 가중치가 지수적으로 감쇠한다.**

$k$ 스텝 전의 student가 갖는 몫은

$$w_k = (1-\lambda)\lambda^{k}$$

$k$에 대한 **등비수열**(공비 $\lambda$)이다. $\lambda = 0.996$일 때

| $k$ (몇 스텝 전) | $w_k = 0.004 \times 0.996^k$ | 최신 대비 비율 $\lambda^k$ |
|---|---|---|
| 0 | 0.004000 | 1.000 |
| 100 | 0.002681 | 0.670 |
| 250 | 0.001469 | 0.367 ($\approx 1/e$) |
| 500 | 0.000539 | 0.135 |
| 1000 | 0.0000727 | 0.018 |

가까운 과거는 크게, 먼 과거는 작게 — 그러나 **완전히 0은 아니게** 반영된다. 이게 "지수이동평균(EMA, exponential moving average)"이라는 이름의 뜻이다.

**(c) 이건 통계의 "여러 모델을 평균내기"와 같다.** 논문은 이 성질을 지수감쇠를 가진 **Polyak–Ruppert 평균**이라 부르며, 그래서 teacher가 학습 내내 student보다 성능이 좋다고 보고한다. 노이즈가 섞인 값들을 평균내면 노이즈가 줄어드는 것과 같은 원리다 (확률과 통계의 "표본평균의 분산이 $\sigma^2/n$로 줄어든다"와 같은 직관).

---

## 3. 유효 평균 길이: 왜 $1/(1-\lambda)$인가

"그래서 teacher는 대략 **몇 스텝**을 평균내는 건가?" 이를 정량화하는 표준적인 방법이 두 가지 있다. 둘 다 같은 답을 준다.

### 방법 1 — 가중치의 무게중심(평균 지연 시간)

물리의 무게중심 계산과 같다. $k$ 스텝 전 항의 무게가 $w_k=(1-\lambda)\lambda^k$이므로, 평균적으로 **얼마나 과거를 보고 있는지**는

$$\bar{k} = \sum_{k=0}^{\infty} k\,w_k = (1-\lambda)\sum_{k=0}^{\infty} k\,\lambda^{k}$$

여기서 $\sum_{k=0}^{\infty} k\lambda^k = \dfrac{\lambda}{(1-\lambda)^2}$ (등비급수 $\sum \lambda^k = \frac{1}{1-\lambda}$를 $\lambda$로 미분한 뒤 $\lambda$를 곱하면 나온다 — 미적분 활용). 따라서

$$\bar{k} = (1-\lambda)\cdot\frac{\lambda}{(1-\lambda)^2} = \frac{\lambda}{1-\lambda} \;\approx\; \frac{1}{1-\lambda}\quad(\lambda \approx 1)$$

### 방법 2 — 감쇠 시간 상수

가중치가 최신값의 $1/e \approx 0.368$배로 줄어드는 $k$를 찾는다. $\lambda^{k} = e^{-1}$에서

$$k = \frac{-1}{\ln \lambda} \approx \frac{1}{1-\lambda}$$

(마지막 근사는 $\lambda = 1-\varepsilon$, $\varepsilon$이 작을 때 $\ln(1-\varepsilon)\approx-\varepsilon$을 쓴 것. 자연로그의 근사식.)

### 숫자로 확인

$$\lambda = 0.996 \;\Longrightarrow\; \frac{1}{1-\lambda} = \frac{1}{0.004} = 250$$

**teacher는 대략 최근 250 스텝의 student를 평균낸 모델이다.** ImageNet 배치 1024 기준으로 보면 대략 한 epoch의 1/5쯤 되는 구간이다.

$\lambda$를 조금만 바꿔도 극적으로 달라진다.

| $\lambda$ | $1/(1-\lambda)$ | 의미 |
|---|---|---|
| 0 | 1 | teacher = student 복사 (평균 없음) |
| 0.9 | 10 | 아주 짧은 평균 |
| 0.99 | 100 | |
| **0.996** | **250** | **DINO 시작값** |
| 0.9999 | 10,000 | 거의 얼어붙음 |
| $\to 1$ | $\to \infty$ | **사실상 정지 (teacher 고정)** |

극단을 확인해 보면 식의 의미가 확실해진다.

- $\lambda = 0$: $\theta_t \leftarrow \theta_s$. teacher가 student의 완전한 복사본이 된다. 논문은 이 설정이 **수렴하지 않는다**고 보고한다 (student가 자기 자신을 목표로 삼으면 상수 출력으로 붕괴하기 쉽다).
- $\lambda = 1$: $\theta_t \leftarrow \theta_t$. teacher가 **전혀 변하지 않는다.** 목표가 완전히 고정된다.

DINO는 이 두 극단 사이를 오간다. 그것도 **시간에 따라**.

---

## 4. cosine 스케줄: $\lambda$를 0.996에서 1로 끌어올리기

$\lambda$는 상수가 아니다. 전체 학습 스텝 수를 $T$, 현재 스텝을 $t$라 할 때

$$\boxed{\;\lambda(t) = 1 - (1-\lambda_0)\cdot\frac{1+\cos(\pi t/T)}{2}\;},\qquad \lambda_0 = 0.996$$

### 식 읽기

$\frac{1+\cos\theta}{2}$는 삼각함수의 낯익은 도구다. $\theta = \pi t/T$가 $0 \to \pi$로 갈 때 $\cos\theta$는 $1 \to -1$이므로

$$\frac{1+\cos(\pi t/T)}{2}:\quad 1 \;\longrightarrow\; 0 \quad\text{(부드럽게 감소)}$$

이 값을 $s(t)$라 하면 $\lambda(t) = 1 - (1-\lambda_0)\,s(t)$이므로

- $t=0$: $s=1$ $\Rightarrow$ $\lambda = 1-(1-0.996)\cdot 1 = 0.996$ ✓
- $t=T/2$: $s = \frac{1+\cos(\pi/2)}{2} = 0.5$ $\Rightarrow$ $\lambda = 1 - 0.004\cdot 0.5 = 0.998$
- $t=T$: $s=0$ $\Rightarrow$ $\lambda = 1$ ✓

정확히 "0.996에서 1까지"를 만족한다.

### 왜 직선이 아니라 cosine인가

$\lambda(t)$를 미분해 보자 (합성함수의 미분).

$$\lambda'(t) = -(1-\lambda_0)\cdot\frac{1}{2}\cdot\left(-\sin\frac{\pi t}{T}\right)\cdot\frac{\pi}{T} = \frac{\pi(1-\lambda_0)}{2T}\sin\frac{\pi t}{T}$$

$\sin(\pi t/T)$는 $t=0$과 $t=T$에서 0, $t=T/2$에서 최대다. 즉

- **양 끝($t=0$, $t=T$)에서 기울기가 0** — 시작과 끝에서 $\lambda$가 완만하게 붙는다. 갑작스러운 변화가 없어 학습이 흔들리지 않는다.
- **중간에서 가장 빠르게 변한다.**

직선 스케줄이었다면 $t=T$ 직전까지 일정한 속도로 변하다 뚝 끊긴다. cosine은 **부드럽게 착지**한다. 이 성질 때문에 learning rate 스케줄에서도 cosine이 표준으로 쓰인다 (DINO 논문도 learning rate와 weight decay에 같은 cosine을 쓴다).

### 유효 윈도로 번역하면 — 이게 진짜 의미다

$\lambda$ 자체는 0.996 → 1로 "겨우 0.004" 변한다. 별것 아닌 것처럼 보인다. 하지만 우리가 3절에서 구한 $1/(1-\lambda)$로 바꿔 보면 이야기가 완전히 달라진다.

$$\frac{1}{1-\lambda(t)} = \frac{1}{(1-\lambda_0)\cdot\frac{1+\cos(\pi t/T)}{2}} = \frac{250}{\frac{1+\cos(\pi t/T)}{2}}$$

| 진행률 $t/T$ | $\lambda(t)$ | 유효 윈도 $1/(1-\lambda)$ |
|---|---|---|
| 0.00 | 0.996000 | 250 스텝 |
| 0.25 | 0.996586 | 293 스텝 |
| 0.50 | 0.998000 | 500 스텝 |
| 0.75 | 0.999414 | 1,707 스텝 |
| 0.90 | 0.999902 | 10,215 스텝 |
| 0.99 | 0.9999990 | 약 101만 스텝 |
| 1.00 | 1.000000 | $\infty$ (정지) |

**$\lambda$는 선형에 가깝게 조금 변하지만, 유효 윈도는 발산한다.** 분모 $1-\lambda$가 0으로 가기 때문이다 (반비례 관계의 극한). 이것이 "학습 후반으로 갈수록 teacher를 더 느리게 움직인다"는 말의 정확한 의미다.

### 왜 그렇게 설계했나 — 학습 단계별 역할

**초반 ($\lambda \approx 0.996$, 윈도 250):**
student는 아직 형편없고, 매 스텝 크게 바뀐다. teacher가 너무 느리면 옛날의 나쁜 표현을 계속 목표로 주게 되어 학습이 진척되지 않는다. 그래서 비교적 민첩하게(250 스텝 평균) 따라간다.

**후반 ($\lambda \to 1$, 윈도 $\to \infty$):**
student는 이미 좋은 표현에 도달했고, 남은 것은 배치 샘플링과 augmentation에서 오는 **노이즈**뿐이다. 평균 구간을 넓힐수록 노이즈가 더 지워진다(확률과 통계: 평균낼 표본이 많을수록 분산이 준다). 동시에 목표가 거의 고정되므로 student가 **안정된 과녁**을 향해 수렴할 수 있다.

$$\text{초반: 빠른 추종(bias 감소)} \quad\longleftrightarrow\quad \text{후반: 강한 평활(variance 감소)}$$

learning rate를 후반에 0으로 줄이는 것과 같은 철학이다. 실제로 두 스케줄은 서로 짝을 이룬다: learning rate가 줄어 student가 덜 움직이고, $\lambda$가 커져 teacher는 더욱 안 움직인다. 학습 끝에서 두 네트워크가 함께 "얼어붙으며" 수렴한다.

---

## 5. 붕괴(collapse)를 막는 역할

DINO는 라벨이 없다. student가 teacher의 출력을 맞추도록 학습하는데, 만약 teacher = student라면 "모든 입력에 대해 같은 값을 뱉는다"가 손실 0인 완벽한(그러나 쓸모없는) 해가 된다. 이것이 **붕괴**다.

EMA teacher는 두 가지로 이를 막는다.

1. **시간 지연.** teacher는 250 스텝 전 student들의 평균이므로, student가 지금 만들어낸 지름길을 즉시 따라오지 않는다. 목표가 항상 "과거"에 있어 자기참조 고리가 끊긴다.
2. **stop-gradient.** teacher 쪽으로는 그래디언트가 흐르지 않는다. teacher는 학습되는 것이 아니라 오직 위 EMA 식으로만 바뀐다.

논문의 ablation(Table 7)에서 momentum(=EMA teacher)을 빼면 프레임워크가 아예 작동하지 않고, Sinkhorn-Knopp 같은 추가 정규화 없이는 붕괴한다고 보고한다. 반대로 momentum이 있으면 단순한 centering + sharpening만으로 충분하다.

또한 논문은 "이전 epoch의 student를 teacher로 쓰기"(붕괴하지 않지만 성능은 열등), "이전 iteration의 student"나 "student 복사본"(수렴 실패)과 비교해, **EMA가 가장 좋다**고 정리한다.

---

## 6. 요약

| 항목 | 내용 |
|---|---|
| 갱신 규칙 | $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ (가중평균 점화식) |
| 펼친 형태 | $\theta_t^{(n)} = (1-\lambda)\sum_{k}\lambda^{k}\theta_s^{(n-k)} + \lambda^n\theta_t^{(0)}$ |
| 가중치 성질 | 공비 $\lambda$의 등비수열, 계수 합 $=1$ → 지수감쇠 가중평균(EMA) |
| 유효 평균 길이 | $\dfrac{\lambda}{1-\lambda}\approx\dfrac{1}{1-\lambda}$ |
| $\lambda$ 스케줄 | $\lambda(t) = 1-(1-\lambda_0)\dfrac{1+\cos(\pi t/T)}{2}$, $\lambda_0=0.996 \to 1$ |
| 윈도 변화 | 250 스텝 → $\infty$ (후반일수록 teacher가 느려짐 = 목표 고정) |
| 별칭 | momentum encoder, mean teacher, 지수감쇠 Polyak–Ruppert 평균 |
| 효과 | 노이즈 평활, 붕괴 방지, teacher가 student보다 항상 우수한 목표 제공 |

**한 문장:** DINO의 teacher는 학습되는 네트워크가 아니라 student 궤적의 지수가중 평균이며, 그 "기억의 길이"가 250 스텝에서 무한대로 서서히 늘어나도록 cosine 스케줄이 조절한다.
