# 고정된 teacher가 있을 때 DINO가 최소화하는 것

## 0. 한 줄 답

$$\min_{\theta_s} H\big(P_t(x),\, P_s(x)\big), \qquad H(a,b) = -\sum_{i=1}^{K} a^{(i)} \log b^{(i)}$$

즉 **teacher의 출력 분포 $P_t(x)$를 정답처럼 놓고, student의 출력 분포 $P_s(x)$가 그것과 같아지도록 student 파라미터 $\theta_s$만 움직인다.** (논문 §3.1 식 (2), 원문 표기는 축약형 $H(a,b) = -a\log b$)

---

## 1. 무대 설정: 두 개의 "확률분포"

고교 확률과 통계에서 배운 확률분포를 떠올리자. 주사위 눈 1~6에 확률을 하나씩 붙이면 확률분포다. 여기서는 눈이 6개가 아니라 $K$개다.

student 신경망 $g_{\theta_s}$와 teacher 신경망 $g_{\theta_t}$는 이미지 $x$를 받아 $K$개의 실수를 뱉는다. 이 실수 벡터를 softmax로 눌러서 확률분포로 만든다 (논문 식 (1)):

$$P_s(x)^{(i)} = \frac{\exp\!\big(g_{\theta_s}(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\!\big(g_{\theta_s}(x)^{(k)}/\tau_s\big)}$$

- 분자가 지수함수 $\exp$ 이므로 **항상 양수**
- 분모가 모든 항의 합이므로 **전부 더하면 1**

두 조건이 곧 "확률분포"의 정의다. teacher도 같은 식에 온도 $\tau_t$를 써서 $P_t(x)$를 만든다.

목표: 이 두 확률분포를 **같게** 만들기. 그런데 "두 분포가 얼마나 다른가"를 재는 자를 하나 정해야 한다. 그 자가 cross-entropy $H(a,b)$다.

> 표기 주의: 이 노트에서 $H(a,b)$(인자 2개)는 **cross-entropy**, $H(a)$ 또는 $h(a)$(인자 1개)는 **entropy**다. 논문은 식 (5)에서 entropy를 소문자 $h$로 써서 구별한다.

---

## 2. 왜 $H(a,b) = -\sum_i a^{(i)}\log b^{(i)}$ 가 "$b$를 $a$에 맞추는" 함수인가

### 2-1. 로그의 성질부터

$\log$ (자연로그 $\ln$으로 생각하자)는 $0 < b \le 1$ 구간에서

- $b = 1$ 이면 $\log b = 0$
- $b$ 가 0에 가까워지면 $\log b \to -\infty$

따라서 $-\log b$ 는 **"확률 $b$를 배정해 놓고 실제로 그 사건이 일어났을 때 받는 벌점"** 이다. 확신했는데($b\approx1$) 맞으면 벌점 0, 거의 0이라고 했는데($b\approx0$) 일어나면 벌점이 무한대로 커진다.

### 2-2. 가중평균 구조

$$H(a,b) = \sum_{i=1}^{K} a^{(i)} \cdot \big(-\log b^{(i)}\big)$$

이건 벌점 $-\log b^{(i)}$ 를 **가중치 $a^{(i)}$ 로 가중평균한 것**, 즉 기댓값이다. 고교식으로 쓰면

$$H(a,b) = E_{i \sim a}\big[-\log b^{(i)}\big]$$

"teacher가 중요하다고 본 항목($a^{(i)}$이 큰 곳)에서 student가 낮은 확률을 주면 크게 혼난다"는 뜻이다. 반대로 teacher가 무시하는 항목($a^{(i)} \approx 0$)은 student가 뭘 하든 가중치가 0이라 벌점에 거의 기여하지 않는다.

### 2-3. 구체적 숫자로 확인

$K=3$, teacher 분포를 고정:

$$a = P_t = (0.7,\; 0.2,\; 0.1)$$

student 후보 세 개를 넣어 보자 (자연로그, $\ln 0.7 \approx -0.357$, $\ln 0.2 \approx -1.609$, $\ln 0.1 \approx -2.303$, $\ln \tfrac13 \approx -1.099$):

| student $b$ | 계산 | $H(a,b)$ |
|---|---|---|
| $b_1=(0.7,0.2,0.1)$ (teacher와 동일) | $0.7(0.357)+0.2(1.609)+0.1(2.303)$ | $\approx 0.802$ |
| $b_2=(1/3,1/3,1/3)$ (아무것도 모름) | $1.0 \times 1.099$ | $\approx 1.099$ |
| $b_3=(0.1,0.2,0.7)$ (teacher와 반대) | $0.7(2.303)+0.2(1.609)+0.1(0.357)$ | $\approx 1.969$ |

**teacher와 똑같이 맞췄을 때 값이 가장 작다.** 특히 $b_3$처럼 teacher가 0.7을 준 자리에 0.1을 주면, 가장 큰 가중치 0.7에 가장 큰 벌점 2.303이 곱해져서 손실이 폭발한다.

### 2-4. 미분으로 본 "밀어내는 힘"

$b^{(i)}$ 에 대해 편미분하면

$$\frac{\partial H(a,b)}{\partial b^{(i)}} = -\frac{a^{(i)}}{b^{(i)}}$$

항상 음수이므로 경사하강법은 모든 $b^{(i)}$를 키우려 한다. 그런데 $\sum_i b^{(i)} = 1$ 이라 다 키울 수는 없다. "키우려는 힘"의 세기가 $a^{(i)}/b^{(i)}$ 이므로, **teacher 확률에 비해 student 확률이 작은 자리일수록 더 세게 끌어올려진다.** 힘이 평형을 이루는 지점이 $b^{(i)} \propto a^{(i)}$, 즉 $b = a$다.

### 2-5. 최소가 정말 $b=a$ 인가 (엄밀히)

$\ln x \le x-1$ 을 쓰면 된다. 이건 고교 미적분으로 바로 증명된다. $f(x) = x - 1 - \ln x$ 라 두면

$$f'(x) = 1 - \frac1x,\qquad f'(x)=0 \iff x=1$$

$x<1$에서 $f'<0$, $x>1$에서 $f'>0$ 이므로 $x=1$이 최소점이고 $f(1)=0$. 따라서 모든 $x>0$에서 $f(x)\ge 0$, 즉 $\ln x \le x-1$ (등호는 $x=1$일 때만).

이제 $x = b^{(i)}/a^{(i)}$ 를 대입하고 $a^{(i)}$를 곱해 더하면

$$\sum_i a^{(i)} \ln\frac{b^{(i)}}{a^{(i)}} \;\le\; \sum_i a^{(i)}\left(\frac{b^{(i)}}{a^{(i)}} - 1\right) = \sum_i b^{(i)} - \sum_i a^{(i)} = 1 - 1 = 0$$

부호를 뒤집으면

$$\sum_i a^{(i)} \ln\frac{a^{(i)}}{b^{(i)}} \;\ge\; 0, \qquad \text{등호} \iff b^{(i)}=a^{(i)}\ \ \forall i$$

왼쪽 식이 바로 다음 절의 KL divergence다.

---

## 3. $H(a,b) = H(a) + D_{KL}(a\|b)$ 분해 — 한 줄씩 쌓아 올리기

논문 §5.3 식 (5)에 나오는 분해다: $H(P_t,P_s) = h(P_t) + D_{KL}(P_t\|P_s)$.

### 3-1. 유도 (로그의 성질만 씀)

시작:

$$H(a,b) = -\sum_i a^{(i)} \log b^{(i)}$$

**1단계.** 아무것도 안 바꾸는 트릭으로 $\log a^{(i)}$ 를 더하고 뺀다.

$$H(a,b) = -\sum_i a^{(i)} \log b^{(i)} + \sum_i a^{(i)} \log a^{(i)} - \sum_i a^{(i)} \log a^{(i)}$$

**2단계.** 앞의 두 항을 묶고 $\log A - \log B = \log \dfrac{A}{B}$ 를 쓴다.

$$-\sum_i a^{(i)} \log b^{(i)} + \sum_i a^{(i)} \log a^{(i)} = \sum_i a^{(i)}\big(\log a^{(i)} - \log b^{(i)}\big) = \sum_i a^{(i)} \log \frac{a^{(i)}}{b^{(i)}}$$

**3단계.** 이름 붙이기.

$$\underbrace{\sum_i a^{(i)} \log \frac{a^{(i)}}{b^{(i)}}}_{D_{KL}(a\|b)},\qquad \underbrace{-\sum_i a^{(i)} \log a^{(i)}}_{H(a)\ \text{(entropy)}}$$

**결론.**

$$\boxed{\,H(a,b) = H(a) + D_{KL}(a\|b)\,}$$

### 3-2. 각 항의 뜻

- $H(a)$ (entropy): 분포 $a$ **하나만으로** 정해지는 값. "$a$가 얼마나 퍼져 있나(불확실한가)"의 척도. 한 곳에 확률이 몰리면 작고, 균등분포면 최대.
- $D_{KL}(a\|b)$: 2-5절에서 보았듯 **항상 $\ge 0$, 그리고 $0$이 되는 것은 오직 $a=b$일 때**. 즉 "$b$가 $a$에서 얼마나 벗어났나"의 척도.

숫자 예로 검산하면: $H(a) = 0.802$ (위 표의 $b_1$ 행 값과 같다). 그러면
$D_{KL}(a\|b_2) = 1.099 - 0.802 = 0.297$, $D_{KL}(a\|b_3) = 1.969 - 0.802 = 1.167$, $D_{KL}(a\|b_1)=0$. 전부 0 이상이고 일치할 때만 0이다.

### 3-3. 그래서 cross-entropy 최소화 = KL 최소화

우리가 움직일 수 있는 변수는 $\theta_s$ 뿐이다. $\theta_s$는 $P_s$만 바꾸고 $P_t$는 건드리지 못한다. 따라서

$$\min_{\theta_s} H(P_t, P_s) = \min_{\theta_s} \Big[\underbrace{h(P_t)}_{\theta_s\text{와 무관한 상수}} + D_{KL}(P_t\|P_s)\Big] = h(P_t) + \min_{\theta_s} D_{KL}(P_t\|P_s)$$

상수를 더하는 것은 최소점의 **위치**를 바꾸지 않는다 (그래프를 위아래로 평행이동할 뿐이다 — 이차함수 $y=x^2$와 $y=x^2+5$의 최소점이 둘 다 $x=0$인 것과 같다). 미분으로 봐도

$$\frac{\partial}{\partial \theta_s} h(P_t) = 0 \quad\Longrightarrow\quad \frac{\partial}{\partial \theta_s} H(P_t,P_s) = \frac{\partial}{\partial \theta_s} D_{KL}(P_t\|P_s)$$

즉 **경사하강법이 보는 기울기가 완전히 같다.** 그러므로

> cross-entropy를 최소화하는 것 = KL divergence를 최소화하는 것 = $P_s$를 $P_t$에 일치시키는 것

이고, 이론상 최적해는 $P_s(x) = P_t(x)$, 손실의 하한은 $h(P_t)$다.

---

## 4. "teacher가 고정"이라는 전제가 왜 중요한가

### 4-1. 3-3절 논증의 급소

위 유도에서 **딱 하나** 결정적으로 쓴 가정이 "$h(P_t)$는 $\theta_s$와 무관한 상수"였다. 만약 teacher도 같은 손실로 함께 학습된다면 $P_t$도 최적화 변수가 되고, 그러면 시스템 전체가 훨씬 쉬운 탈출구를 찾는다:

$$\text{"student와 teacher가 입력과 무관하게 똑같은 상수 분포를 뱉기로 담합"} \;\Rightarrow\; H = h(P_t),\ D_{KL}=0$$

손실은 낮아지지만 배운 것은 아무것도 없다. 이것이 **collapse**다. 손실 함수는 "$P_s$를 $P_t$ 쪽으로" 끌어당기도록 설계됐는데, $P_t$까지 자유롭게 두면 표적이 사수 쪽으로 걸어와 버린다.

### 4-2. stop-gradient (`detach`)

그래서 DINO는 teacher 쪽 경로로 기울기가 흐르지 못하게 막는다. 논문 Figure 2의 `sg` 연산자이고, Algorithm 1 의사코드에서는 손실 함수 첫 줄이다:

```python
def H(t, s):
    t = t.detach()                      # stop gradient  ← teacher는 상수 취급
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1)   # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```

`t.detach()`의 의미는 정확히 "이 값을 **$\theta_s$와 무관한 숫자 상수**로 간주하라"이다. 3-3절의 유도를 코드로 강제한 것이 stop-gradient다. 역전파는 오직 student 가지로만 흘러 $\theta_s$만 갱신된다.

### 4-3. 그런데 DINO의 teacher는 진짜 고정이 아니다 — 준(準)고정 목표

지식 증류(knowledge distillation) 원래 설정에서는 미리 학습된 teacher가 주어진다. 그러나 DINO는 라벨도 없고 미리 준비된 teacher도 없다. 논문(§3.1, "Teacher network")은 **student의 과거 이력으로 teacher를 만든다**:

$$\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$$

$\lambda$는 코사인 스케줄로 $0.996 \to 1$. 이것이 EMA(exponential moving average, 지수이동평균)이자 momentum encoder다.

$\lambda = 0.996$ 이면 한 스텝에서 teacher는 student 쪽으로 **0.4%만** 움직인다. 즉

- **한 스텝 안에서는** teacher가 완전히 고정된 상수다 → 식 (2)의 논리가 그대로 성립
- **긴 시간 축에서는** teacher가 천천히 따라 움직인다 → 표적이 조금씩 좋아지면서 student를 계속 앞으로 끌어준다

이런 성질을 "준-고정(quasi-static) 목표"라 부른다. 물리에서 준정적 과정(피스톤을 아주 천천히 밀면 매 순간을 평형 상태로 취급할 수 있는 것)과 같은 발상이다. 왜 천천히여야 하는가:

- $\lambda$가 작아 teacher가 student를 즉시 복사하면 ($\theta_t = \theta_s$) → 논문 표현대로 **수렴 실패**. 4-1절의 담합 문제가 그대로 살아난다.
- 반대로 EMA는 student의 최근 궤적을 평균낸 것이라 잡음이 줄어든 앙상블 모델이 된다 (Polyak-Ruppert 평균). 논문은 이 teacher가 **학습 내내 student보다 성능이 좋았다**고 보고한다. 즉 표적이 항상 사수보다 조금 앞서 있어서, student가 따라가면 실제로 나아진다.

또 하나의 뉘앙스: EMA teacher에서는 $h(P_t)$가 학습 **전체**에 걸쳐 상수는 아니다. 매 스텝의 기울기 계산에 대해서만 상수다. 그리고 $h(P_t)$가 0으로 붕괴하거나(한 차원 지배) 최대로 커지는(균등분포) 두 종류의 collapse를 막기 위해 DINO는 teacher 출력에 **centering**(평균 빼기, $h$를 키우는 방향)과 **sharpening**(낮은 온도 $\tau_t$, $h$를 줄이는 방향)을 동시에 걸어 균형을 잡는다. 논문 §5.3이 식 (5)의 분해를 꺼내는 이유가 정확히 이것이다.

---

## 5. 실제로 학습되는 최종 형태

식 (2)를 그대로 쓰면 $P_t(x)$와 $P_s(x)$가 같은 입력 $x$를 보므로, 두 네트워크가 같아지면 끝나는 시시한 문제다. DINO는 **서로 다른 crop을 보게** 해서 의미 있는 문제로 만든다 (식 (3)):

$$\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)$$

teacher는 전역(global) crop만, student는 전역 + 지역(local) crop 전부를 본다. "이미지의 좁은 조각만 보고도, 전체를 본 teacher가 내린 분포를 맞혀라" → local-to-global 대응을 배우게 된다. 의사코드의 `loss = H(t1, s2)/2 + H(t2, s1)/2`가 crop 2개짜리 최소 버전이다.

---

## 6. 요약 카드

| 질문 | 답 |
|---|---|
| 목적함수 | $\min_{\theta_s} H(P_t(x), P_s(x))$, $H(a,b) = -\sum_i a^{(i)}\log b^{(i)}$ (논문 식 (2)) |
| 최적화 변수 | student 파라미터 $\theta_s$ **만** |
| 왜 이 함수인가 | teacher 확률을 가중치로 한 $-\log(\text{student 확률})$의 기댓값 → teacher가 중시하는 자리에서 student가 낮은 확률을 주면 벌점 폭발 |
| 분해 | $H(a,b) = H(a) + D_{KL}(a\|b)$, $D_{KL}\ge 0$ (등호는 $a=b$) |
| 그래서 | $H(P_t)$는 $\theta_s$와 무관한 상수 → CE 최소화 = KL 최소화 = $P_s \to P_t$ |
| 최적해와 하한 | $P_s(x) = P_t(x)$, 손실 하한 $h(P_t)$ |
| "고정"의 역할 | stop-gradient(`t.detach()`)로 강제. 안 그러면 두 망이 상수 출력으로 담합(collapse) |
| DINO의 현실 | teacher는 $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ ($\lambda: 0.996 \to 1$)로 천천히 움직이는 **준-고정** 목표 |
