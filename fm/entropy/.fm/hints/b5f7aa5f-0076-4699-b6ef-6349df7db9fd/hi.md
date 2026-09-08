# 교차엔트로피 = 엔트로피 + KL

$$H(q, p) \;=\; h(q) \;+\; D_{KL}(q \,\|\, p)$$

이 한 줄이 DINO에서 왜 붕괴(collapse)가 일어나는지를 설명하는 열쇠다. 고등학교에서 배운 로그 성질과 접선 부등식만으로 전부 유도할 수 있다.

---

## 1. 등장인물 세 명

확률분포란 $K$개의 칸에 확률을 나눠 담은 것이다. $p = (p_1, \dots, p_K)$, 모든 $p_i \ge 0$, $\sum_i p_i = 1$.

DINO에서는 $q$ = teacher 네트워크의 softmax 출력, $p$ = student 네트워크의 softmax 출력이다.

| 이름 | 식 | 한 줄 뜻 |
|---|---|---|
| 엔트로피 | $h(q) = -\sum_i q_i \log q_i$ | $q$ 하나가 얼마나 "퍼져 있나" |
| 교차엔트로피 | $H(q, p) = -\sum_i q_i \log p_i$ | $q$를 정답으로 놓고 $p$를 채점한 점수 (= DINO의 loss) |
| KL 발산 | $D_{KL}(q\|p) = \sum_i q_i \log \dfrac{q_i}{p_i}$ | $p$가 $q$와 얼마나 다른가 |

셋 다 "$q_i$를 가중치로 하는 평균"이라는 점이 같다. 안에 든 것만 각각 $-\log q_i$, $-\log p_i$, $\log(q_i/p_i)$로 다르다.

---

## 2. 항등식 유도 — 로그 한 줄 트릭

핵심은 딱 이거다. $\log p_i$를 억지로 $\log q_i$가 보이게 쪼갠다.

$$\log p_i \;=\; \log q_i \;-\; \bigl(\log q_i - \log p_i\bigr)$$

당연히 참이다(오른쪽을 풀면 $\log q_i - \log q_i + \log p_i = \log p_i$). 이걸 교차엔트로피에 그대로 대입한다.

$$
\begin{aligned}
H(q,p) &= -\sum_i q_i \log p_i \\
       &= -\sum_i q_i \Bigl[\log q_i - (\log q_i - \log p_i)\Bigr] \\
       &= \underbrace{-\sum_i q_i \log q_i}_{h(q)} \;+\; \underbrace{\sum_i q_i (\log q_i - \log p_i)}_{D_{KL}(q\|p)} \\
       &= h(q) + D_{KL}(q \| p)
\end{aligned}
$$

끝이다. 미분도, 극한도 필요 없다. **로그 하나를 더하고 빼서 항을 갈라놓은 것뿐**이다.

로그 성질 $\log q_i - \log p_i = \log\frac{q_i}{p_i}$를 쓰면 뒷항이 정확히 KL 정의와 같다는 것도 바로 보인다.

> **주의**: 이 항등식은 $q$ 기준으로만 성립한다. $H(q,p) = h(p) + \dots$ 같은 건 없다. 앞에 붙는 엔트로피는 **가중치로 쓰인 쪽**, 즉 teacher의 엔트로피다.

---

## 3. "코드 길이" 비유 — 각 항이 뜻하는 것

정보이론에서 $-\log_2 p_i$는 "확률 $p_i$인 사건에 배정할 부호(코드)의 비트 수"다. 흔한 사건은 짧게, 드문 사건은 길게 적는 게 이득이다. (모스 부호에서 E가 점 하나인 것과 같은 원리다.)

이 관점에서 세 양을 다시 읽으면:

- $h(q) = \sum_i q_i \cdot (-\log q_i)$
  → **진짜 분포 $q$에 딱 맞춰 만든 최적 코드북의 평균 길이.** 이론적 하한이다. 아무리 잘해도 이보다 짧게는 못 줄인다.

- $H(q,p) = \sum_i q_i \cdot (-\log p_i)$
  → **$p$가 맞다고 착각하고 만든 코드북을, 실제로는 $q$에서 데이터가 나오는 상황에서 쓸 때의 평균 길이.** 잘못된 통계로 코드북을 짰으니 손해를 본다.

- $D_{KL}(q\|p)$
  → **그 손해분.** "$p$를 믿은 대가로 추가로 지불하는 비트 수".

$$\underbrace{\text{실제 지불한 길이}}_{H(q,p)} = \underbrace{\text{피할 수 없는 최소 길이}}_{h(q)} + \underbrace{\text{잘못 믿어서 낸 벌금}}_{D_{KL}(q\|p)}$$

**벌금은 0 이상이고, $p = q$일 때만 0이다.** 이게 다음 절이다.

---

## 4. $D_{KL} \ge 0$ 증명 — 접선 부등식 하나로

### 준비: $\log x \le x - 1$

$f(x) = x - 1 - \log x$ ($x > 0$)를 미분하면

$$f'(x) = 1 - \frac{1}{x}, \qquad f''(x) = \frac{1}{x^2} > 0$$

$f'(x) = 0$은 $x = 1$에서, 그리고 $f'' > 0$이므로 $x=1$이 **최솟값**. 그 값은 $f(1) = 1 - 1 - 0 = 0$.

따라서 모든 $x > 0$에서

$$\boxed{\log x \le x - 1}, \qquad \text{등호는 } x = 1 \text{일 때만}$$

기하적으로는 "$y = \log x$ 곡선은 점 $(1,0)$에서 그은 접선 $y = x-1$ 아래에 항상 있다"는 뜻이다. $\log$가 위로 볼록(concave)하니까.

### 본론

$-D_{KL}$을 만들어 부등식을 적용한다. $x = \dfrac{p_i}{q_i}$로 놓자.

$$
\begin{aligned}
-D_{KL}(q\|p) &= \sum_i q_i \log \frac{p_i}{q_i} \\
&\le \sum_i q_i \left(\frac{p_i}{q_i} - 1\right) \qquad (\log x \le x-1) \\
&= \sum_i p_i - \sum_i q_i \\
&= 1 - 1 = 0
\end{aligned}
$$

즉 $-D_{KL} \le 0$, 곧

$$D_{KL}(q \| p) \ge 0$$

등호 조건은 모든 $i$에서 $p_i/q_i = 1$, 즉 **$p = q$일 때만**이다.

### 따라 나오는 것

$$H(q, p) = h(q) + \underbrace{D_{KL}(q\|p)}_{\ge 0} \;\ge\; h(q)$$

**교차엔트로피는 절대로 $h(q)$ 아래로 못 내려간다.** $q$를 고정해 놓고 $p$를 아무리 잘 맞춰도 바닥은 $h(q)$다.

---

## 5. 왜 DINO에서 이게 결정적인가

DINO의 loss는 (레이블 없이) teacher 출력 $q$와 student 출력 $p$의 교차엔트로피다.

```python
loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
#      = H(q, p)
```

이제 분해해 보자.

$$\text{loss} = h(q) + D_{KL}(q\|p)$$

**loss를 낮추는 길이 두 개** 있다.

### (a) $D_{KL}(q\|p) \downarrow$ — 우리가 원하는 것

student가 teacher를 따라간다. $p \to q$. 이게 지식 증류(distillation)의 본래 의도다. teacher가 "이 이미지는 3번 프로토타입"이라 말하면 student도 그렇게 말하도록 배운다. 서로 다른 crop을 봤는데도 같은 답을 내려면 **이미지의 의미를 봐야** 하므로, 표현이 학습된다.

### (b) $h(q) \downarrow$ — 붕괴 통로

**teacher 분포 자체를 뾰족하게 만든다.** $q$가 원-핫에 가까워지면 $h(q) \to 0$이고, $p$도 같이 그 원-핫이면 $D_{KL} = 0$이라 **loss = 0**이다. 완벽한 점수다.

지도학습이라면 (b)는 불가능하다. $q$가 사람이 준 레이블이라 고정이고, $h(q)$는 상수다. 상수는 최적화 대상이 아니므로 $\arg\min$ 관점에서 $H$를 줄이는 것과 $D_{KL}$을 줄이는 것이 완전히 같다.

**그런데 DINO에서 teacher는 student의 EMA다.** 즉 $q$도 학습에 따라 움직인다. $h(q)$가 더 이상 상수가 아니다. 경사하강은 loss가 내려가는 방향이면 뭐든 잡는데, 하필 (b)가 (a)보다 훨씬 쉽다.

- (a)는 "이미지를 이해해서 crop이 달라도 같은 답을 내라" — 어렵다.
- (b)는 "입력이 뭐든 무조건 2번 차원에 몰빵해라" — 파라미터 몇 개만 키우면 된다.

이게 **원-핫 붕괴**다. 노트북 4절이 보여주듯, 입력을 전혀 안 보는 상수 출력 네트워크의 loss가 정확히 0이고, 제대로 학습된 네트워크의 loss와 **같다**. loss만 봐서는 구분이 안 된다.

### 반대 극단: 균등 붕괴

$q = p = \text{uniform}$이면 $D_{KL} = 0$이지만 $h(q) = \log K$라 loss는 $\log K$에서 멈춘다. student는 teacher를 완벽히 따라갔는데 teacher가 아무 정보도 안 주는 상태다. loss가 $\log K$ 근처 평지에 갇힌다.

### 두 방지 장치가 분해의 두 항에 대응한다

```python
teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
#                        └── centering ──┘   └─ sharpening ─┘
```

| 장치 | $h(q)$에 하는 일 | 막는 붕괴 | 혼자 두면 |
|---|---|---|---|
| **centering** (배치 평균 EMA를 뺌) | $h(q)$를 **올린다** — 늘 큰 차원은 center도 커져 상쇄 | 원-핫 붕괴 ($h(q) \to 0$) | 균등 붕괴로 감 |
| **sharpening** ($\tau_t = 0.04 < \tau_s = 0.1$) | $h(q)$를 **내린다** | 균등 붕괴 ($h(q) \to \log K$) | 원-핫 붕괴로 감 |

즉 두 장치는 **$h(q)$를 양쪽에서 붙잡아 중간에 묶어 두는 장치**다. $h(q)$가 고정되면 loss를 줄이는 길은 (a) $D_{KL} \downarrow$ 하나만 남고, 그때 비로소 학습이 "student가 teacher를 따라가기"라는 원래 의도대로 굴러간다.

---

## 6. 한 줄 요약

$$H(q,p) = h(q) + D_{KL}(q\|p), \qquad D_{KL} \ge 0$$

loss가 내려갈 때 **그 감소분이 $D_{KL}$에서 나왔는지 $h(q)$에서 나왔는지**가 학습과 붕괴를 가른다. loss 숫자 하나로는 그걸 알 수 없다 — 그래서 DINO는 teacher 출력의 엔트로피를 따로 로깅하고, k-NN 평가를 별도로 돌린다.
