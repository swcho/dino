# DINO 최종 손실 식 — 안쪽부터 바깥으로 읽기

$$
\mathcal{L} = \frac{1}{|\mathcal{N}|}\sum_{u\in\{1,2\}}\ \sum_{\substack{v=1\\ v\neq u}}^{2+N}
\left(-\sum_{k=1}^{K} P_t^{(u)}(k)\,\log P_s^{(v)}(k)\right),
\qquad |\mathcal{N}| = 2(2+N)-2
$$

$\sum$ 가 세 겹으로 겹쳐 있어서 겁먹기 쉬운데, 사실 **가장 안쪽 괄호 하나만 이해하면 나머지는 "그걸 몇 번 반복해서 평균 내는가"** 일 뿐이다. 안에서 밖으로 한 겹씩 벗겨 보자.

미리 알아 둘 기호:

| 기호 | 뜻 |
|---|---|
| $K$ | 출력 차원 (DINO 기본값 65536). "학생·교사가 고를 수 있는 선택지 개수" |
| $P_t^{(u)}$ | 교사가 view $u$ 를 보고 낸 확률분포. 길이 $K$, 합 = 1 |
| $P_s^{(v)}$ | 학생이 view $v$ 를 보고 낸 확률분포. 길이 $K$, 합 = 1 |
| $N$ | local crop 개수 (기본 8). 전체 view 수는 global 2개 + local $N$개 = $2+N$ |

---

## ① 가장 안쪽: $-\sum_k P_t(k)\log P_s(k)$ — 한 쌍의 벌점

이건 **교차엔트로피(cross-entropy)** 다. 고등학교에서 배운 기댓값 형태로 다시 쓰면 정체가 바로 보인다.

$$
-\sum_{k} P_t(k)\,\log P_s(k) \;=\; \sum_k \underbrace{P_t(k)}_{\text{가중치}} \cdot \underbrace{\big(-\log P_s(k)\big)}_{\text{벌점}}
\;=\; \mathbb{E}_{k\sim P_t}\big[-\log P_s(k)\big]
$$

즉 **"교사가 정답이라 믿는 정도를 가중치로 삼아, 학생이 그 답에 매긴 확률의 $-\log$ 벌점을 평균 낸 값"** 이다.

벌점 $-\log p$ 의 성질을 먼저 감각으로 잡자 (자연로그 기준):

| 학생이 매긴 확률 $p$ | $-\log p$ |
|---|---|
| $1.0$ | $0$ (확신했고 맞음 → 벌점 없음) |
| $0.7$ | $0.357$ |
| $0.1$ | $2.303$ |
| $0.001$ | $6.908$ |
| $\to 0$ | $\to \infty$ (교사가 중요하다는 답에 0을 줬다 → 폭발) |

$p$ 가 작아질수록 벌점이 급격히 커진다. 그래서 이 식은 **교사가 무겁게 보는 $k$ 에서 학생이 확률을 낮게 준 경우를 특히 심하게 때린다.**

### $K=3$ 손계산

선택지가 3개뿐이라고 하자. 교사 분포를 고정한다.

$$
P_t = (0.7,\ 0.2,\ 0.1)
$$

**경우 A — 학생이 교사와 똑같이 답한 경우** $P_s = (0.7,\ 0.2,\ 0.1)$

$$
\begin{aligned}
-\sum_k P_t(k)\log P_s(k)
&= -\big(0.7\ln 0.7 + 0.2\ln 0.2 + 0.1\ln 0.1\big)\\
&= -\big(0.7(-0.357) + 0.2(-1.609) + 0.1(-2.303)\big)\\
&= 0.250 + 0.322 + 0.230 = \mathbf{0.802}
\end{aligned}
$$

**경우 B — 학생이 1등과 3등을 뒤바꾼 경우** $P_s = (0.1,\ 0.2,\ 0.7)$

$$
\begin{aligned}
&= -\big(0.7\ln 0.1 + 0.2\ln 0.2 + 0.1\ln 0.7\big)\\
&= 1.612 + 0.322 + 0.036 = \mathbf{1.969}
\end{aligned}
$$

**경우 C — 학생이 아무것도 모르겠다며 균등하게 답한 경우** $P_s = (\tfrac13,\tfrac13,\tfrac13)$

$$
= -\sum_k P_t(k)\ln\tfrac13 = -\ln\tfrac13 = \mathbf{1.099}
$$

정리하면 $0.802 \;(<)\; 1.099 \;(<)\; 1.969$. **학생이 교사를 그대로 따라할 때 최솟값**이고, 어긋날수록 커진다. 이 최솟값이 왜 0이 아닌지는 ⑥에서 다룬다.

---

## ② 두 번째 겹: $\sum_{v\neq u}$ — 교사 view 하나당 학생 view 몇 개?

DINO는 이미지 한 장을 여러 방식으로 잘라 **view** 를 만든다 (multi-crop).

$$
V = \underbrace{\{x_1^g,\ x_2^g\}}_{\text{global, }224\text{px}} \cup \underbrace{\{x_1^l,\dots,x_N^l\}}_{\text{local, }96\text{px}},
\qquad |V| = 2+N
$$

여기서 **비대칭**이 핵심이다.

- 교사는 **global view만** 본다 → $u \in \{1, 2\}$
- 학생은 **전부** 본다 → $v \in \{1, \dots, 2+N\}$

$\sum_{v\neq u}$ 는 "교사 view $u$ 를 고정해 두고, 학생의 view 를 전부 훑되 **자기 자신($v=u$)만 뺀다**"는 뜻이다. 왜 빼는가? $v=u$ 면 같은 그림을 교사와 학생이 나란히 보는 자명한 항이 되어, 배울 것이 없다.

따라서 $u$ 하나당 항의 개수는

$$
(2+N) - 1
$$

$N=8$ 이면 $10 - 1 = 9$개.

> 이 겹이 만들어 내는 것이 DINO의 표어 **"local-to-global"** 이다. 학생은 96px짜리 작은 조각만 보고서, 224px 전체를 본 교사의 답을 맞혀야 한다. → *"부분을 보고 전체를 추론하라"*.

---

## ③ 세 번째 겹: $\sum_{u\in\{1,2\}}$ — 교사 view 는 2개

global view 가 2장이므로 ②의 묶음을 두 번 돌린다. 그게 전부다.

---

## ④ 항의 개수 세기: $|\mathcal{N}| = 2(2+N)-2$

②와 ③을 곱하면 끝이다.

$$
|\mathcal{N}| = \underbrace{2}_{u\ \text{개수}} \times \underbrace{\big((2+N)-1\big)}_{u\ \text{하나당 } v\ \text{개수}} = 2(2+N) - 2
$$

$N=8$ 이면

$$
|\mathcal{N}| = 2 \times 10 - 2 = \mathbf{18}
$$

18개를 성격별로 나눠 보면 이해가 더 선명하다.

| 쌍의 종류 | 개수 | 내용 |
|---|---|---|
| global 교사 ↔ **다른** global 학생 | $2$ | $(u,v) = (1,2), (2,1)$ |
| global 교사 ↔ local 학생 | $2 \times 8 = 16$ | local-to-global 대응 |
| **제외됨** $(v=u)$ | $2$ | $(1,1), (2,2)$ — 자명해서 뺀 항 |

$2 + 16 = 18$. 그리고 "제외된 2개"가 바로 식의 $-2$ 다.

**이미지 한 장마다 18개의 교차엔트로피 항**이 생긴다는 뜻이다.

---

## ⑤ 왜 $\frac{1}{|\mathcal{N}|}$ 로 나누는가

18개를 더하기만 하고 나누지 않으면, **$N$ 을 바꿀 때마다 손실의 크기 자체가 변한다.**

| $N$ | 항 개수 | 나누지 않으면 손실 크기 |
|---|---|---|
| 0 | 2 | 기준 |
| 8 | 18 | 약 9배 |
| 16 | 34 | 약 17배 |

손실이 9배면 **gradient도 대략 9배**다. 즉 학습률 $\eta$ 를 그대로 두어도 실제 파라미터 갱신폭 $\eta \cdot \nabla \mathcal{L}$ 이 9배로 뛰어 버린다. "local crop 을 8개에서 16개로 늘렸을 뿐인데 학습이 발산" 하는 사고가 여기서 난다.

$|\mathcal{N}|$ 로 나누면 손실이 **항들의 평균**이 되어, $N$ 을 어떻게 바꿔도 값이 $O(1)$ 수준(대략 $\log K$ 근처)에 머문다. → **하이퍼파라미터 $N$ 과 학습률 $\eta$ 가 서로 독립**해진다. 이게 나누는 유일하고 실용적인 이유다.

---

## ⑥ 코드와 수식의 대응

`main_dino.py` 의 `DINOLoss.forward` 는 위 식을 **글자 그대로** 옮긴 이중 루프다.

```python
for iq, q in enumerate(teacher_out):          # 바깥 루프
    for v in range(len(student_out)):         # 안쪽 루프
        if v == iq:
            continue                          # 자기 자신 제외
        loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        total_loss += loss.mean()
        n_loss_terms += 1
total_loss /= n_loss_terms
```

| 코드 | 수식 | 설명 |
|---|---|---|
| `iq` | $u$ | 교사 view 인덱스 (0, 1 → 2개) |
| `v` | $v$ | 학생 view 인덱스 (0 … $1+N$ → $2+N$개) |
| `q` | $P_t^{(u)}$ | 이미 softmax·center·sharpen 을 거친 교사 분포 |
| `F.log_softmax(student_out[v])` | $\log P_s^{(v)}$ | $\log$ 를 softmax 와 **합쳐서** 계산 |
| `torch.sum(-q * ..., dim=-1)` | $-\sum_{k=1}^{K}P_t(k)\log P_s(k)$ | $k$ 축(마지막 축)을 따라 합 — ①의 안쪽 괄호 |
| `if v == iq: continue` | $v \neq u$ | 자명한 쌍 제외 → 최종 $-2$ |
| `loss.mean()` | 배치 평균 | 위 식엔 안 보이는 축. 미니배치 $B$ 장에 대한 $\mathbb{E}_{x\sim\mathcal{D}}$ 근사 |
| `n_loss_terms` | $|\mathcal{N}|$ | 루프가 실제로 센 값 = 18 |
| `total_loss /= n_loss_terms` | $\frac{1}{\lvert\mathcal{N}\rvert}$ | ⑤의 나눗셈 |

두 가지 잔주름:

- **`log_softmax` 를 쓰는 이유**: $\log(\text{softmax}(z))$ 를 따로 계산하면 $p$ 가 $10^{-30}$ 같이 작을 때 부동소수점에서 0이 되고 $\log 0 = -\infty$ 가 된다. `log_softmax` 는 $\log$ 를 지수 안으로 밀어 넣어 이 폭발을 피한다. ①의 표에서 본 "$p\to 0$ 이면 벌점 $\to\infty$" 가 실제 수치 문제로 나타나는 지점이다.
- **`q` 에는 `.detach()` 가 걸려 있다**: 교사 쪽으로는 gradient가 흐르지 않는다. 교사는 오직 $\theta_t \leftarrow m\theta_t + (1-m)\theta_s$ (EMA) 로만 갱신된다. 식에서 $P_t$ 는 "그때그때 주어진 상수 목표"로 읽어야 한다.

---

## ⑦ 이 손실의 최솟값은 0이 아니다

①의 경우 A 를 다시 보자. 학생이 교사를 **완벽히** 따라 했는데도 값이 $0.802$ 였다. 우연이 아니다. 교차엔트로피는 이렇게 분해된다.

$$
\underbrace{-\sum_k P_t(k)\log P_s(k)}_{H(P_t,\,P_s)} \;=\; \underbrace{H(P_t)}_{\text{교사 분포 자체의 엔트로피}} \;+\; \underbrace{D_{\mathrm{KL}}(P_t \,\|\, P_s)}_{\ \ge 0,\ \text{두 view 정렬 정도}}
$$

$D_{\mathrm{KL}} \ge 0$ 이고 $P_s = P_t$ 일 때만 0이므로, **손실이 도달할 수 있는 최솟값은 $0$ 이 아니라 $H(P_t)$** 다. 실제로 경우 A 의 $0.802$ 는 $P_t=(0.7,0.2,0.1)$ 의 엔트로피 $H(P_t)$ 와 정확히 같은 값이다.

**함의 한 줄:** 학생을 아무리 잘 학습시켜도 손실은 $H(P_t)$ 아래로 못 내려가므로, 손실을 더 낮추려는 최적화가 **정렬($D_{\mathrm{KL}}$)을 배우는 대신 $H(P_t)$ 자체를 0으로 눌러 버리는 지름길** — 모든 입력에 같은 one-hot 을 내뱉는 붕괴(collapse) — 로 새는 것이 DINO의 근본 위험이고, 그래서 centering(uniform 쪽으로 밀기)과 sharpening(one-hot 쪽으로 밀기)이 반대 방향으로 균형을 잡아 준다.

---

## 한 줄 요약

$$
\mathcal{L} = \Big(\text{"교사 global view 2개} \times \text{자기 자신 뺀 학생 view }(2+N)-1\text{개"} = 18\text{쌍의 교차엔트로피}\Big)\ \text{의 평균}
$$
