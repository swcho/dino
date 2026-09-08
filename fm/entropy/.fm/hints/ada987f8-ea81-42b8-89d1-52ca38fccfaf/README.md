# 교차엔트로피 $H(q,p)$ — 정의와 두 분포의 역할

> **Q.** 교차엔트로피 $H(q,p)$의 정의와 각 분포의 역할은?
> **A.** $H(q,p) = -\sum_i q_i \log p_i$. $q$는 정답 취급하는 target 분포, $p$는 모델의 예측.

---

## 1. 정의

같은 표본공간 $\{1,\dots,K\}$ 위의 두 확률분포 $q=(q_1,\dots,q_K)$, $p=(p_1,\dots,p_K)$에 대해

$$H(q,p) \;=\; -\sum_{i=1}^{K} q_i \log p_i \;=\; \mathbb{E}_{i \sim q}\big[-\log p_i\big]$$

읽는 법이 전부다: **$q$로 뽑고, $p$로 점수를 매긴다.**

| 기호 | 이름 | 역할 | 누가 만드나 |
|---|---|---|---|
| $q$ | target / 정답 분포 | **기댓값을 취하는 분포**. 어느 항에 얼마나 가중치를 줄지 결정 | 데이터 레이블, 혹은 DINO에서는 teacher |
| $p$ | 예측 분포 | **로그 안에 들어가는 분포**. 실제로 벌점을 받는 쪽 | student / 학습 중인 모델 |

노트북(`.fm/assets/dino_collapse_cross_entropy.py`, 2절)의 구현도 정확히 이 한 줄이다.

```python
def cross_entropy(q, p):
    return -(q * p.clamp_min(1e-12).log()).sum(-1)
```

`clamp_min(1e-12)`가 붙은 이유는 아래 3절과 5절에서 다시 나온다 — $p_i = 0$인데 $q_i > 0$이면 $H = \infty$이기 때문이다.

기본 성질 몇 가지:

- $H(q,p) \ge 0$ (확률은 $\le 1$이므로 $-\log p_i \ge 0$).
- 로그의 밑이 2면 단위가 **비트(bit)**, 자연로그면 **내트(nat)**. 노트북과 PyTorch는 모두 자연로그를 쓴다. $\log 8 = 2.079\ \text{nat} = 3\ \text{bit}$.
- $q$에 대해 **선형**, $p$에 대해 **볼록(convex)**. $p$에 대해 볼록이라는 게 최적화가 잘 되는 이유다.

---

## 2. 왜 $-\log p_i$가 "놀람"이고 "코드 길이"인가

교차엔트로피는 $-\log p_i$의 기댓값이므로, 이 한 항의 의미만 잡으면 나머지는 따라온다.

### (a) 놀람(surprisal)

사건 $i$가 일어났을 때 얼마나 놀라야 하는가를 재는 함수 $S(p)$에 요구할 조건:

1. 확률이 높을수록 덜 놀란다 → $S$는 감소함수
2. 확실한 일($p=1$)은 전혀 안 놀랍다 → $S(1) = 0$
3. **독립 사건이 둘 다 일어나면 놀람은 더해진다** → $S(p_1 p_2) = S(p_1) + S(p_2)$

곱을 합으로 바꾸는 연속함수는 로그뿐이고, 감소해야 하니 부호가 붙는다:

$$S(p) = -\log p$$

동전을 던져 앞면($p=1/2$)이 나온 놀람은 $\log 2 = 1$비트, 공정한 주사위에서 3이 나온 놀람은 $\log 6 \approx 2.58$비트. $p \to 0$인 사건이 실제로 일어나면 놀람은 $\infty$로 발산한다 — 이것이 "확률 0을 준 곳에 정답이 있으면 손실이 무한대"의 정체다.

### (b) 코드 길이 (Shannon–Kraft)

$K$개 기호를 이진 접두부호(prefix code)로 보낼 때, 길이 $\ell_i$인 코드워드들이 존재할 필요충분조건은 **Kraft 부등식** $\sum_i 2^{-\ell_i} \le 1$이다. 그러니 $\ell_i = \log_2 (1/p_i) = -\log_2 p_i$로 두면 정확히 등호가 성립한다. 즉

> **분포 $p$를 믿으면, 기호 $i$에 $-\log_2 p_i$ 비트짜리 코드를 배정하는 것이 최적이다.**

그런데 실제로 기호가 나오는 빈도는 $q$다. 그러면 기호 하나당 평균 전송 길이는

$$\sum_i q_i \cdot \underbrace{(-\log_2 p_i)}_{p\text{를 믿고 만든 코드 길이}} \;=\; H(q,p)$$

**$H(q,p)$ = "틀린 확률표 $p$로 만든 코드북을, 진짜 빈도 $q$인 데이터에 썼을 때의 평균 코드 길이"** 다. 여기서 $q$가 기댓값을 취하는 쪽, $p$가 코드북을 만드는 쪽이라는 비대칭이 구조적으로 드러난다.

---

## 3. $q$와 $p$를 바꾸면 무엇이 달라지나 — 비대칭성

$$\boxed{H(q,p) \ne H(p,q)}$$

정의만 봐도 두 분포가 하는 일이 다르다.

- **$q$는 "어디를 보는가"를 정한다.** $q_i = 0$인 항은 $p_i$가 무엇이든 손실에 기여하지 않는다. 모델은 $q$가 무게를 실은 곳에서만 채점받는다.
- **$p$는 "얼마나 벌점을 받는가"를 정한다.** 그리고 $-\log$는 0 근처에서 폭발하므로, **$q$가 무게를 둔 곳에 $p$가 0을 주는 것**이 치명적이다. 반대로 $q$가 0인 곳에 $p$가 확률을 낭비하는 것은 직접적인 벌점이 없다(정규화 $\sum p_i = 1$을 통해 간접적으로만 손해).

### 수치로 보기 ($K=8$, 노트북과 같은 분포)

노트북에서 쓰는 세 분포:

| 이름 | 값 | $h(\cdot)$ |
|---|---|---|
| `one_hot` | 인덱스 3에 1, 나머지 0 | $0$ |
| `peaked` | $\mathrm{softmax}(4,1,0,\dots,0) = (0.862,\,0.043,\,0.016 \times 6)$ | $0.656$ |
| `uniform` | $1/8$ 씩 | $\log 8 = 2.079$ |

역할을 뒤집어 보면:

| 방향 | $H$ | 해석 |
|---|---|---|
| $H(\texttt{peaked},\ \texttt{uniform})$ | $2.079$ | 정답은 뾰족한데 모델이 "모르겠다"고 함. 벌점은 딱 $\log 8$ (모든 항에 $-\log \tfrac18$이므로) |
| $H(\texttt{uniform},\ \texttt{peaked})$ | $3.523$ | 정답이 퍼져 있는데 모델이 한 곳을 확신함. 모델이 버린 6개 칸에서 큰 벌점을 받아 **더 크다** |
| $H(\texttt{one\_hot},\ \texttt{peaked})$ | $4.148$ | $=-\log 0.0158$. 정답 칸 하나만 본다 |
| $H(\texttt{peaked},\ \texttt{one\_hot})$ | $\infty$ (구현상 $27.195$) | `peaked`가 확률을 준 칸에 `one_hot`이 0을 줌 → 발산. `clamp_min(1e-12)` 덕에 $\log 10^{-12} = -27.63$으로 잘려 유한하게 나올 뿐 |

마지막 두 줄이 비대칭성의 극단이다. $4.148$ 대 $\infty$. **"어느 쪽을 정답으로 두느냐"는 편의상의 명명이 아니라 손실의 성질을 완전히 바꾼다.**

이 비대칭의 실전적 이름:

- **$H(q,p)$를 $p$에 대해 최소화 (forward, mode-covering / zero-avoiding)**: $p$는 $q$가 무게를 둔 곳을 **전부** 덮어야 한다. 안 덮으면 $\infty$. 그래서 $p$가 $q$보다 퍼지는 경향.
- **$H(p,q)$ 꼴로 최소화 (reverse, mode-seeking / zero-forcing)**: $p$는 $q$가 0인 곳을 피하면 되고, $q$의 봉우리 하나에 숨어도 된다. VI에서 자주 나오는 그 성질.

지도학습과 DINO는 모두 **forward** 쪽, 즉 $H(q,p)$를 $p$에 대해 최소화한다.

---

## 4. 지도학습: 원-핫 $q$에서 $-\log p_{\text{정답}}$로 줄어드는 과정

정답 레이블이 클래스 $c$ 하나로 주어지면 $q$는 원-핫이다: $q_i = [\,i = c\,]$.

$$H(q,p) = -\sum_{i=1}^K q_i \log p_i = -\big(0\cdot\log p_1 + \cdots + \underbrace{1}_{i=c}\cdot\log p_c + \cdots\big) = -\log p_c$$

**합에서 항이 하나만 살아남는다.** 이것이 흔히 말하는 negative log-likelihood(NLL)이고, PyTorch `F.cross_entropy(logits, labels)`가 계산하는 값이다. 위 표의 3행 $H(\texttt{one\_hot}, \texttt{peaked}) = 4.148 = -\log 0.0158$이 바로 그 예다.

여기서 중요한 부수적 사실 두 가지:

1. **원-핫이면 $h(q) = 0$ 이라 손실 = KL이다.** $H(q,p) = 0 + D_{KL}(q\|p)$. 그래서 지도학습에서 "교차엔트로피를 0으로 만든다"는 말이 자연스럽게 통한다. 실제로 도달 가능한 최솟값이 0이다.
2. **$h(q)$는 데이터가 정하는 상수라 학습이 건드릴 수 없다.** 레이블 스무딩이나 지식 증류처럼 $q$가 soft해지면 $h(q) > 0$이 되고, 손실의 하한이 $h(q)$로 올라간다. "손실이 0으로 안 내려가네?"의 흔한 원인이다.

softmax 로짓 $z$에 대한 기울기가 깔끔하다는 것도 여기서 나온다:

$$\frac{\partial H(q,\ \mathrm{softmax}(z))}{\partial z_i} = p_i - q_i$$

"예측 빼기 정답". $q$는 상수처럼 들어가고 $p$만 밀린다 — $q$가 target, $p$가 예측이라는 역할이 기울기에서도 그대로 보인다.

---

## 5. DINO: $q$가 teacher의 softmax 출력이 될 때

DINO에는 레이블이 없다. 그래서 **teacher 네트워크의 softmax 출력이 $q$ 자리를 대신한다.** 노트북 3절의 `MiniDINOLoss.forward` (원본 `main_dino.py:380` `DINOLoss.forward`와 줄 단위로 대응):

```python
teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)
...
loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
```

마지막 줄이 정확히 $-\sum_i q_i \log p_i$이고, $p = \mathrm{softmax}(z^{\text{student}}/0.1)$, $q = \mathrm{softmax}\big((z^{\text{teacher}} - \text{center})/0.04\big)$이다. 지도학습과 달라지는 점:

### (a) $q$가 soft target이다

원-핫이 아니므로 합이 한 항으로 줄지 않는다. $K = 65536$($\texttt{--out\_dim}$)개 항이 전부 살아 있고, teacher가 "2번 프로토타입 0.7, 17번 0.2, 나머지 조금씩"처럼 **순위 정보(dark knowledge)** 를 통째로 전달한다. 지식 증류와 같은 구조다.

### (b) $h(q)$가 더 이상 상수가 아니다 — 이게 붕괴의 통로다

지도학습에서 $h(q)$는 데이터가 준 상수였다. DINO에서 $q$는 **학습 중인 네트워크가 만들어낸 것**이다. teacher는 student의 EMA이므로($\texttt{main\_dino.py:346}$), student가 움직이면 다음 스텝의 $q$도 움직인다. 즉 손실을 낮추는 경로가 두 개다:

$$H(q,p) = \underbrace{h(q)}_{\text{teacher 분포의 엔트로피}} + \underbrace{D_{KL}(q\|p)}_{\text{student가 teacher와 다른 정도}}$$

- **(a) $D_{KL} \downarrow$**: student가 teacher를 따라간다 — 우리가 원하는 것.
- **(b) $h(q) \downarrow$**: teacher 분포 자체가 뾰족해진다 — **치트**.

`detach()`가 걸려 있어 한 스텝 안에서는 $q$로 기울기가 흐르지 않지만, EMA를 통해 **스텝 사이에** (b)의 경로가 열려 있다. 노트북 4절의 시연이 결말을 보여준다: 모든 입력에 대해 같은 원-핫을 내면 $h(q) = 0$, $D_{KL} = 0$ → **loss = 0**. 입력을 전혀 보지 않고 만점이다.

```
one-hot collapse : loss = 0.0000   ← 0. 완벽한 점수
uniform collapse : loss = 2.0794   ← log 8, KL은 0
healthy one-hot  : loss = 0.0000   ← 이것도 0
```

붕괴한 모델과 건강한 모델의 loss가 **같다.** 교차엔트로피는 "$q$와 $p$가 서로 맞는가"만 볼 뿐 "$q$가 입력에 따라 달라지는가"는 보지 않기 때문이다. DINO의 centering(원-핫 붕괴 방지)과 sharpening(균등 붕괴 방지)은 이 두 방향을 각각 막는 장치이고, 실전에서 `eval_knn.py`를 따로 돌리는 이유도 여기 있다.

---

## 6. 노트북 5-case 표 — 숫자로 읽는 정리

노트북 2절의 출력 (자연로그, $K = 8$):

| case | $H(q,p)$ | $h(q)$ | $D_{KL}(q\Vert p)$ | 읽는 법 |
|---|---:|---:|---:|---|
| `q=peaked, p=peaked` (완벽히 맞춤) | $0.656$ | $0.656$ | $0.000$ | $p=q$라 KL은 0. 그런데 **loss는 0이 아니다** — 바닥이 $h(q)$이기 때문 |
| `q=peaked, p=uniform` (student가 아무것도 못함) | $2.079$ | $0.656$ | $1.423$ | 차이 $1.423$이 전부 student 탓 |
| `q=one_hot, p=peaked` (지도학습 상황) | $4.148$ | $0.000$ | $4.148$ | $h(q)=0$이라 loss = KL = $-\log p_c$ |
| `q=uniform, p=uniform` (둘 다 균등) | $2.079$ | $2.079$ | $0.000$ | **균등 붕괴.** student는 teacher를 완벽히 따라갔는데 teacher가 아무 말도 안 함 |
| `q=one_hot, p=one_hot` (둘 다 같은 원-핫) | $0.000$ | $0.000$ | $0.000$ | **원-핫 붕괴.** 진짜 0점. 입력 무관하게 이 한 쌍만 내면 됨 |

이 표에서 배울 것:

- 1행과 4행: **$D_{KL} = 0$인데 $H$가 다르다.** 차이는 전부 $h(q)$. "student가 teacher를 잘 따라간다"는 것과 "loss가 낮다"는 별개다.
- 4행과 5행: 둘 다 $D_{KL}=0$인 완전 합의 상태인데 loss는 $2.079$와 $0$. **손실이 시키는 방향은 5행이다** — 그래서 아무 장치 없는 DINO는 원-핫 붕괴로 간다.
- 3행: 지도학습은 $h(q)=0$인 특수 케이스일 뿐이라는 것.

---

## 7. 분해 $H(q,p) = h(q) + D_{KL}(q\|p)$

한 줄 유도. $\log p_i = \log q_i - \log\frac{q_i}{p_i}$를 넣으면

$$H(q,p) = -\sum_i q_i \log p_i = \underbrace{-\sum_i q_i \log q_i}_{h(q)} + \underbrace{\sum_i q_i \log \frac{q_i}{p_i}}_{D_{KL}(q\|p)}$$

- $h(q)$: **$q$만의 함수.** $p$가 무엇이든 바뀌지 않는 바닥.
- $D_{KL}(q\|p) \ge 0$, 등호는 $p = q$일 때만 (깁스 부등식).

따라서

$$H(q,p) \ \ge\ h(q), \qquad \text{등호} \iff p = q$$

**$p$에 대한 최적화만 놓고 보면 $H(q,p)$와 $D_{KL}(q\|p)$는 상수 차이라 완전히 같은 문제다.** 지도학습에서 그 상수가 0이라 구분할 필요가 없었고, DINO에서는 그 상수가 학습 가능한 양이라 구분이 결정적이다. 카드 답의 "$q$는 정답 취급하는 target"이라는 문구를 "$q$는 손실의 바닥 $h(q)$까지 결정하는 분포"로 확장해 기억하면 좋다.

(깁스 부등식의 고교 수준 증명은 같은 폴더 `hi.md` 참고.)

---

## 8. 정리 / 자주 하는 실수

- 정의: $H(q,p) = -\sum_i q_i \log p_i = \mathbb{E}_{i\sim q}[-\log p_i]$.
- $q$ = target(기댓값을 취하는 쪽), $p$ = 예측(로그 안에 들어가 벌점을 받는 쪽). **인자 순서가 곧 역할**이고 $H(q,p) \ne H(p,q)$.
- $-\log p_i$ = 놀람 = $p$를 믿고 만든 최적 코드의 길이. $H(q,p)$ = 틀린 코드북으로 잰 평균 길이.
- 원-핫 $q$ → $-\log p_{\text{정답}}$ 하나로 축소, $h(q)=0$이라 loss = KL.
- soft $q$ → $h(q) > 0$이라 **loss는 0으로 안 내려간다**. 하한이 $h(q)$.
- DINO는 $q$ = teacher softmax이므로 $h(q)$가 학습 대상이 되어 버리고, $h(q) \to 0$ 경로가 원-핫 붕괴다.

흔한 실수:

1. **인자 순서 착각.** `F.cross_entropy(input, target)`는 첫 인자가 로짓($p$ 쪽), 둘째가 레이블($q$ 쪽)이다. 수식 표기 $H(q,p)$와 순서가 반대다.
2. **$p$에 log를 두 번 씌우기.** `F.log_softmax`의 출력에 다시 `.log()`를 하는 실수. 노트북/원본 모두 `-q * F.log_softmax(...)` 형태로 안전하게 쓴다 (`log(softmax)`를 따로 계산하면 수치가 터진다).
3. **$p_i = 0$ 방치.** $q_i > 0$인데 $p_i = 0$이면 $\infty$. `clamp_min(1e-12)`나 log-softmax로 막는다.
4. **soft target인데 loss가 0이 아니라고 걱정하기.** 하한이 $h(q)$이므로 정상이다. 비교하려면 $H - h(q) = D_{KL}$을 보라.
5. **loss로 붕괴를 판단하기.** 노트북 4-5절의 핵심: 붕괴한 모델의 loss가 건강한 모델보다 **낮을 수 있다**(`sharp only` 0.2 vs `both` 0.8). `h_each`/`h_mean`/`top_share`를 따로 로깅해야 한다.

---

### 참고

- 소스 노트북: `/home/sungwoo/projects/swcho/dino/fm/entropy/.fm/assets/dino_collapse_cross_entropy.py` (1절 엔트로피, 2절 교차엔트로피 · 5-case 표, 3절 `MiniDINOLoss`, 4절 붕괴 시연)
- 원본 구현: `main_dino.py:363-416` `DINOLoss`, `main_dino.py:346` teacher EMA
- 논문: Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294 — Eq. (1), (2) 및 Fig. 7 (centering/sharpening과 엔트로피)
