# `q=uniform, p=uniform` — KL은 0인데 loss는 최댓값

> **Q.** `q=uniform, p=uniform`인 경우 KL과 loss 값은 각각 어떻게 되는가?
>
> **A.** KL은 0이지만 loss는 $\log 8 = 2.079$이다. student가 teacher를 완벽히 따라갔지만 teacher가 아무 정보도 주지 않는 상태다.

이 카드는 "**KL이 0인데 왜 loss가 최악인가**"라는 겉보기 모순을 다룬다. 답은 분해식 한 줄에 다 들어 있다.

---

## 1. 분해식이 모순을 없앤다

교차엔트로피는 항상 두 조각으로 쪼개진다 (논문 식 (5)).

$$H(q, p) \;=\; \underbrace{h(q)}_{\text{teacher 분포의 엔트로피}} \;+\; \underbrace{D_{KL}(q \,\|\, p)}_{\text{student가 teacher와 다른 정도}}$$

`q = p = uniform`을 대입해 보자. $K = 8$일 때:

| 항 | 값 | 의미 |
|---|---|---|
| $D_{KL}(q\,\|\,p)$ | $0$ | 두 분포가 **완전히 같다** |
| $h(q)$ | $\log 8 = 2.0794$ | 균등분포 = 엔트로피 **최댓값** |
| $H(q,p)$ = loss | $0 + 2.0794 = 2.0794$ | 전부 $h(q)$에서 온다 |

즉 **KL과 loss는 서로 다른 것을 재는 계기판**이다.

- KL은 "student가 teacher를 얼마나 잘 따라갔는가" → 0점, 즉 **완벽**
- loss는 거기에 "teacher가 애초에 얼마나 정보를 담고 있는가"($h(q)$)를 **더한** 값

student는 만점을 받았는데, 따라간 대상이 "나도 모르겠다, 1/8씩 다 같다"였다. 만점짜리 백지 답안이다.

노트북 2절 표를 직접 돌려 확인한 값:

```
case                                              H(q,p)    h(q)      KL   h(q)+KL
q=peaked,  p=peaked  (완벽히 맞춤)                 0.656   0.656   0.000     0.656
q=peaked,  p=uniform (student가 아무것도 못함)     2.079   0.656   1.423     2.079
q=uniform, p=uniform (둘 다 균등)                 2.079   2.079   0.000     2.079   ← 이 카드
q=one_hot, p=one_hot (둘 다 같은 원-핫)           0.000   0.000   0.000     0.000
```

> **곁다리 관찰 하나.** 2행과 3행의 loss가 **똑같이 2.079**다. $p$가 균등이면
> $H(q,p) = -\sum_i q_i \log \frac{1}{K} = \log K$가 되어 **$q$가 무엇이든 상관없이** loss는 $\log K$다.
> "student가 균등을 내면 teacher가 무슨 말을 하든 점수는 $\log K$"라는 뜻이고,
> 그래서 $\log K$는 학습 곡선에서 "student가 아무것도 안 하고 있다"를 가리키는 기준선 역할을 한다.

---

## 2. 원-핫 붕괴와의 대비 — 그런데 왜 못 빠져나오나

DINO의 붕괴는 두 종류다. **loss 값이 정반대**라는 게 핵심이다.

| | 출력 | $h(q)$ | KL | **loss** | 성격 |
|---|---|---|---|---|---|
| **원-핫 붕괴** | 모든 입력 → 같은 차원 하나 | $\to 0$ | $\to 0$ | **$0$ (만점)** | gradient가 **적극적으로 끌고 간다** |
| **균등 붕괴** | 모든 입력 → $1/K$씩 | $\to \log K$ | $\to 0$ | **$\log K$ (최악)** | gradient가 **밀어낼 힘이 없다** |

원-핫 붕괴가 왜 생기는지는 쉽다. loss가 0이니 최적화가 자발적으로 그쪽으로 달려간다.
헷갈리는 건 균등 붕괴다. **loss가 최댓값인데 왜 거기 머물러 있나?**

### 답: 그 지점에서 gradient가 정확히 0이다

student 로짓 $z$, student 온도 $\tau_s$, $p = \mathrm{softmax}(z/\tau_s)$일 때 교차엔트로피의 로짓 미분은 딱 이 모양이다 ($\sum_i q_i = 1$이므로 softmax 미분이 깨끗하게 정리된다):

$$\frac{\partial H(q,p)}{\partial z} \;=\; \frac{p - q}{\tau_s}$$

teacher가 균등, 즉 $q = \frac{1}{K}\mathbf{1}$이면

$$\frac{\partial H}{\partial z} \;=\; \frac{1}{\tau_s}\Big(p - \tfrac{1}{K}\mathbf{1}\Big), \qquad\text{경사하강 방향}\;\; -\frac{\partial H}{\partial z} = \frac{1}{K\tau_s}\big(\mathbf{1} - K\mathbf{p}\big)$$

student도 균등이면 $K\mathbf{p} = \mathbf{1}$이므로 **gradient가 성분마다 정확히 0**이다. PyTorch로 확인:

```python
K, tau_s = 8, 0.1
q = torch.full((K,), 1/K)
z = torch.zeros(K, requires_grad=True)          # 로짓 전부 0 → p = uniform
L = -(q * F.log_softmax(z/tau_s, -1)).sum()
L.backward()
# L.item()  = 2.0794  (= log 8, 최댓값)
# z.grad    = [0., 0., 0., 0., 0., 0., 0., 0.]  ← 완전히 0
```

로짓 하나만 1로 올려 대칭을 깨면 `z.grad = [8.747, -1.250, ...]`가 되고 이는 $(p-q)/\tau_s$와 소수점까지 일치한다.
즉 **평평한 지점 정확히 그 위에서만 힘이 사라진다**.

### 왜 "고원(plateau)"인가 — 두 겹의 이유

1. **student 쪽**: $q$가 고정된 균등분포일 때 $H(q,p)$는 $p = q$에서 **최소**다. 즉 균등 teacher를 상대로는 균등 student가 이미 최선의 답이고, loss $\log K$는 "더 내려갈 데가 없는 값"이다. loss를 진짜로 더 내리려면 $h(q)$ 자체가 줄어야 하는데 이건 student의 gradient가 건드리는 항이 아니다 ($q$에는 `detach`가 걸려 있다).
2. **teacher 쪽**: teacher는 student의 EMA다. student가 안 움직이면 teacher도 안 움직인다. 서로가 서로에게 "너 맞아"라고만 말하는 **자기일관 고정점(self-consistent fixed point)** 이다.

그래서 균등 붕괴는 "loss가 낮아서 좋은 곳에 갇힌" 게 아니라 **loss가 $\log K$로 최악인 채로, 아무도 밀어주지 않아서 갇힌** 상태다. 전역 최소(원-핫 붕괴의 loss 0)가 저 아래 있는데도 거기로 가는 힘이 국소적으로 존재하지 않는다.

노트북 4절이 이걸 학습 없이 상수 출력만으로 재현한다:

```python
zeros = torch.zeros(2*B, out_dim)                 # 모든 샘플, 모든 차원 로짓 0
loss_fn(zeros, zeros)   # → 2.0794 = log 8,  KL은 0

const = torch.zeros(B, out_dim); const[:, 2] = 50.0
loss_fn(cat, cat)       # → 0.0000            원-핫 붕괴, 만점
```

---

## 3. 실전 진단 — 학습 곡선에서 $\log K$를 찾아라

$\log K$ 값 (python3로 검증):

| $K$ | $\log K$ | 어디 |
|---|---|---|
| 8 | **2.0794** | 이 카드 / 노트북 1~4절 |
| 32 | 3.4657 | 노트북 5절 토이 실험 `out_dim=32` |
| 65536 | **11.0904** | 실제 DINO `--out_dim 65536` |

**진단 팁**: 실제 DINO 학습에서 loss가 **11 근처에서 딱 멈춰** 더 안 내려가면 균등 붕괴를 의심해야 한다.
논문 부록 D도 정확히 이 표현을 쓴다 — teacher 온도가 0.06보다 높으면 "the training loss consistently converges to $\ln(K)$".

반대로 loss가 **0 근처로 뚝 떨어져도** 안심하면 안 된다. 노트북 4절이 보여주듯 원-핫 붕괴한 모델과 건강한 모델의 loss가 **똑같이 0**이다. loss 하나로는 붕괴를 못 잡는다.

| 관측 | 의심 |
|---|---|
| loss $\to \log K$ (65536이면 11.09)에서 정체 | **균등 붕괴** |
| loss $\to 0$인데 k-NN 정확도가 안 오름 | **원-핫 붕괴** |
| loss 중간값 + `h_each` 낮고 `h_mean` 높음 | 건강 |

`eval_knn.py`를 주기적으로 돌리거나, teacher 출력의 `h_each` / `h_mean` / `top_share`를 로깅해야 조기 감지가 된다.

---

## 4. 무엇이 균등 붕괴를 유발하나

$$\texttt{teacher\_out} = \mathrm{softmax}\big((\underbrace{\texttt{teacher\_output} - \texttt{center}}_{\text{centering}}) / \underbrace{\texttt{teacher\_temp}}_{\text{sharpening}}\big)$$

균등 붕괴는 이 중 **sharpening이 빠졌을 때** 온다.

1. **centering만 걸고 sharpening이 없을 때** — teacher 온도 = student 온도 (예: 둘 다 0.1). teacher가 student보다 뾰족해질 이유가 전혀 없다. centering이 한 차원 지배는 계속 눌러 주므로, 남은 방향은 균등뿐이다.
2. **teacher 온도가 너무 높을 때** — 논문 부록 D: $\tau_t > 0.06$이면 붕괴. 그래서 실제 DINO는 0.04 → 0.07 선형 warmup을 쓴다 (0.04부터 시작하면 0.07도 버틴다).

노트북 5절 토이 실험을 직접 재현한 결과 ($K = 32$, $\log 32 = 3.47$):

```
config         loss   h_each   h_mean   top_share   code_acc
none           2.91     2.90     3.01        0.83       0.33
center only    3.37     3.36     3.46        0.33       0.83   ← 균등 붕괴 방향
sharp only     0.22     0.00     1.24        0.50       0.67   ← 원-핫 붕괴 방향
both = DINO    0.82     0.59     2.38        0.17       1.00
```

`center only` 행이 이 카드의 상태 그 자체다. `h_each = 3.36`이 $\log K = 3.47$에 거의 붙었고, **loss도 3.37로 $\log K$ 근처에서 정체**한다. 카드의 $\log 8 = 2.079$가 $K=32$로 바뀐 것뿐이다.

그리고 `sharp only`의 loss(0.22)가 `both`(0.82)보다 **낮다**. 표현 품질은 `both`가 압도적인데(`code_acc` 1.00 vs 0.67) loss는 반대로 말한다 — 2절에서 말한 "loss는 붕괴를 못 잡는다"가 실제 학습 곡선에서도 그대로 나온다.

---

## 5. 논문 Fig. 7과의 대응

논문 5.3절 "Avoiding collapse"가 ImageNet에서 같은 그림을 그린다. Fig. 7은 두 패널이다 — **왼쪽: teacher target entropy $h$**, **오른쪽: teacher-student KL**.

> If one operation is missing, **the KL converges to zero, indicating a collapse.** However, the entropy $h$ converges to different values: **0 with no centering** and **$-\log(1/K)$ with no sharpening**, indicating that both operations induce different form of collapse.

$-\log(1/K) = \log K$다. 즉 논문이 말하는 "no sharpening → $h \to \log K$, KL $\to 0$"이 **이 카드가 묻는 바로 그 상태**다.

| | Fig. 7 왼쪽 ($h$) | Fig. 7 오른쪽 (KL) | loss $= h + KL$ |
|---|---|---|---|
| centering 없음 (sharpening만) | $\to 0$ | $\to 0$ | $\to 0$ — 원-핫 붕괴 |
| **sharpening 없음 (centering만)** | $\to \log K$ | $\to 0$ | $\to \log K$ — **균등 붕괴 (이 카드)** |
| 둘 다 | 중간값에서 안정 | 0으로 안 감 | 중간값 |

**KL이 0으로 수렴한다는 것 자체가 붕괴의 신호**라는 점을 놓치지 말자. KL은 "student가 얼마나 못 따라가는가"이므로 보통은 작을수록 좋아 보이지만, 여기서는 "입력이 뭐든 두 네트워크가 똑같은 상수를 낸다"는 뜻이 된다. 건강한 DINO에서는 student가 local crop까지 포함해 더 어려운 문제를 풀기 때문에 KL이 0으로 붙지 않는다.

---

## 한 줄 정리

`q = p = uniform`에서 **KL = 0**(student는 teacher를 완벽히 복제)이지만 **loss $= h(q) = \log K = 2.079$** ($K=8$)로 최댓값이다.
원-핫 붕괴가 loss 0이라 gradient에 *끌려가는* 붕괴라면, 균등 붕괴는 loss가 최악인데도 **gradient $(p-q)/\tau_s$가 정확히 0이 되어 아무도 밀어내지 못하는** 붕괴다.
centering만 있고 sharpening이 없을 때(또는 $\tau_t > 0.06$) 여기로 간다.

---

### 참고

- 노트북: `.fm/assets/dino_collapse_cross_entropy.py` — 2절(case 표), 4절(자명해 시연), 5절(`center only` 설정)
- 논문: `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md` — 5.3절 + Fig. 7 (식 (5) 분해), 부록 D "Sharpening" ($\tau_t > 0.06$ → loss $\to \ln K$)
- 원본 코드: `main_dino.py:363-416` `DINOLoss`
