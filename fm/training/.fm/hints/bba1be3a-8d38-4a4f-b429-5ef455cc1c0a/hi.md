# DINO의 learning rate linear scaling rule

## 0. 결론 먼저

$$
\texttt{lr}_{eff} \;=\; 0.0005 \times \frac{\texttt{batch\_size\_per\_gpu} \times \texttt{world\_size}}{256}
$$

`batch_size_per_gpu`는 GPU 한 장이 한 번에 처리하는 이미지 수, `world_size`는 GPU 개수다.
둘을 곱한 값이 **한 step에서 실제로 쓰이는 전체 배치 크기** $B_{total}$이고,
이 값이 256일 때의 학습률을 $0.0005$로 정해 두고 **배치에 정비례**해서 늘린다.

기본 설정인 64/GPU × 8 GPU면 $B_{total} = 512$이므로

$$
\texttt{lr}_{eff} = 0.0005 \times \frac{512}{256} = 0.0005 \times 2 = 0.001
$$

아래는 "왜 하필 정비례인가"를 고등학교 수준에서부터 쌓아 올린 설명이다.

---

## 1. 준비: 경사하강법 한 걸음

신경망 학습은 손실함수 $L(\theta)$를 최소화하는 문제다. $\theta$는 파라미터(가중치)를 전부 모아 놓은 것이라고 보면 되고, 지금은 **변수가 하나인 함수** $L(\theta)$로 생각해도 아이디어는 똑같다.

미적분에서 배운 대로, $L$이 가장 빠르게 **증가**하는 방향은 미분값 $L'(\theta)$ 쪽이다. 그러니 최소화하려면 반대 방향으로 가면 된다:

$$
\theta \;\leftarrow\; \theta - \eta \, g, \qquad g = L'(\theta)
$$

- $g$: gradient(기울기). "어느 쪽으로 얼마나 가파른가"
- $\eta$ (eta): **learning rate**, 즉 **걸음 폭**

$g$가 방향을, $\eta$가 보폭을 정한다. 이 카드의 주제는 오직 $\eta$를 어떻게 정하느냐다.

---

## 2. 미니배치 gradient는 "표본평균"이다

데이터가 $N$장 있을 때, 진짜 손실은 전체 평균이다:

$$
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell_i(\theta)
$$

$N$이 100만 장이면 매 step마다 100만 개를 다 계산할 수 없다. 그래서 무작위로 $B$장만 뽑아 평균을 낸다(= **미니배치**):

$$
\hat{g}_B \;=\; \frac{1}{B}\sum_{i \in \text{배치}} \nabla \ell_i(\theta)
$$

여기서 **확률과 통계**가 등장한다. 무작위 표본 $B$개의 평균이므로:

- **기대값**: $E[\hat{g}_B] = g$ — 배치가 크든 작든 **평균적으로는 같은 방향**을 가리킨다(불편추정량).
- **표준편차**: 개별 $\nabla \ell_i$의 표준편차를 $\sigma$라 하면, 표본평균의 표준편차는

$$
\text{SD}(\hat{g}_B) = \frac{\sigma}{\sqrt{B}}
$$

즉 배치를 4배로 키우면 gradient의 **노이즈는 $1/\sqrt{4} = 1/2$로 줄지만, 방향(기대값)은 그대로**다.

> 요점: 큰 배치는 "더 좋은 방향"을 주는 게 아니라 **"같은 방향을 더 정확하게"** 준다.

---

## 3. 배치를 키우면 걸음 수가 줄어든다

1 epoch = 데이터 $N$장을 한 번씩 다 보는 것. 배치가 $B$면 한 epoch의 step 수는

$$
\text{steps per epoch} = \frac{N}{B}
$$

**배치와 반비례**한다. 배치를 $k$배 키우면 걸음 수는 $1/k$배가 된다.

| 배치 $B$ | $N=100{,}000$일 때 한 epoch step 수 |
|---|---|
| 256 | 391 |
| 512 | 195 |
| 1024 | 97 |

같은 epoch 수를 돌린다면, 큰 배치는 **걸음 수가 적다**. 보폭을 그대로 두면 총 이동 거리가 짧아져 학습이 덜 진행된다.

---

## 4. Linear scaling rule의 직관: "$k$ 걸음의 합 ≈ 큰 걸음 하나"

작은 배치로 $k$번 연속으로 걷는다고 하자(배치 $B$, 학습률 $\eta$):

$$
\theta_{t+k} = \theta_t - \eta\big(\hat g^{(1)} + \hat g^{(2)} + \cdots + \hat g^{(k)}\big)
$$

여기서 **핵심 가정**을 하나 둔다.

> **가정**: $k$ step 동안 $\theta$가 조금밖에 안 움직여서, 그 사이 gradient가 거의 변하지 않는다.
> 즉 $\hat g^{(1)} \approx \hat g^{(2)} \approx \cdots$ — 모두 **같은 지점**에서 뽑은 표본으로 봐도 된다.

이 가정이 맞으면 $k$개의 미니배치를 다 합친 것은 **크기 $kB$짜리 배치 하나**와 같다:

$$
\hat g^{(1)} + \cdots + \hat g^{(k)} \;\approx\; k \cdot \hat g_{kB}
$$

따라서 $k$ 걸음의 총 이동은

$$
\theta_{t+k} - \theta_t \;\approx\; -\,(k\eta)\,\hat g_{kB}
$$

오른쪽은 "배치 $kB$로 **학습률 $k\eta$**를 써서 한 걸음 걸은 것"과 정확히 같은 모양이다.

$$
\boxed{\;\text{배치를 } k \text{배 키우면 학습률도 } k \text{배}\;}
$$

이게 **linear scaling rule**이다(Goyal et al., 2017 "Accurate, Large Minibatch SGD"; 아이디어 자체는 Krizhevsky 2014에도 나온다).

**비유**: 산을 내려가는데 100 걸음 × 1 m로 가든, 10 걸음 × 10 m로 가든 총 100 m를 내려간다. 단, 10 m씩 뛰어도 되는 건 "그 구간 경사가 거의 안 변할 때"뿐이다. 가정이 깨지면 절벽으로 뛰어내리게 된다(→ 6절 warmup).

---

## 5. 비례식으로 정리하기

$\eta \propto B_{total}$이므로, 기준점 하나만 정하면 비례식으로 전부 정해진다. DINO는 기준을 $B=256$, $\eta_{256}=0.0005$로 잡았다:

$$
\frac{\eta_{eff}}{\eta_{256}} = \frac{B_{total}}{256}
\quad\Longrightarrow\quad
\eta_{eff} = \eta_{256} \cdot \frac{B_{total}}{256} = 0.0005 \times \frac{B_{total}}{256}
$$

그리고 분산 학습에서는 GPU 8장이 각각 64장씩 처리한 gradient를 **평균 내서** 하나의 업데이트를 만들므로(all-reduce), 실효 배치는

$$
B_{total} = \texttt{batch\_size\_per\_gpu} \times \texttt{world\_size}
$$

이 둘을 합치면 맨 앞의 공식이 된다. 코드에서는 딱 한 줄이다:

```python
lr_schedule = utils.cosine_scheduler(
    args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
    ...
)
```

`--lr`의 도움말도 같은 말을 한다: *"linear warmup이 끝났을 때의 학습률(학습 중 최댓값). 학습률은 배치 크기에 선형으로 스케일되며, 여기 적는 값은 **기준 배치 256**에 대한 값이다."*

즉 **`--lr 0.0005`는 실제로 쓰이는 학습률이 아니라 "배치 256일 때의 환산 기준값"**이다. 이 구분이 이 카드의 함정이다.

---

## 6. 계산 예시

| 설정 | $\texttt{bs\_per\_gpu} \times \texttt{world\_size}$ | $B_{total}$ | $\eta_{eff} = 0.0005 \cdot B_{total}/256$ |
|---|---|---|---|
| DINO 논문 기본 | $64 \times 8$ | 512 | $0.0005 \times 2 = \mathbf{0.001}$ |
| GPU 4장 축소 | $32 \times 4$ | 128 | $0.0005 \times 0.5 = \mathbf{0.00025}$ |
| 노트북 단일 GPU | $16 \times 1$ | 16 | $0.0005 \times \tfrac{1}{16} = \mathbf{0.00003125}$ |
| 스모크 테스트 | $8 \times 1$ | 8 | $0.0005 \times \tfrac{1}{32} = \mathbf{1.5625\times10^{-5}}$ |

노트북 §8은 첫 줄을 그대로 코드로 확인한다:

```python
lr_base = 0.0005 * (64 * 8) / 256.   # = 0.001
lr_sched = utils.cosine_scheduler(lr_base, 1e-6, EPOCHS, NITER, warmup_epochs=10)
```

여기서 `lr_base`는 스케줄의 **최댓값**(warmup이 끝나는 지점의 값)이지, 시작값이 아니다. 실제 학습률 곡선은
$0 \to \texttt{lr}_{eff} \to 10^{-6}$ (선형 warmup 후 코사인 감소) 모양이다.

> **왜 하필 256인가?** 수학적 필연성은 없다. ImageNet 분류 실험에서 오래 쓰인 관습적 기준 배치일 뿐이다. 기준을 512, $\eta_{512}=0.001$로 바꿔 적어도 결과 곡선은 완전히 동일하다. 논문끼리 하이퍼파라미터를 비교할 때 "배치가 달라서 lr이 다른 것"과 "정말 lr을 다르게 튜닝한 것"을 구분하려고 **공통 눈금**을 하나 정해 둔 것이다.

---

## 7. 큰 배치에서 warmup이 필요한 이유

4절의 가정을 다시 보자: "$k$ step 동안 gradient가 거의 안 변한다."

학습 **초기**에는 이 가정이 심하게 깨진다. 가중치가 무작위 초기화 상태라 파라미터가 조금만 움직여도 gradient가 확 달라진다. 이때 큰 $\eta$로 크게 한 걸음 뛰면 손실이 발산하거나 NaN이 뜬다.

해결책이 **linear warmup**이다. $\eta$를 처음부터 최댓값으로 쓰지 않고, 초반 $T_w$ step 동안 0에서부터 선형으로 끌어올린다:

$$
\eta_t = \frac{t}{T_w}\,\eta_{eff} \qquad (t < T_w)
$$

DINO의 `--warmup_epochs` 기본값은 **10 epoch**. 그 뒤로는 코사인 스케줄로 $10^{-6}$까지 내려간다:

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}\big(\eta_{eff} - \eta_{\min}\big)\Big(1 + \cos\tfrac{\pi(t-T_w)}{T-T_w}\Big)
\qquad (t \ge T_w)
$$

가중치가 어느 정도 자리를 잡은 뒤에는 손실 지형이 완만해져 4절의 가정이 다시 성립하고, 그때는 큰 $\eta$를 감당할 수 있다. **linear scaling rule과 warmup은 한 세트**로 이해해야 한다 — 스케일링이 만든 큰 $\eta$의 위험을 warmup이 막아 주는 구조다.

> 실무 함정: `utils.cosine_scheduler`에는 `assert len(schedule) == epochs * niter_per_ep`가 있어서 `warmup_epochs`(기본 10) $>$ `epochs`면 죽는다. 2~3 epoch짜리 스모크 테스트에는 `--warmup_epochs 0`을 반드시 준다.

---

## 8. 한계 — 언제 깨지는가

이 규칙은 **정리(theorem)가 아니라 근사**다. 근거가 4절의 한 가지 가정뿐이므로 다음에서 무너진다.

1. **아주 큰 배치**: 배치 8k(≈8192)를 넘어가면 선형 스케일링으로 맞춘 $\eta$가 너무 커져서 warmup을 써도 정확도가 떨어진다. Goyal et al.이 직접 보고한 한계다. 노이즈가 이미 $1/\sqrt{B}$로 충분히 작아져 "노이즈가 곧 정규화 효과"였던 부분까지 사라지는 것도 원인으로 본다.
2. **AdamW에서는 근사적으로만**: 이 유도는 순수 SGD 기준이다. DINO의 옵티마이저인 AdamW는 gradient를 그 크기(2차 모멘트의 제곱근)로 나눠 쓰는 **적응형**이라, 유도가 그대로 적용되지 않는다. 그래도 실무에서 잘 작동하는 편이라 관행적으로 함께 쓴다.
3. **대안: 제곱근 스케일링** $\eta \propto \sqrt{B_{total}}$ — gradient 노이즈가 $1/\sqrt{B}$로 줄어드는 것에 맞춰 보폭을 늘리자는 관점이며, Adam 계열과 아주 큰 배치에서 선형보다 안정적이라는 보고가 많다.

---

## 한 줄 요약

배치를 $k$배 키우면 한 epoch의 걸음 수가 $1/k$로 줄어드니, **같은 거리를 가려면 보폭을 $k$배**로 — 그것이 $\eta_{eff} = 0.0005 \times \frac{\texttt{bs\_per\_gpu} \times \texttt{world\_size}}{256}$이고, 64×8 = 512는 기준 256의 2배라 $0.001$이 된다.
