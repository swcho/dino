# DINO의 weight decay 스케줄이 특이한 점

> **Q.** DINO의 weight decay 스케줄이 특이한 점은?
>
> **A.** $0.04 \to 0.4$ 로 학습이 진행될수록 **증가**한다. 일반적인 스케줄과 반대이며,
> 초기엔 자유롭게 탐색하고 후반에 강하게 정규화해 표현을 압축하려는 의도다.

---

## 1. 코드에서 확인

`main_dino.py` 는 lr·wd·momentum·teacher temp 네 개를 **학습 전에 전체 iteration 길이의 배열로 미리 만들어 둔다**.

```python
# main_dino.py:244
wd_schedule = utils.cosine_scheduler(
    args.weight_decay,        # 0.04   ← 시작
    args.weight_decay_end,    # 0.4    ← 끝
    args.epochs, len(data_loader),
)
```

인자 help 문구가 의도를 그대로 말한다:

> `--weight_decay_end` : *"Final value of the weight decay. We use a cosine schedule for WD and
> **using a larger decay by the end of training improves performance for ViTs**."*

`utils.cosine_scheduler` 는 warmup(선형) + 코사인 두 구간을 이어 붙인다:

$$
v_t =
\begin{cases}
\dfrac{t}{T_w}\, v_{\text{base}} & t < T_w \quad \text{(linear warmup)}\\[8pt]
v_{\text{final}} + \dfrac{1}{2}\big(v_{\text{base}} - v_{\text{final}}\big)
\Big(1 + \cos\dfrac{\pi (t - T_w)}{T - T_w}\Big) & t \ge T_w
\end{cases}
$$

식 자체는 lr과 **똑같다**. 다른 건 넣는 값의 방향뿐이다.

| | `base_value` | `final_value` | 결과 |
|---|---|---|---|
| lr | $10^{-3}$ (linear scaling 후) | $10^{-6}$ | 코사인 **감소** ↘ |
| wd | $0.04$ | $0.4$ | 코사인 **증가** ↗ |

$v_{\text{base}} < v_{\text{final}}$ 이면 $\cos$ 항의 계수 $\tfrac12(v_{\text{base}}-v_{\text{final}})$ 가 음수가 되어
같은 함수가 그대로 뒤집힌 코사인 램프가 된다. wd에는 warmup을 주지 않으므로($T_w=0$) 첫 스텝부터 $0.04$ 다.

주입은 `train_one_epoch` 안에서:

```python
# main_dino.py:308~312
for i, param_group in enumerate(optimizer.param_groups):
    param_group["lr"] = lr_schedule[it]
    if i == 0:                       # 첫 번째 그룹만 정규화
        param_group["weight_decay"] = wd_schedule[it]
```

스케줄러에 내부 상태가 없고 전역 iteration `it` 로 조회만 하므로 **resume이 자동으로 정확**하다는
부수 효과도 있다.

---

## 2. 배경 — AdamW의 decoupled weight decay가 실제로 하는 일

`torch.optim.AdamW` 에서 weight decay는 gradient에 $\lambda\theta$ 를 더하는 L2 페널티가 **아니다**.
Adam 업데이트와 분리(decoupled)되어, 매 step 파라미터를 곧바로 0쪽으로 수축시킨다
(Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019):

$$
\theta_t \;\leftarrow\; \underbrace{(1 - \eta_t \lambda_t)\,\theta_{t-1}}_{\text{수축}} \;-\; \eta_t \cdot \underbrace{\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}}_{\text{Adam 방향}}
$$

즉 gradient와 무관하게 매 스텝 파라미터에 $(1-\eta_t\lambda_t)$ 가 곱해진다.

여기서 중요한 사실 두 개:

1. **실효 수축률은 $\lambda$ 단독이 아니라 곱 $\eta_t\lambda_t$ 다.** wd만 보고 "정규화가 10배 세졌다"고 읽으면 틀린다.
2. Adam 계열은 $\hat m/\sqrt{\hat v}$ 로 gradient 크기를 정규화해 버리므로, L2를 gradient에 섞으면
   큰 gradient를 가진 파라미터일수록 정규화가 약해지는 왜곡이 생긴다. decoupled 방식은 이 결합을 끊어
   $\lambda$ 를 "순수한 수축 노브"로 만든다. DINO가 $\lambda$ 를 시간에 따라 크게 움직일 수 있는 것도
   이 분리 덕분이다.

---

## 3. 무엇이 "특이"한가 — 일반 관행과의 대비

| | 관행 | DINO |
|---|---|---|
| supervised CNN (ResNet) | $\lambda = 10^{-4}$ **고정** | — |
| 일반 ViT 학습 (DeiT 등) | $\lambda = 0.05$ **고정** | — |
| "정규화 스케줄"을 준다면 | 보통 **감소**시켜 후반 fitting 허용 | — |
| DINO | — | $0.04 \to 0.4$ **코사인 증가 (10배)** |

정규화 강도를 후반에 **올리는** 스케줄은 흔치 않다. 보통은 반대 직관이 지배적이다 —
"초기엔 정규화로 안정화하고, 후반엔 풀어서 데이터를 더 잘 맞춘다."
DINO는 그 순서를 뒤집는다. 그리고 논문이 제시하는 근거는 이론이 아니라 **경험적**이다:
ViT에서 후반의 큰 wd가 downstream(k-NN / linear probe) 성능을 올린다는 것.

---

## 4. 직관 — 왜 뒤집는가

DINO는 레이블 없는 self-distillation이다. 학습 목표는 "라벨을 맞히기"가 아니라
"**쓸모 있는 표현을 만들되 붕괴(collapse)하지 않기**"다. 그래서 두 국면의 요구가 다르다.

**초기 (wd 작음, $\lambda=0.04$)**

- 아직 어떤 프로토타입이 무엇을 뜻하는지 정해지지 않았다. teacher도 EMA로 갓 출발한 상태라 타겟이 계속 움직인다.
- 이때 파라미터 노름을 세게 눌러 두면 head가 필요한 방향으로 뻗어 나가지 못한다.
  탐색 공간을 좁혀 놓고 시작하는 셈.
- lr warmup(10 epoch)이 진행되는 구간과 겹친다 — **탐색이 우선인 국면**.

**후반 (wd 큼, $\lambda=0.4$)**

- teacher momentum $m \to 1.0$ 으로 타겟이 거의 얼어붙고, lr도 $10^{-6}$ 로 내려간다.
  구조는 이미 정해졌고 남은 일은 **다듬기**다.
- 강한 수축은 (a) 쓰이지 않는 방향의 가중치를 0으로 밀어 표현을 **압축**하고,
  (b) 파라미터 노름 폭주를 막고, (c) student가 teacher의 미세한 노이즈까지 외우는 **과적합**을 억제한다.
- 특히 SSL에는 **검증셋이 없다**(pretraining loop에 val이 없어 조기 종료도 best 선택도 불가).
  후반의 강한 정규화는 "언제 멈춰도 무너져 있지 않게" 만드는 안전장치 역할도 한다.

한 줄 요약: **탐색은 초기에, 압축은 후반에.** 정보병목(rate–distortion) 관점으로 읽으면,
초반엔 distortion을 줄이고 후반엔 rate를 줄이는 스케줄이다.

---

## 5. 함정 — 실효 수축 $\eta_t\lambda_t$ 는 중간에 최대다

wd 곡선만 보면 "후반에 정규화가 가장 세다"로 읽힌다. 하지만 AdamW의 실제 수축률은 $\eta_t\lambda_t$ 다.
lr은 내려가고 wd는 올라가므로, **둘의 곱은 중간에 봉우리를 만든다.**

ImageNet 기본 설정($100$ epoch, batch $512$ → linear scaling으로 $\text{lr}_{\text{eff}}=0.0005\times\frac{512}{256}=10^{-3}$, warmup 10 epoch)에서:

| epoch | $\eta_t$ (lr) | $\lambda_t$ (wd) | $\eta_t\lambda_t$ (step당 수축) |
|---:|---:|---:|---:|
| 0 | $0$ | 0.040 | $0$ |
| 5 | $5.0\times10^{-4}$ | 0.042 | $2.1\times10^{-5}$ |
| **10** (warmup 끝, lr 최대) | $1.0\times10^{-3}$ | 0.049 | $4.9\times10^{-5}$ |
| 20 | $9.7\times10^{-4}$ | 0.074 | $7.2\times10^{-5}$ |
| 30 | $8.8\times10^{-4}$ | 0.114 | $1.0\times10^{-4}$ |
| 40 | $7.5\times10^{-4}$ | 0.164 | $1.23\times10^{-4}$ |
| **≈48** | $6.1\times10^{-4}$ | 0.209 | $\mathbf{1.30\times10^{-4}}$ ← **최대** |
| 50 | $5.9\times10^{-4}$ | 0.220 | $1.29\times10^{-4}$ |
| 60 | $4.1\times10^{-4}$ | 0.276 | $1.14\times10^{-4}$ |
| 70 | $2.5\times10^{-4}$ | 0.326 | $8.2\times10^{-5}$ |
| 80 | $1.2\times10^{-4}$ | 0.366 | $4.3\times10^{-5}$ |
| 90 | $3.1\times10^{-5}$ | 0.391 | $1.2\times10^{-5}$ |
| 99 | $1.3\times10^{-6}$ | 0.400 | $5.2\times10^{-7}$ |

읽는 법:

- **실제 수축의 피크는 학습 중반(≈ epoch 48)** 이고, 마지막 10 epoch은 lr이 사라지면서 수축도 사실상 멈춘다.
- 그래도 wd 증가는 의미가 있다. wd가 $0.04$ 고정이었다면 $\eta_t\lambda_t$ 는 warmup 끝($4.0\times10^{-5}$)에서
  바로 단조 감소해 버린다. 증가 스케줄은 lr 감쇠를 **부분적으로 상쇄**해,
  수축량을 학습 중후반까지 끌고 가면서 총 수축량도 늘린다.
- 즉 "후반에 정규화를 폭발시킨다"기보다 **"lr이 죽는 동안에도 정규화가 죽지 않게 붙잡는다"** 에 가깝다.

---

## 6. wd가 안 걸리는 파라미터 — `get_params_groups`

증가하는 wd는 **모든** 파라미터에 걸리지 않는다. `utils.get_params_groups` 가 둘로 쪼갠다.

```python
def get_params_groups(model):
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized},
            {'params': not_regularized, 'weight_decay': 0.}]
```

- **그룹 0** — 2차원 이상 텐서(Linear/Conv 가중치, patch embed, `last_layer.weight_v` 등). 여기에만 `wd_schedule[it]` 이 주입된다.
- **그룹 1** — bias와 1차원 파라미터(LayerNorm의 $\gamma,\beta$, cls token, pos embed). `weight_decay: 0.` 이 **한 번** 박히고 이후 아무도 덮어쓰지 않는다.

`train_one_epoch` 의 `if i == 0:` 가드가 정확히 이 대응이다. 그룹 1까지 wd를 넣어 버리면
LayerNorm scale이 0으로 눌리고 pos embed가 사라져 학습이 망가진다.

로깅도 그룹 0만 본다: `metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])`.

---

## 7. (선택) 스케일 불변 파라미터와 실효 lr

ViT 블록의 가중치는 대부분 LayerNorm 앞에 있어 **스케일 불변**이다 — $W \to cW$ 해도 출력이 같다.
이런 파라미터에서는 gradient가 노름에 반비례($\|\nabla_W\| \propto 1/\|W\|$)하므로,
정작 의미 있는 양은 파라미터 값이 아니라 **한 스텝에서 방향이 얼마나 도는가**(각도 업데이트)다.

weight decay가 $\|W\|$ 를 줄이면 같은 lr로도 각도 변화가 커진다 — 즉 **wd가 실효 lr을 올리는 방향으로 작용**한다.
AdamW에서는 이 두 힘이 균형을 이루는 정상상태(rotational equilibrium)가 있고,
스텝당 회전 각도가 대략 $\sqrt{\eta\lambda}$ 스케일로 결정된다는 분석이 있다
(Kosson et al., *Rotational Equilibrium*, ICML 2024).

이 관점에서 보면 DINO의 wd 증가는 순수한 "정규화 강화"가 아니라 **lr 감쇠에 대한 부분적 보상**이기도 하다.
lr이 $10^{-3}\to10^{-6}$ 로 1000배 줄 때 wd가 10배 늘면 $\sqrt{\eta\lambda}$ 는 $\sqrt{1000/10}\approx10$ 배만 줄어,
후반에도 표현이 완전히 얼어붙지 않고 계속 미세 조정된다. (해석 관점이며, 논문의 주장은 아니다.)

---

## 8. 실전

- **고정 wd로 되돌리려면**: `--weight_decay_end 0.04` 로 시작값과 같게 준다.
  `cosine_scheduler` 는 $v_{\text{base}}=v_{\text{final}}$ 일 때 상수 배열을 내므로 자연스럽게 고정 wd가 된다.
- **작은 데이터셋 / 짧은 스케줄**: $0.4$ 는 ImageNet 1.28M 장 × 100~800 epoch 기준으로 튜닝된 값이다.
  수천 장 규모나 수 epoch짜리 스모크 테스트에서 그대로 쓰면 과한 수축이 될 수 있다.
  $0.04 \to 0.1$ 정도로 낮추거나 아예 고정 wd로 두는 편이 안전하다.
- **wd만 따로 튜닝하지 말 것**: 실효량이 $\eta_t\lambda_t$ 이므로 lr(그리고 batch size에 따른 linear scaling)을
  바꿨다면 wd도 함께 다시 봐야 한다.
- **모니터링**: 사전학습에는 검증셋이 없다. loss 곡선만으로는 붕괴를 못 잡으므로,
  wd 튜닝의 효과는 주기적인 k-NN 평가(`eval_knn.py`)로 확인해야 한다.
- **관련 실패 모드**: wd가 과하면 head 출력이 한 프로토타입으로 몰리는 붕괴 쪽으로 밀 수 있다.
  DINO에서 붕괴를 막는 주 장치는 centering + sharpening이지, wd가 아니다 — 역할을 혼동하지 말 것.

---

## 9. 한 줄 정리

> DINO는 lr·momentum·teacher temp와 같은 `cosine_scheduler` 를 쓰되 wd만 시작값 $<$ 끝값으로 넣어
> $0.04 \to 0.4$ **증가** 램프를 만든다. 초기엔 표현 탐색의 자유를, 후반엔 강한 수축으로 압축과 일반화를 얻는 설계이고,
> 근거는 "ViT에서 후반 큰 wd가 성능을 올린다"는 경험적 관찰이다.
> 단, AdamW의 실제 수축은 $\eta_t\lambda_t$ 이므로 **정규화 피크는 후반이 아니라 학습 중반**이며,
> bias·Norm 파라미터는 별도 param group으로 아예 제외된다.

---

### 참고

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021 — Sec. 4 / implementation details
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019 — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Kosson et al., *Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks*, ICML 2024 — [arXiv:2305.17212](https://arxiv.org/abs/2305.17212)
- 코드: [`main_dino.py:244`](../../../main_dino.py) (`wd_schedule`), [`main_dino.py:308`](../../../main_dino.py) (주입),
  [`utils.py:197`](../../../utils.py) (`cosine_scheduler`), `utils.get_params_groups`
- 노트북: `.fm/assets/dino_training_walkthrough.py` §8 「스케줄 4종」, §10 「학습 1 iteration 완전 해부」
