# `momentum_teacher`를 너무 작게 주면?

> **한 줄 답**: 교사가 학생을 거의 즉시 따라가므로 타겟이 매 스텝 요동치고, 결국 학생이 자기 자신을 예측하는 자기참조 루프가 되어 **붕괴(collapse)** 한다. DINO는 기본값 `0.996`에서 코사인 스케줄로 `1.0`까지 **올려서** 타겟을 점점 얼린다.

---

## 1. 이 파라미터가 정확히 무엇인가

DINO의 교사는 gradient를 받지 않는다. 오직 학생 파라미터의 **지수이동평균(EMA)** 으로만 갱신된다.

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s,
\qquad m: 0.996 \nearrow 1.0
$$

`main_dino.py`의 실제 코드는 이게 전부다.

```python
# main_dino.py:250 — 학습 시작 전에 iteration 길이 배열을 통째로 만든다
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                           args.epochs, len(data_loader))

# main_dino.py:347-350 — train_one_epoch 안, optimizer.step() 직후
with torch.no_grad():
    m = momentum_schedule[it]
    for param_q, param_k in zip(student.module.parameters(),
                                teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

`--momentum_teacher`는 이 $m$의 **시작값(base EMA)** 이다. 끝값은 항상 `1`로 하드코딩되어 있어서(스케줄러 두 번째 인자) 사용자가 바꿀 수 있는 건 "얼마나 느린 교사에서 출발할지"뿐이다.

```
--momentum_teacher, default=0.996
  "Base EMA parameter for teacher update. The value is increased to 1 during
   training with cosine schedule. We recommend setting a higher value with
   small batches: for example use 0.9995 with batch size of 256."
```

## 2. $m$이 작다 = 교사의 기억이 짧다

EMA의 유효 시간상수는 $\tau_{\text{eff}} = \dfrac{1}{1-m}$ iteration이다. 즉 **교사는 대략 최근 $1/(1-m)$ step의 학생을 평균한 모델**이다. asset의 walkthrough(§9)가 이 값을 직접 표로 찍는다.

| $m$ | $1/(1-m)$ [iter] | ~epoch (niter=1251) | 교사의 성격 |
|---|---|---|---|
| 0.9 | 10 | 0.008 | 사실상 학생 그 자체 |
| 0.99 | 100 | 0.08 | 너무 빠름 |
| **0.996** | **250** | **0.20** | **DINO 기본값** |
| 0.999 | 1,000 | 0.80 | batch 작을 때 권장 구간 |
| 0.9999 | 10,000 | 8.0 | 거의 얼어붙음 |
| 1.0 | $\infty$ | — | 완전 정지 (스케줄 종점) |

walkthrough는 이것을 수치 실험으로도 확인한다: 학생 파라미터를 `1.0`으로 고정하고 교사를 `0.0`에서 EMA로 굴리면, 정확히 $1/(1-m)$ step 후 교사값이 $1 - 1/e \approx 0.632$에 도달한다 — 교과서적인 1차 저역통과 필터(low-pass filter)의 응답이다.

**핵심**: EMA는 학생 궤적에 걸린 저역통과 필터다. $m$을 작게 주는 것은 이 필터의 차단주파수를 올려서 **필터를 끄는 것**과 같다.

## 3. 필터를 끄면 무슨 일이 생기나 — 두 단계 고장

### (a) 타겟 요동 — "움직이는 표적 쏘기"

DINO 한 스텝의 loss는 교사 분포를 정답으로 삼는 cross-entropy 18개 항의 평균이다.

$$
\min_{\theta_s}\ \frac{1}{|\mathcal{N}|}\sum_{u\in V^g}\sum_{v \neq u} H\big(P_{\theta_t}(u),\, P_{\theta_s}(v)\big)
$$

$m$이 작으면 $P_{\theta_t}$가 매 iteration 크게 흔들린다. 흔들리는 원인은 많다 — multi-crop 증강이 배치마다 다른 crop을 뽑고, AdamW 스텝은 노이즈가 있고, `out_dim=65536`짜리 프로토타입 배정은 조금만 밀려도 argmax가 튄다. 학생이 방금 맞춘 타겟이 다음 스텝에 이미 다른 곳에 있으므로 gradient 방향이 일관되지 않고 loss가 진동한다. MoCo가 말한 "dictionary consistency"가 깨지는 것과 같은 현상이다.

### (b) 자기참조 붕괴 — 더 치명적인 쪽

$m \to 0$의 극한에서 교사는 **학생의 stop-gradient 사본**이다. 그러면 목적함수는 자명해(trivial solution)를 갖는다: 입력이 무엇이든 같은 one-hot 분포를 출력하면 cross-entropy가 0으로 내려간다. 이미지 정보를 하나도 쓰지 않고 loss를 완벽하게 만족시킬 수 있으므로 최적화가 그쪽으로 끌려간다.

momentum이 있으면 이 지름길이 막힌다. 타겟이 **과거 학생들의 평균**이라 현재 학생이 즉석에서 "합의"할 수 없고, 타겟을 맞추려면 실제 이미지에 근거한 표현을 만들어야 한다. DINO 논문은 이 교사를 **Polyak–Ruppert averaging**으로 해석하고, 그 결과 "교사가 학습 내내 학생보다 계속 더 좋다"고 관찰한다. 학생은 자기보다 나은 목표를 쫓게 되고(bootstrap), 그 향상이 다시 EMA로 교사에 흘러든다.

centering·sharpening이 붕괴를 막아주지 않느냐고 물을 수 있는데, **막아주지 못한다**. center는 교사 출력의 배치 평균 EMA이므로 교사가 학생과 같아지면 center도 학생 자신의 통계일 뿐이고, 자기참조 루프를 끊지 못한다.

## 4. 논문 근거 — 숫자로 얼마나 나쁜가

### DINO Fig. 6(right) / §5.2 — 교사 갱신 규칙 비교 (ViT-S/16, 300ep, ImageNet k-NN top-1)

| 교사 갱신 방식 | k-NN Top-1 |
|---|---|
| Student copy (= $m$ 없음) | **0.1** ← 랜덤 확률 수준 |
| Previous iter (= 아주 작은 $m$) | **0.1** ← 붕괴 |
| Previous epoch | 66.6 |
| **Momentum (DINO)** | **72.8** |

> "In our setting, using a teacher based on a recent version of the student **does not converge**. This setting requires more normalizations to work."

1000-class ImageNet에서 0.1%는 정확히 무작위 추측이다. 즉 "성능이 나쁘다"가 아니라 **완전 붕괴**다. 흥미롭게도 *previous epoch* 교사(=매우 큰 $m$에 대응)는 붕괴하지 않고 66.6까지 나온다 — 고장은 **느린 쪽이 아니라 빠른 쪽에서** 난다.

### DINO Table 7 (§5.1) — momentum 유무

| # | Mom. | SK | MC | Loss | k-NN | Linear |
|---|---|---|---|---|---|---|
| 1 | ✓ | ✗ | ✓ | CE | **72.8** | **76.1** |
| 2 | **✗** | ✗ | ✓ | CE | **0.1** | **0.1** |
| 3 | ✓ | ✓ | ✓ | CE | 72.2 | 76.0 |
| 9 | ✗ | ✓ (SwAV) | ✓ | CE | 64.7 | 71.8 |

> "in the absence of momentum, our framework does not work (row 2) and more advanced operations, SK for example, are required to avoid collapse (row 9). However, with momentum, using SK has little impact (row 3)."

### DINO Table 15 (Appendix B) — 무엇이 붕괴를 대신 막을 수 있나 (linear top-1)

| Momentum | 교사 연산 | Top-1 |
|---|---|---|
| ✓ | Centering | **76.1** |
| ✗ | Centering | **0.1** |
| ✗ | Softmax(batch) | 72.2 |
| ✗ (SwAV) | Sinkhorn-Knopp | 71.8 |

momentum을 빼면 centering만으로는 0.1로 죽고, 살리려면 배치 전체에 걸친 강한 정규화(Softmax over batch, SK)를 도입해야 한다. 논문 결론:

> "these ablations highlight the importance of the momentum encoder, **not only for performance but also to stabilize training**, removing the need for normalization beyond centering."

### 계보 — BYOL / MoCo의 같은 ablation

DINO의 $0.996 \to 1$ 코사인 스케줄은 BYOL에서 그대로 가져온 것이다. 두 선행 연구가 같은 U자 곡선을 보고한다.

**BYOL Table 5(a)** (ResNet-50, 300ep, linear top-1) — target decay $\tau$:

| $\tau_{\text{base}}$ | Top-1 |
|---|---|
| 0 (stop-grad of online, 즉 즉시 복사) | **0.3** |
| 0.9 | 68.4 |
| 0.99 | **72.5** (최적) |
| 0.999 | 69.8 |
| 1 (랜덤 네트워크 고정) | 18.8 |

> "**Instantaneously updating the target network ($\tau=0$) destabilizes training**, yielding very poor performance while never updating the target ($\tau=1$) makes the training stable but prevents iterative improvement."

**MoCo §4.1** (ResNet-50, linear) — momentum $m$:

| $m$ | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|---|---|---|---|---|---|
| Acc (%) | **fail** | 55.2 | 57.8 | **59.0** | 58.9 |

> "at the extreme of no momentum ($m$ is 0), **the training loss oscillates and fails to converge**."

**세 논문의 합의**: $m \to 0$ 쪽 끝은 **하드 실패**(붕괴·발산)이고, $m \to 1$ 쪽 끝은 **소프트 실패**(느려서 덜 좋아짐)다. 그래서 스케줄이 굳이 **증가** 방향인 것이다.

## 5. 그래서 왜 `1.0`까지 올리는가

walkthrough §8의 스케줄 4종 표가 방향을 나란히 놓는다.

| 스케줄 | 시작 → 끝 | 방향 | 왜 |
|---|---|---|---|
| learning rate | $0 \to$ lr $\to 10^{-6}$ | warmup 후 감소 | 표준 |
| weight decay | $0.04 \to 0.4$ | **증가** | 초기엔 자유 탐색, 후반에 표현 압축 |
| **teacher momentum** $m$ | $\mathbf{0.996 \to 1.0}$ | **증가** | **교사를 점점 얼려 타겟을 안정화** |
| teacher temp $\tau_t$ | $0.04 \to 0.07$ | linear (warmup만) | 초기 고온은 불안정 |

- **초기**: 학생이 랜덤에 가까우니 교사가 아예 안 따라오면(=처음부터 $m=1$) 타겟이 영원히 랜덤 네트워크다. BYOL $\tau=1$의 18.8%가 정확히 그 실험이다. 그래서 0.996으로 "조금은" 따라가게 한다.
- **후반**: $m \to 1$이면 교사가 사실상 얼어붙고 타겟이 고정되어 학습이 수렴한다. lr 감쇠와 같은 역할이다.

## 6. 실전 규칙과 흔한 오해

**배치 크기에 따라 올려라.** 공식 권장은 batch 256에서 `0.9995`다. 유효 평균 창을 *iteration*이 아니라 *본 이미지 수* 기준으로 비슷하게 유지하려는 것: batch 1024에서 $250 \times 1024 \approx 2.6\times10^5$장, batch 256에서 $2000 \times 256 \approx 5\times10^5$장. 배치가 작으면 스텝당 노이즈가 커지므로 필터를 더 세게 걸어야 한다. 스모크 테스트처럼 아주 작은 배치로 돌릴 때 `0.996`을 그대로 두면 타겟 요동이 커진다.

**붕괴는 loss를 봐서 알 수 없다.** walkthrough 함정 목록 4번: "사전학습에 검증이 없다 → 조기 종료도 best 선택도 불가. **loss만 보면 붕괴를 놓친다**." 붕괴하면 loss는 오히려 매끄럽게 낮아진다. 실제로 봐야 하는 것은 교사 출력 분포의 **엔트로피**와 argmax 프로토타입의 분산(walkthrough §7 패널 B/C가 이 진단을 그린다), 또는 주기적 k-NN 평가다.

**`center_momentum`과 혼동하지 말 것.** 이름이 비슷하지만 완전히 다른 knob이다.

| 파라미터 | 기본값 | 대상 | 잘못 주면 |
|---|---|---|---|
| `momentum_teacher` | 0.996 → 1 | **교사 파라미터** EMA | **작으면 타겟 요동 → 붕괴** |
| `center_momentum` | 0.9 (고정) | 교사 출력 **center** 벡터 EMA | 너무 크면 편향 추적 실패 |

DINO 논문 Appendix D에 $m \in \{0, 0.9, 0.99, 0.999\} \to$ k-NN $\{69.1, 69.7, 69.4, 0.1\}$ 표가 있는데, 이건 **center EMA rate**이고 교사 momentum이 아니다. 게다가 여기선 방향이 반대로("update가 **너무 느릴 때**, 즉 $m=0.999$에서만 붕괴") 나오므로 잘못 인용하기 쉽다.

**스케줄러는 무상태(stateless)라 resume이 정확하다.** `cosine_scheduler`는 학습 전에 `epochs × niter_per_ep` 길이 배열을 만들고 루프에서 `schedule[it]`로 조회한다. 다만 `assert len(schedule) == epochs * niter_per_ep`가 있어서 `warmup_epochs`(기본 10) > `epochs`면 여기서 죽는다 — 짧게 테스트할 땐 `--warmup_epochs 0`.

---

## 7. 30초 복습

1. 교사는 gradient 없이 $\theta_t \leftarrow m\theta_t + (1-m)\theta_s$로만 갱신된다.
2. 교사의 기억 길이 = $1/(1-m)$ iteration. $m$이 작으면 기억이 짧다.
3. $m$이 작으면 → 타겟이 매 스텝 요동 + 교사가 학생 사본이 되어 자기참조 → **붕괴**.
4. 실측: DINO student-copy 교사 = k-NN **0.1%**(확률 수준), BYOL $\tau=0$ = **0.3%**, MoCo $m=0$ = **loss 진동, 미수렴**.
5. centering만으로는 못 막는다. momentum encoder 자체가 붕괴 방지 장치의 일부다.
6. 그래서 기본값 `0.996`에서 **1.0까지 올려** 교사를 점점 얼린다. 배치가 작으면 시작값도 올려라(256 → `0.9995`).

## 참고

- asset: `.fm/assets/dino_training_walkthrough.py` §8 (스케줄 4종), §9 (EMA teacher 갱신), §14 (하이퍼파라미터 표)
- 코드: `main_dino.py:61` (인자), `:250` (스케줄 생성), `:347-350` (EMA 갱신), `utils.py:197` (`cosine_scheduler`)
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), arXiv:2104.14294 — §3.1, §5.1 Table 7, §5.2 Fig. 6, Appendix B Table 15
- Grill et al., *Bootstrap Your Own Latent* (BYOL), arXiv:2006.07733 — §4.1, §5 Table 5(a)
- He et al., *Momentum Contrast* (MoCo), arXiv:1911.05722 — §3.1, §4.1 "Ablation: momentum"
