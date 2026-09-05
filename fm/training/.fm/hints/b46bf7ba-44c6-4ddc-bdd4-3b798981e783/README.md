# EMA 전후 `max|θs − θt|` 가 거의 안 변한 이유

> **한 줄 답**: EMA 한 step은 학생–교사 차이를 **정확히 $m$ 배**로 줄인다. $m = 0.996$ 이므로 한 step에 겨우 $0.4\%$ 만 좁혀진다. $1.568\times10^{-5} \times 0.996 = 1.5617\times10^{-5}$ → 출력의 $1.562\times10^{-5}$ 와 일치.

---

## 1. 문제의 셀 (§10, 12단계 중 마지막)

`dino_training_walkthrough.py` §10 "학습 1 iteration 완전 해부" 의 12번 단계다.

```python
optimizer.step()                                                           # 11)

with torch.no_grad():                                                      # 12)
    m = mo_s[gi]
    d0 = max((ps - pt).abs().max().item()
             for ps, pt in zip(student.parameters(), teacher.parameters()))
    for pq, pk in zip(student.parameters(), teacher.parameters()):
        pk.data.mul_(m).add_((1 - m) * pq.detach().data)
    d1 = max((ps - pt).abs().max().item()
             for ps, pt in zip(student.parameters(), teacher.parameters()))
print(f"\nEMA 전 max|θs-θt| = {d0:.3e}  →  후 {d1:.3e}  (m={m:.5f} 이라 교사는 아주 조금만 따라감)")
```

출력:

```
EMA 전 max|θs-θt| = 1.568e-05  →  후 1.562e-05  (m=0.99600 이라 교사는 아주 조금만 따라감)
```

이 갱신은 `main_dino.py:346-350` 의 실제 학습 코드와 한 글자도 다르지 않다.

```python
# EMA update for the teacher
with torch.no_grad():
    m = momentum_schedule[it]  # momentum parameter
    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

---

## 2. 왜 "거의 안 변하나" — 차이는 정확히 $m$ 배가 된다

EMA 갱신은

$$
\theta_t' \;=\; m\,\theta_t + (1-m)\,\theta_s
$$

이다. **이 시점에 학생은 움직이지 않는다** (`optimizer.step()` 은 이미 끝났고, EMA 루프 안에서 $\theta_s$ 는 읽기만 한다). 그러므로 갱신 후의 차이는

$$
\begin{aligned}
\theta_s - \theta_t'
&= \theta_s - \big(m\,\theta_t + (1-m)\,\theta_s\big) \\
&= \theta_s - m\,\theta_t - \theta_s + m\,\theta_s \\
&= m\,(\theta_s - \theta_t)
\end{aligned}
$$

즉 **모든 파라미터의 차이가 일제히 정확히 $m$ 배로 스칼라 스케일**된다. 원소별 스케일이 전부 같은 양수 $m$ 이므로 절댓값의 최댓값을 취하는 위치(argmax)도 바뀌지 않고, 결과적으로

$$
d_1 \;=\; \max_i |\theta_s^{(i)} - \theta_t'^{(i)}| \;=\; m \cdot \max_i |\theta_s^{(i)} - \theta_t^{(i)}| \;=\; m \, d_0
$$

가 **근사가 아니라 항등식**으로 성립한다.

### 검산

$$
d_1 = 0.996 \times 1.568\times10^{-5} = 1.561728\times10^{-5} \;\approx\; \underline{1.562\times10^{-5}}
$$

출력의 세 자리 유효숫자와 정확히 맞는다. 줄어든 양은

$$
d_0 - d_1 = (1-m)\,d_0 = 0.004 \times 1.568\times10^{-5} = 6.3\times10^{-8}
$$

로, $d_0$ 대비 $0.4\%$. "거의 안 변했다"는 관측은 버그가 아니라 **$m=0.996$ 의 정의 그 자체**다.

---

## 3. 왜 애초에 $d_0$ 이 $1.5\times10^{-5}$ 처럼 작은가

두 가지가 겹친다.

### (a) 교사는 학생의 **복사본**에서 출발한다

`build_pair()` 안에서

```python
teacher.load_state_dict(student.state_dict())   # 같은 가중치에서 출발
for p in teacher.parameters():
    p.requires_grad = False                     # 교사는 backprop 없음
```

이므로 iteration 0 진입 시점에는 $\theta_s = \theta_t$, 즉 $d = 0$ 이다. 이 셀에서 두 모델이 벌어진 유일한 원인은 **`optimizer.step()` 단 한 번**이다.

### (b) 그 한 번의 step 크기가 곧 lr 이다

§10 셀 설정:

```python
BATCH = 8
lr_s = utils.cosine_scheduler(0.0005 * BATCH / 256., 1e-6, NEPOCHS, niter, warmup_epochs=0)
wd_s = utils.cosine_scheduler(0.04, 0.4, NEPOCHS, niter)
mo_s = utils.cosine_scheduler(0.996, 1.0, NEPOCHS, niter)
```

**linear scaling rule** ($\texttt{lr}_{\text{eff}} = 0.0005 \times \tfrac{\text{batch}}{256}$) 때문에 배치 8짜리 노트북 설정의 기저 lr은

$$
\texttt{lr} = 0.0005 \times \frac{8}{256} = 1.5625\times10^{-5}
$$

이고, `warmup_epochs=0` 이라 코사인 스케줄이 곧장 기저값에서 시작한다 → `lr_s[0] = 1.5625e-05`.

> 참고: 노트북이 warmup을 끈 이유는 §8의 함정 노트 그대로다 — `cosine_scheduler` 의 `assert len(schedule) == epochs * niter_per_ep` 때문에 `warmup_epochs > epochs` 면 죽는다. 실전 `main_dino.py` 기본값은 `--warmup_epochs 10` 이라 **진짜 it=0 에서는 lr이 0에 가깝고**, 그 경우 $d_0$ 은 여기보다도 더 작다.

여기에 AdamW의 첫 step 성질이 붙는다. $t=1$ 에서 bias correction 을 적용하면

$$
\hat m_1 = \frac{(1-\beta_1)g}{1-\beta_1} = g,\qquad
\hat v_1 = \frac{(1-\beta_2)g^2}{1-\beta_2} = g^2
$$

$$
\Delta\theta = -\,\texttt{lr}\cdot\frac{\hat m_1}{\sqrt{\hat v_1}+\varepsilon} - \texttt{lr}\cdot\texttt{wd}\cdot\theta
\;=\; -\,\texttt{lr}\cdot\frac{g}{|g|+\varepsilon} - \texttt{lr}\cdot\texttt{wd}\cdot\theta
\;\approx\; -\,\texttt{lr}\cdot\operatorname{sign}(g) - \texttt{lr}\cdot\texttt{wd}\cdot\theta
$$

**AdamW 첫 step의 이동량은 gradient 크기와 거의 무관하게 $\approx \texttt{lr}$ 이다.** 그래서

$$
d_0 \;\approx\; \texttt{lr}\,\big(1 + \texttt{wd}\,|\theta|\big)
$$

이고, 실측 $1.568\times10^{-5}$ 를 $\texttt{lr}=1.5625\times10^{-5}$ 로 나누면 $1.0035$ — 즉 $\texttt{wd}\cdot|\theta| \approx 0.0035$, `wd_s[0] = 0.04` 이므로 $|\theta| \approx 0.09$ 인 (0번 param_group의) 가중치 하나가 최댓값 자리를 차지하고 있다는 뜻이다. 앞자리가 lr과 일치하는 게 우연이 아니다.

> **부수 확인**: 10단계 `cancel_gradients_last_layer(epoch=0, ..., freeze_last_layer=1)` 로 `head.last_layer.weight_v.grad = None` 이 됐다. AdamW는 `grad is None` 인 파라미터를 통째로 건너뛰므로 그 텐서는 weight decay조차 안 먹고 $\theta_s = \theta_t$ 그대로 남는다 — 최댓값 후보에서 자동 탈락.

---

## 4. `max|·|` 를 진단량으로 쓴다는 것

$$
d = \max_i \big|\theta_s^{(i)} - \theta_t^{(i)}\big| = \lVert \theta_s - \theta_t \rVert_\infty
$$

전체 파라미터 벡터에 대한 **$L_\infty$ 노름**, 즉 "가장 크게 어긋난 스칼라 하나"다.

- 노름(L2)이 아니라 max를 쓰는 이유: 스케일이 파라미터 수에 안 휩쓸린다. 5.5M개 중 하나만 튀어도 잡아낸다.
- 해석: **교사–학생 괴리(lag)의 상한**. 이 값이 0에 붙어 있으면 교사가 학생 복사본과 다를 바 없다는 뜻이고 (self-distillation 신호가 사라짐), 반대로 폭주하면 교사가 학생 궤적을 전혀 못 따라가고 있다는 뜻이다.
- 단점: 위치 정보가 없다. 실전 진단에서는 `{name: (ps-pt).abs().max()}` 를 텐서별로 찍어 **어느 층이** 벌어지는지 보는 편이 낫다.

---

## 5. 이 실험이 실제로 보여주는 것 — EMA는 저역통과 필터

한 step만 보면 "거의 안 변한다"지만, 그것이 바로 EMA의 설계 의도다.

**학생이 계속 움직이는 경우**를 넣어 보자. 매 step 학생이 $\delta$ 만큼 이동한다고 하면, 차이 $e_k = \theta_s - \theta_t$ 의 점화식은

$$
e_{k+1} = m\,e_k + \delta
$$

(EMA가 $m$배로 줄이고, 학생의 새 step이 $\delta$만큼 다시 벌린다.) 고정점은

$$
e_\star = \frac{\delta}{1-m} \;=\; \tau_{\text{eff}}\,\delta,
\qquad \tau_{\text{eff}} = \frac{1}{1-m} = 250 \text{ (m=0.996)}
$$

즉 **교사는 늘 학생 뒤에서 약 $1/(1-m)$ step 지연으로 따라간다.** §9의 그림(학생을 1.0에 고정하고 EMA를 1500 step 돌린 궤적)이 정확히 이 시상수를 시각화한 것이고, $1/(1-m)$ step에서 $1-1/e \approx 0.632$ 를 통과한다.

§10 설정 숫자로 감을 잡으면 $\delta \approx \texttt{lr} = 1.5625\times10^{-5}$ 일 때 정상 상태 $e_\star \approx 3.9\times10^{-3}$ — 관측한 $1.57\times10^{-5}$ 보다 **250배 큰** 값이다. 즉 지금 본 것은 정상 상태가 아니라 **막 출발한 과도구간의 첫 점**이다.

### 실제 학습에서 이 값의 궤적 예상

| 구간 | $d$ 의 거동 | 이유 |
|---|---|---|
| 초기 (warmup~) | $0$ 에서 **증가** | lr이 올라가며 $\delta$ 증가, $e$가 $e_\star$ 로 지수 접근 (시상수 250 iter ≈ 0.2 epoch @ niter=1251) |
| 중반 | **정상 상태 (plateau)** | $\delta$ 와 $(1-m)e$ 가 균형. lr·$m$ 이 천천히 변하므로 plateau도 천천히 표류 |
| 후반 | **감소 → 0 근처** | lr $\to 10^{-6}$ 으로 $\delta \downarrow$. 단 $m \to 1$ 이 반대로 작용 |

주의: 실제 SGD 궤적은 방향이 매 step 코히런트하지 않아(랜덤워크에 가까워) 위 $e_\star$ 는 **상한**에 가깝다.

### $m \to 1$ 이면?

`mo_s = cosine_scheduler(0.996, 1.0, ...)` 로 학습 후반 $m \to 1$ 이면 $(1-m) \to 0$, 즉

- 차이가 **거의 안 줄어든다** → 교사가 사실상 **얼어붙는다(frozen target)**.
- $\tau_{\text{eff}} = 1/(1-m) \to \infty$: $m=0.9999$ 면 10000 iter(≈8 epoch), $m=0.99999$ 면 100000 iter(≈80 epoch) 평균.
- 이게 §9와 §14 표에서 말하는 "타겟을 고정해 후반 학습을 안정화"의 실체다. 교사가 안 움직이면 학생이 쫓는 목표가 고정되어 붕괴/진동이 줄어든다.

---

## 6. 정밀도 한 줄

$|\theta|\approx 0.09$ 근처에서 fp32의 ULP는 $7.5\times10^{-9}$ 이라 $1.5\times10^{-5}$ 차이는 유효숫자 3~4자리 여유로 안전하지만, **fp16의 ULP는 $6.1\times10^{-5}$** 라 이 갱신량이 통째로 반올림으로 사라진다 — DINO가 AMP(`fp16_scaler`)를 쓰면서도 **파라미터와 EMA 누산은 fp32로 유지**하는 이유다.

---

## 7. 재현 스니펫

개념만 5줄로:

```python
import torch
m, theta_s, theta_t = 0.996, torch.tensor(1.0), torch.tensor(0.0)
d0 = (theta_s - theta_t).abs().max()
theta_t.mul_(m).add_((1 - m) * theta_s)          # main_dino.py 와 동일한 갱신
print(d0.item(), (theta_s - theta_t).abs().max().item(), m * d0.item())  # 1.0  0.996  0.996
```

노트북 상황(모델 전체 + 정상 상태 수렴)까지 확인하려면:

```python
import torch, copy
student = torch.nn.Linear(64, 64)
teacher = copy.deepcopy(student)                       # build_pair 의 load_state_dict 와 같은 출발점
opt = torch.optim.AdamW(student.parameters(), lr=1.5625e-5, weight_decay=0.04)
m = 0.996

def maxdiff():
    return max((ps - pt).abs().max().item()
               for ps, pt in zip(student.parameters(), teacher.parameters()))

for step in range(1500):
    opt.zero_grad(); student(torch.randn(8, 64)).pow(2).mean().backward(); opt.step()
    d0 = maxdiff()
    with torch.no_grad():                              # EMA
        for pq, pk in zip(student.parameters(), teacher.parameters()):
            pk.data.mul_(m).add_((1 - m) * pq.detach().data)
    d1 = maxdiff()
    if step in (0, 1, 10, 250, 1499):
        print(f"step {step:4d}  d0={d0:.3e} → d1={d1:.3e}   d1/d0={d1/d0:.5f}")
```

- `step 0` 에서 `d0 ≈ lr` (첫 AdamW step 이동량 ≈ lr) 을 확인.
- `d1/d0` 이 **모든 step에서 정확히 0.996** 으로 찍히는 것을 확인 — §2의 항등식.
- `d0` 자체가 step이 갈수록 커져 $\approx \texttt{lr}/(1-m)$ 부근에서 plateau 하는 것을 확인 — §5의 $e_\star$.

---

## 정리

1. EMA 한 step은 학생–교사 차이를 **정확히 $m$ 배**로 만든다: $\theta_s - \theta_t' = m(\theta_s - \theta_t)$.
2. $m = 0.996$ → 한 step에 $0.4\%$ 만 수축. $1.568\text{e-}5 \times 0.996 = 1.562\text{e-}5$ 로 실측과 일치.
3. 출발 차이가 $1.5\text{e-}5$ 로 작은 건 교사가 학생 복사본이고 `optimizer.step()` 을 **딱 한 번** 했으며, 그 step 크기가 batch 8의 linear scaling rule lr $=1.5625\text{e-}5$ 와 사실상 같기 때문.
4. EMA는 **저역통과 필터**다. 교사는 최근 $1/(1-m) = 250$ iteration 의 학생 평균이고, 학생이 계속 움직이면 정상 상태에서 $\approx \delta/(1-m)$ 만큼 뒤처져 따라간다.
5. 후반 $m \to 1$ 이면 교사는 사실상 고정 타겟이 된다.

### 관련 파일

- `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` — §9 (EMA teacher 갱신), §10 (1 iteration 해부, 문제의 셀)
- `/home/sungwoo/projects/swcho/dino/main_dino.py:346-350` — 실제 EMA 블록
- `/home/sungwoo/projects/swcho/dino/utils.py:187-198` — `cosine_scheduler`
