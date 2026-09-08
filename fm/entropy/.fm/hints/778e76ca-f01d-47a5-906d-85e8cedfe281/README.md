# 토이 실험 ↔ DINO 논문 Fig. 7

**Q.** 토이 실험 결과가 DINO 원본 논문의 어떤 그림과 대응하는가?
**A.** **Fig. 7**(§5.3 *Avoiding collapse*, 논문 9쪽). ImageNet에서 centering만 쓰면 teacher 엔트로피가 $\log K$로 올라가고, sharpening만 쓰면 0으로 내려가는 바로 그 그림이다.

![DINO 논문 Fig. 7](fig-7.jpeg)

---

## 1. Fig. 7은 정확히 무엇을 그린 그림인가

### 캡션 원문

> **Figure 7: Collapse study. (left):** evolution of the teacher's target entropy along training epochs; **(right):** evolution of KL divergence between teacher and student outputs.

### 번역

> **그림 7: 붕괴 연구. (왼쪽):** 학습 epoch에 따른 **teacher의 타깃 엔트로피** 변화; **(오른쪽):** **teacher와 student 출력 사이의 KL divergence** 변화.

### 축과 곡선

패널은 두 개, 곡선은 각 패널에 **세 개**다.

| | 왼쪽 패널 | 오른쪽 패널 |
|---|---|---|
| x축 | `epochs`, 0 → 100 | `epochs`, 0 → 100 |
| y축 | `Target Entropy` — teacher 타깃 분포의 엔트로피 $h(P_t)$, 눈금 0 / 2 / 4 / 6 / 8 | `KL divergence` — $D_{KL}(P_t \| P_s)$, 눈금 0 / 2 |

범례의 세 곡선은 **teacher 출력에 어떤 연산을 적용했는가**를 가리킨다.

| 범례 | 선 모양 | 실제 설정 | 왼쪽(엔트로피) | 오른쪽(KL) |
|---|---|---|---|---|
| `sharpening` | 파란 실선 | sharpening만 (**centering 없음**) | 초반 아주 잠깐 튀었다가 곧장 **0에 붙어** 100 epoch 내내 0 | **0에 딱 붙어** 평평 |
| `centering` | 빨간 점선 | centering만 (**sharpening 없음**) | 처음부터 끝까지 **$\approx 8.3$ 에서 수평** | **0에 딱 붙어** 평평 |
| `both` | 주황 실선 | 둘 다 = DINO | 8 근처에서 시작 → 20 epoch 부근에서 급락 → 100 epoch에서 **$\approx 1$** 로 완만히 수렴 | 0에서 출발해 **25~30 epoch에 $\approx 1.5$로 피크**, 이후 서서히 낮아져 100 epoch에 $\approx 0.9$ — **끝까지 0이 아님** |

빨간 점선의 수평 높이 $\approx 8.3$은 $\ln 4096 = 8.317$과 맞아떨어진다. 논문은 Fig. 7의 $K$를 캡션에 명시하지 않았지만, Appendix D의 여러 ablation이 "The output dimensionality $K$ is set to 4096 in this experiment"라고 밝히고 있어 이 그림도 $K=4096$ 설정일 가능성이 높다(본문 기본값은 $K = 65536$, 그 경우 $\ln K = 11.09$). **눈금값 자체를 외우지 말고 "$\log K$ 상한에 딱 붙어 있다"를 외울 것.**

---

## 2. 두 곡선의 해석 — 그리고 KL 패널이 진짜 판정자다

### 논문 본문(§5.3) 인용

> There are two forms of collapse: regardless of the input, the model output is uniform along all the dimensions or dominated by one dimension. The centering avoids the collapse induced by a dominant dimension, but encourages an uniform output. Sharpening induces the opposite effect.

붕괴는 두 종류다 — 입력과 무관하게 출력이 (a) 모든 차원에 **균등**하거나 (b) **한 차원이 지배**하거나. centering은 (b)를 막지만 (a)를 부추기고, sharpening은 정반대다.

교차엔트로피를 쪼개면 두 붕괴가 서로 다른 성분에서 드러난다 — 논문 식 (5):

$$H(P_t, P_s) = h(P_t) + D_{KL}(P_t \| P_s). \qquad (5)$$

> A KL equal to zero indicates a constant output, and hence a collapse. In Fig. 7, we plot the entropy and KL during training with and without centering and sharpening. If one operation is missing, the KL converges to zero, indicating a collapse. However, the entropy $h$ converges to different values: 0 with no centering and $-\log(1/K)$ with no sharpening, indicating that both operations induce different form of collapse. Applying both operations balances these effects.

정리하면:

- **왼쪽 패널(엔트로피)** = *어떤 종류의* 붕괴인지 알려주는 지표
  - centering만 → $h \to -\log(1/K) = \log K$. teacher가 균등분포로 퍼짐 = **균등 붕괴**.
  - sharpening만 → $h \to 0$. teacher가 원-핫으로 뾰족해짐 = **한 차원 지배 붕괴**.
  - 둘 다 → 두 힘이 서로를 상쇄해 **중간값($\approx 1$)에서 안정**.
- **오른쪽 패널(KL)** = *붕괴했는가 안 했는가*를 판정하는 지표
  - 붕괴한 두 설정(파란·빨간)은 **KL이 모두 0**이다. 엔트로피 값은 정반대(0 vs $\log K$)인데 KL은 똑같이 0 — 즉 **엔트로피만 봐서는 두 붕괴를 붕괴로 묶어낼 수 없고, KL이 공통 판정 기준**이다.
  - $D_{KL}(P_t\|P_s) = 0$ 은 "student가 teacher를 **완벽히** 따라잡았다"는 뜻이다. 손실 $H = h + 0 = h$ 는 student 파라미터에 대해 상수가 되고, **더 배울 것이 남지 않는다**. 학습 신호가 사라진 상태 = 붕괴.
  - 건강한 `both` 곡선만 KL이 0에서 떠올라 유지된다 — teacher가 계속 student보다 앞서 있고, 그 격차가 곧 학습 신호다. (같은 관찰을 다른 각도에서 본 것이 Fig. 6 왼쪽의 "teacher가 student를 계속 앞선다"이다.)

> **한 줄 요약**: 엔트로피 = *어느 쪽으로* 무너지는가, KL = *무너졌는가*.

---

## 3. 토이 ↔ 논문 대응표

토이(`dino_collapse_cross_entropy.py` §5): 2D 평면 위 6개 클러스터, student/teacher 모두 2→128→128→32 MLP, EMA $m = 0.99$, `out_dim = 32`이므로 $\log K = \log 32 = 3.47$. `student_temp = 0.1` 고정.

토이의 네 설정과 논문 세 곡선의 대응:

| 토이 config | teacher 설정 | 논문 Fig. 7 곡선 | 논문의 관찰 | 토이 실측 (노트북 §5 판독) |
|---|---|---|---|---|
| `none` (center off, temp 0.1) | 둘 다 없음 | **없음** (논문은 3곡선뿐) | — | `h_each ≈ h_mean ≈ 2.9`(높음) 인데 `top_share ≈ 0.8` — 퍼져 있으면서 다들 같은 차원을 가리킴, `code_acc` ≈ chance |
| `center only` (center on, temp 0.1) | centering만 | 빨간 점선 `centering` | $h \to \log K$, KL $\to 0$ | `h_each → log K = 3.47`, `top_share ≈ 0.33`, loss도 $\log K$ 근처에서 멈춤 → **논문과 일치** |
| `sharp only` (center off, temp 0.04) | sharpening만 | 파란 실선 `sharpening` | $h \to 0$, KL $\to 0$ | `h_each = 0` ✅ 그러나 `h_mean ≈ 1.2`, `top_share = 0.5`, **코드 4개**만 쓰임, loss ≈ 0.2 → **엔트로피는 일치, "완전한" 붕괴에는 미달** |
| `both = DINO` (center on, temp 0.04) | 둘 다 | 주황 실선 `both` | 중간값에서 안정, KL > 0 유지 | `h_each ≈ 0.6`(확신) + `h_mean ≈ 2.4`(차원 골고루) + `top_share = 1/6` + `code_acc = 1.0` — **6코드가 6클러스터에 1:1** |

### 어디가 어긋나는가 — 정직하게

- **토이는 KL을 직접 로깅하지 않는다.** 노트북 `hist`가 남기는 것은 `loss / h_each / h_mean / top_share / code_acc`뿐이다. 식 (5)로 역산하면 $D_{KL} \approx \texttt{loss} - \texttt{h\_each}$:
  - `center only`: $3.47 - 3.47 \approx 0$ → 논문의 "KL → 0" 재현 ✅
  - `sharp only`: $0.2 - 0 \approx 0.2$ → **0으로 가지 않는다** ❌
  - `both`: $0.8 - 0.6 \approx 0.2$ → 0은 아니지만 `sharp only`와 구분이 잘 안 된다
  (엄밀히는 토이 loss가 *서로 다른 crop*에 대한 교차 항이라 같은 view 위의 $H(P_t,P_s)$와 정확히 같지는 않다. 어림값으로만 읽을 것.)
- **`sharp only`가 완전한 원-핫 붕괴까지 가지 않는 이유** — 형제 카드가 파고든 지점:
  1. **규모**. 논문은 ImageNet 100 epoch, $K$가 4096~65536이다. 토이는 6클러스터 · 768점 · 1500 step · $K=32$. 한 차원이 전체를 삼키기까지의 "런웨이"가 짧다.
  2. **증강 강도**. 토이 crop은 $\sigma = 0.7$ 가우시안 노이즈로, 클러스터 간격의 1/4 정도다. 서로 다른 클러스터의 crop이 *가끔만* 겹친다. DINO의 multi-crop(global 224² + local 96²)은 이보다 훨씬 공격적이라, 서로 다른 이미지의 crop이 훨씬 자주 구별 불가능해지고 → 한 코드로 몰아붙이는 압력이 훨씬 세다.
  3. 그래서 토이의 `sharp only`는 **6→4 코드로 합쳐지고 한 코드가 데이터 절반을 먹는**, *원-핫 붕괴의 초기 모습*에서 멈춘다. 방향은 논문과 같고 종점만 못 간 것이다.
- **토이가 추가한 것**: `h_mean`(전체 점의 *평균* 분포 엔트로피)은 논문 Fig. 7에 없는 지표다. 그런데 이게 중요하다 — `both`도 `h_each`가 0.6으로 **낮다**. 즉 per-sample 엔트로피만으로는 "건강한 확신"과 "원-핫 붕괴"를 못 가른다. 논문은 그 구분을 KL 패널로 하고, 토이는 `h_mean`/`top_share`/`code_acc`로 한다. **같은 문제에 대한 두 가지 처방**이라고 읽으면 된다.
- **loss는 어느 쪽에서도 못 믿는다**: 토이의 `sharp only` loss(≈0.2)가 `both`(≈0.8)보다 **낮다**. 표현 품질은 정반대인데 loss는 거꾸로 말한다. 논문이 Fig. 7에서 loss를 안 그리고 굳이 $h$와 $D_{KL}$로 **쪼개서** 그린 이유가 이것이다.

---

## 4. 논문에서 이 그림이 하는 역할

§5.3의 논증 구조는 아주 짧고 단선적이다.

1. **주장** (§3, "Avoiding collapse" 단락에서 미리 선언):
   > centering prevents one dimension to dominate but encourages collapse to the uniform distribution, while the sharpening has the opposite effect. Applying both operations balances their effects which is sufficient to avoid collapse in presence of a momentum teacher.
2. **도구**: 식 (5)로 손실을 $h + D_{KL}$로 분해 — 붕괴를 *측정 가능한 양*으로 바꾼다.
3. **증거**: **Fig. 7**. "centering만" / "sharpening만" / "둘 다" 세 번을 실제로 돌려, 하나만 쓰면 KL이 0으로 죽고 엔트로피는 **서로 반대 방향**으로 극단에 붙는다는 것을 보인다.
4. **결론**: 두 연산은 **상보적**이며 하나만으로는 안 된다. 정도(강도)는 Appendix D의 $\tau_t$ ablation으로 넘긴다.

### 식 (4)와의 연결

Fig. 7의 빨간 점선(centering만)은 식 (4)가 무엇을 하는 장치인지를 눈으로 보여준다. centering은 teacher 로짓에 bias $c$를 더하는 것이고,

$$c \leftarrow m c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i) \qquad (4)$$

배치 평균 로짓을 EMA로 추적해 빼주므로 **어떤 차원도 평균적으로 튀어나올 수 없다**. 그 결과 "한 차원 지배"는 원천 봉쇄되지만, 아무도 튀지 못하니 출력은 균등으로 눕는다 → $h \to \log K$. 반대편에서 $\tau_t$로 나누는 sharpening이 다시 뾰족하게 당겨줘야 균형이 잡힌다. Fig. 7 왼쪽의 주황 곡선이 8에서 1까지 내려와 **멈추는** 것이 그 균형점이다.

또한 논문은 centering이 **1차 배치 통계에만** 의존한다는 점을 강조한다("the centering operation only depends on first-order batch statistics") — 이 덕분에 §5.5의 작은 배치(배치 8까지)에서도 동작한다. SwAV의 Sinkhorn-Knopp 같은 배치 전체에 대한 최적화가 아니라 값싼 bias 하나라는 것이 설계 포인트다.

---

## 5. "teacher 엔트로피를 로깅하라"의 출처

Fig. 7은 사실상 **자기지도 학습에서 붕괴 진단 지표를 어떻게 잡는가**의 레퍼런스가 됐다. 실무에서 흔히 쓰는 조합:

- **teacher 출력 엔트로피** $h(P_t)$ — 0으로 내려가면 원-핫 붕괴, $\log K$로 올라가면 균등 붕괴. 두 극단 사이에 머무르는지만 보면 된다.
- **teacher–student KL** $D_{KL}(P_t\|P_s)$ — 0으로 수렴하면 학습 신호 소멸. 논문의 공식 판정 기준.
- **prototype 사용 히스토그램 / top_share** — 토이가 쓰는 지표. $K$개 코드 중 몇 개가 실제로 쓰이는지. (Fig. 7에는 없지만 원-핫 붕괴의 조기 경보로 유용하다.)
- 요즘 라이브러리의 DINO 학습 콜백은 여기에 **effective rank**나 **Gram 행렬** 기반 지표를 얹기도 한다.

후속 연구는 진단보다 **붕괴 방지 장치 자체를 교체**하는 쪽으로 갔다. iBOT은 DINO 프레임워크에 masked image modeling 목적함수를 얹었고, DINOv2는 iBOT을 데이터·모델 규모로 키우면서 teacher centering을 **Sinkhorn-Knopp**로 바꾸고 특징이 뭉치지 않도록 **KoLeo** 정규화를 추가했다. 흥미롭게도 DINO 논문 스스로도 Appendix(Tab. 15, "Relation to SwAV")에서 teacher 연산을 Centering / Sinkhorn-Knopp / batch축 Softmax 셋으로 비교하며 **momentum encoder가 있으면 centering만으로도 붕괴를 피한다**는 것을 확인해 뒀다. Fig. 7은 그 계보의 출발점이다.

참고: [DINOv2 (arXiv:2304.07193)](https://arxiv.org/html/2304.07193v2), [DINO (arXiv:2104.14294)](https://arxiv.org/pdf/2104.14294)

---

## 6. Fig. 7과 같이 읽어야 하는 Appendix D 표 두 개

Fig. 7이 "on/off" 실험이라면, Appendix D는 같은 축을 **연속적으로** 훑은 것이다. 짝으로 읽어야 그림이 완성된다.

### (a) Online centering — center momentum $m$ (식 (4)의 $m$)

| $m$ | 0 | 0.9 | 0.99 | **0.999** |
|---|---|---|---|---|
| $k$-NN top-1 | 69.1 | 69.7 | 69.4 | **0.1** |

> The convergence is robust to a wide range of smoothing, and the model only collapses when the update is too slow, i.e., $m = 0.999$.

**읽는 법**: $m$이 너무 크면 center $c$가 배치 통계를 못 따라간다 → 사실상 centering이 **꺼진** 것과 같아진다 → Fig. 7의 파란 곡선(sharpening만)으로 굴러떨어진다. $k$-NN 0.1%는 chance(=1/1000) 수준, 완전 붕괴의 흔적이다. 반대로 $m=0$(EMA 없이 매 배치 평균 그대로)도 69.1로 멀쩡하다 — centering은 **있기만 하면** 되고 세기에는 관대하다.

### (b) Sharpening — teacher 온도 $\tau_t$

| $\tau_t$ | 0 | 0.02 | 0.04 | 0.06 | **0.08** | 0.04 → 0.07 |
|---|---|---|---|---|---|---|
| $k$-NN top-1 | 43.9 | 66.7 | 69.6 | 68.7 | **0.1** | **69.7** |

> we observe that a temperature lower than 0.06 is required to avoid collapse. When the temperature is higher than 0.06, the training loss consistently converges to $\ln(K)$. However, we have observed that using higher temperature than 0.06 does not collapse if we start the training from a smaller value and increase it during the first epochs. In practice, we use a linear warm-up for $\tau_t$ from 0.04 to 0.07 during the first 30 epochs of training. Finally, note that $\tau \rightarrow 0$ (extreme sharpening) correspond to the `argmax` operation and leads to one-hot hard distributions.

**읽는 법**: 이 표는 Fig. 7의 **빨간 점선을 연속적으로 재현한 것**이다. $\tau_t = 0.08$에서 "training loss consistently converges to $\ln(K)$"는 곧 $h \to \log K$, $D_{KL} \to 0$ — 그림의 빨간 점선 그 자체다. 반대쪽 끝 $\tau_t = 0$(= `argmax`, 완전 원-핫)은 43.9로 붕괴는 아니지만 성능이 크게 깎인다 — centering이 버텨주기 때문에 파란 곡선까지 굴러가지는 않는다. 즉 **Fig. 7의 두 극단 사이 좁은 골짜기 안에서만 DINO가 산다**. 그리고 논문의 최종 레시피인 $\tau_t$ **warm-up 0.04 → 0.07**(첫 30 epoch)이 69.7로 가장 좋다 — 시작은 뾰족하게 잡아 붕괴를 피하고, 자리를 잡은 뒤 온도를 올려 여유를 준다.

두 표를 한 문장으로: **centering은 켜져 있기만 하면 관대하고, sharpening은 세기가 예민하다.** Fig. 7은 이 비대칭의 on/off 버전이다.

---

## 7. 이 카드의 핵심만 다시

- 대응하는 그림은 **Fig. 7**, §5.3 *Avoiding collapse*, 논문 9쪽. 왼쪽 = teacher 타깃 엔트로피, 오른쪽 = teacher–student KL, 곡선 3개(`sharpening` / `centering` / `both`).
- centering만 → $h \to \log K$ (균등 붕괴) · sharpening만 → $h \to 0$ (원-핫 붕괴) · 둘 다 → 중간에서 안정.
- **붕괴 판정은 KL**: 붕괴한 두 설정 모두 KL → 0 = student가 teacher를 완벽히 따라잡음 = 학습 신호 소멸.
- 토이는 $K=32$의 축소판으로 **방향은 그대로 재현**하되, `sharp only`가 코드 4개에서 멈추고 KL도 0으로 가지 않는다 — 규모와 증강 강도가 논문보다 훨씬 약하기 때문.
