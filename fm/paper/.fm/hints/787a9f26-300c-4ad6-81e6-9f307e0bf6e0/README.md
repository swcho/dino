# multi-crop의 정확도/시간 트레이드오프 (DINO §5.4, Table 8)

## 카드 요약

- multi-crop 없음($2\times224^2$): **46시간(45.9h, 300 epochs)** 학습 → **72.5%**
- multi-crop 사용($2\times224^2 + 10\times96^2$): **24시간(24.2h, 100 epochs)** → **74.6%**
- 즉 **시간은 약 $2\times$ 적게 쓰면서 +2.1% 높음**. 대신 **peak GPU 메모리는 9.3G → 15.4G**.

원문(§5.4) 그대로:

> the performance is 72.5% after 46 hours of training without multi-crop (i.e. $2\times224^2$) while DINO in $2\times224^2 + 10\times96^2$ crop setting reaches 74.6% in 24 hours only. This is an improvement of +2% while requiring $2\times$ less time, though the memory usage is higher (15.4G versus 9.3G).

---

## Table 8 전체 재현 (실제 수치)

실험 조건: **ViT-S/16**, 8-GPU 머신 2대(= GPU 16장), batch size 1024. top-1은 ImageNet val **linear evaluation**. "mem."은 **GPU 1장당 peak memory**.

| # | crop 설정 | 100 ep top-1 | 100 ep time | 300 ep top-1 | 300 ep time | mem. |
|---|---|---|---|---|---|---|
| (a) | $2\times224^2$ | 67.8 | 15.3h | **72.5** | **45.9h** | **9.3G** |
| (b) | $2\times224^2 + 2\times96^2$ | 71.5 | 17.0h | 74.5 | 51.0h | 10.5G |
| (c) | $2\times224^2 + 6\times96^2$ | 73.8 | 20.3h | 75.9 | 60.9h | 12.9G |
| (d) | $2\times224^2 + 10\times96^2$ | **74.6** | **24.2h** | 76.1 | 72.6h | **15.4G** |

### 카드 수치가 표의 어느 칸인가

| 카드의 표현 | Table 8 위치 |
|---|---|
| "multi-crop 없이 46시간 학습 시 72.5%" | **(a)행의 300 epochs 칸** — 45.9h / 72.5 (46시간은 45.9h의 반올림) |
| "$2\times224^2+10\times96^2$가 24시간 만에 74.6%" | **(d)행의 100 epochs 칸** — 24.2h / 74.6 |
| "메모리 9.3G → 15.4G" | **(a)행 mem. → (d)행 mem.** |

핵심은 **서로 다른 epoch 수를 비교한 것**이라는 점이다. 같은 wall-clock 예산(약 하루)에서 multi-crop은 100 epoch밖에 못 돌지만, multi-crop 없는 설정이 3배 많은 300 epoch을 돌린 결과보다 여전히 더 좋다. 논문 표현대로 *"the performance boost brought with multi-crop cannot be caught up by more training in the $2\times224^2$ setting"* — 이는 단순한 연산 효율 문제가 아니라 **"local-to-global" 대응 학습 자체의 가치**를 보여준다.

부수적으로 표에서 읽히는 것:

- **수확 체감**: 300 ep에서 $6\times96^2 \to 10\times96^2$는 $75.9 \to 76.1$로 **+0.2%뿐**. 반면 100 ep에서는 $73.8 \to 74.6$로 +0.8% → 오래 학습할수록 crop 추가의 이득이 줄어든다.
- 최고 성능 76.1%는 (d) 300 ep = 72.6h ≈ **8-GPU 서버 2대로 3일**.
- (a)의 300 ep 값 72.5%는 Table 7 row 4 / Table 14 row 4의 "multi-crop 제거" 수치와 동일하다.

---

## 왜 작은 crop을 많이 넣는 게 "시간당 성능"에서 유리한가

### 1) $96^2$ crop은 픽셀 기준 약 $0.18$배

$$\frac{96^2}{224^2}=\left(\frac{96}{224}\right)^2=\left(\frac{3}{7}\right)^2\approx 0.1837$$

### 2) ViT 토큰 수 기준으로도 거의 같은 비율 (patch 16)

$$N_{224}=\left(\frac{224}{16}\right)^2+1=14^2+1=197,\qquad N_{96}=\left(\frac{96}{16}\right)^2+1=6^2+1=37$$

$$\frac{N_{96}}{N_{224}}=\frac{37}{197}\approx 0.188$$

즉 local crop 한 장의 forward 비용은 global crop의 **1/5 수준**. 뒤집어 말하면 **global crop 한 장 값으로 local crop 5장**을 넣을 수 있다.

### 3) attention은 $O(N^2)$이라 절약이 더 커진다

Transformer 블록의 비용은 토큰 선형 항(MLP, QKV projection)과 attention의 제곱 항으로 나뉜다.

- 선형 항: $37/197\approx 0.19\times$
- attention 항: $\dfrac{37^2}{197^2}=\dfrac{1369}{38809}\approx 0.035\times$ → 약 **28배 저렴**

따라서 해상도를 낮추면 비용이 픽셀 비율보다 **더 빠르게** 떨어진다. 실제로 Table 8의 측정값이 이 예측과 맞는다:

- Table 8 시간은 epoch 수에 정확히 선형이다: (a) $45.9/300 = 15.3/100 = 0.153$ h/ep, (d) $72.6/300 = 24.2/100 = 0.242$ h/ep.
- local crop 1장당 추가 시간(100 ep 기준): $(17.0-15.3)/2=0.85$, $(20.3-15.3)/6=0.83$, $(24.2-15.3)/10=0.89$ h → 거의 **일정하게 crop당 약 0.86h**.
- 순수 토큰 선형 모델로 예측하면 crop당 약 1.08h가 나오는데(아래 계산), 실측은 0.86h로 **더 싸다**. 이 차이가 바로 attention 제곱 항에서 오는 추가 절약분이다.

### 4) "같은 GPU 시간에 더 많은 비교 항"

DINO 손실은 **teacher의 global view**와 **student의 모든 view** 쌍에 대한 교차 엔트로피 합이다:

$$\min_{\theta_s}\sum_{x\in\{x_1^g,\,x_2^g\}}\ \sum_{x'\in V,\ x'\neq x} H\big(P_t(x),\,P_s(x')\big)$$

- $2\times224^2$: teacher view 2개 × student view 2개 − 자기쌍 2개 = **2개 항**
- $2\times224^2+10\times96^2$: teacher 2개 × student 12개 − 자기쌍 2개 = **22개 항**

즉 **비교 항이 11배**로 늘어나는데 **시간은 1.58배**($24.2/15.3$)만 늘어난다.

$$\frac{\text{비교 항}}{\text{시간}}:\quad \frac{2}{15.3}=0.13 \ \longrightarrow\ \frac{22}{24.2}=0.91\ \ (\approx 7\times)$$

**GPU 시간당 학습 신호가 약 7배**가 되는 셈이고, 이게 "24시간 100 epoch이 46시간 300 epoch을 이긴다"의 정체다.

### 5) 시간 1.58배의 근거 (연산량 추정)

student는 forward+backward(≈ forward의 3배), teacher는 forward만(그리고 teacher에는 **global view만** 통과 → 설정과 무관하게 고정 $2\times197$).

이미지당 토큰 수:

| 설정 | student 토큰 | teacher 토큰 | 가중 비용 $3S+T$ |
|---|---|---|---|
| $2\times224^2$ | $2\cdot197=394$ | $394$ | $3(394)+394=1576$ |
| $+10\times96^2$ | $394+10\cdot37=764$ | $394$ | $3(764)+394=2686$ |

$$\frac{2686}{1576}\approx 1.70\quad\text{(예측)}\qquad \text{vs}\qquad \frac{24.2}{15.3}\approx 1.58\quad\text{(실측)}$$

실측이 예측보다 낮은 이유는 위 (3)의 attention 제곱 항 절약 + 작은 crop들이 배치로 묶여 처리되는 커널 효율 때문이다. 반대로 student 토큰만 보면 $764/394\approx1.94$배라 실측(1.58배)보다 과대 추정된다 — **teacher는 local crop을 아예 안 보므로 비용이 늘지 않는다**는 점이 트레이드오프를 유리하게 만드는 숨은 요인이다.

---

## 메모리는 왜 늘어나는가 (9.3G → 15.4G)

시간은 "순차적으로 더 많이 계산하면 되는" 자원이지만, **메모리는 동시에 살아 있어야 하는 자원**이다.

- backward를 위해 student의 **모든 view의 activation을 동시에 유지**해야 한다. 이미지당 student 토큰이 $394 \to 764$ (약 $1.94\times$)로 늘면 저장할 activation 총 토큰 수도 그만큼 늘어난다.
- 파라미터, optimizer state(AdamW: 파라미터당 momentum 2개), teacher activation(gradient 불필요, global view만) 등 **crop 수와 무관한 고정분**이 있어서 전체 메모리는 $1.94\times$가 아니라 $15.4/9.3\approx 1.66\times$에 그친다.
- Table 8의 메모리는 local crop 수 $n$에 거의 완벽히 선형이다:

$$\text{mem}(n)\approx 9.3 + 0.61\,n\ \text{[GB]}$$

검산: $n=2 \Rightarrow 10.5$, $n=6 \Rightarrow 12.9$, $n=10 \Rightarrow 15.4$ (모두 표와 일치). → **local crop 1장당 약 0.6GB**, 고정분 9.3GB.

즉 multi-crop은 **"시간을 메모리로 바꾸는" 거래**다. 연산은 싸게 늘리지만 그 activation을 한꺼번에 들고 있어야 한다.

---

## 트레이드오프를 어떻게 조절하는가

Table 8이 사실상 "다이얼 눈금표"다.

1. **local crop 개수 $n$** — 가장 직접적인 다이얼. $n\in\{0,2,6,10\}$에 대해 시간은 $+0.86n$ h/100ep, 메모리는 $+0.61n$ GB로 거의 선형 증가.
   - 메모리 16GB(V100) 한 장에 겨우 들어가는 상황이면 $n=6$(12.9G)이 안전한 타협점: 300 ep에서 75.9%로 최고치 76.1%와 **0.2% 차이**뿐인데 메모리는 2.5GB, 시간은 11.7h 절약.
   - 메모리가 더 빡빡하면 $n=2$(10.5G)만으로도 300 ep 74.5% — multi-crop 없는 72.5%보다 +2%.
   - (공개 구현 `main_dino.py`의 `--local_crops_number` 기본값은 8로, 표의 (c)와 (d) 사이에 해당한다.)
2. **local crop 해상도** — $96^2$을 낮추면 토큰 수가 $(\text{res}/16)^2+1$로 줄어 시간·메모리 모두 감소. 단 너무 작아지면 local view가 담는 의미 정보가 사라져 "local-to-global" 신호가 약해진다.
3. **crop scale 범위 $s$** — global은 $(s,1)$, local은 $(0.05, s)$에서 샘플링(`RandomResizedCrop`). 논문 실험에서 최적은 $s\approx 0.3$ (SwAV의 0.14보다 큼). 비용은 그대로인데 성능이 바뀌는, **공짜 다이얼**.
4. **patch 크기** — 성능/처리량 트레이드오프의 또 다른 축이며, 방향은 multi-crop과 반대다(정확도↑, throughput↓). $8\times8$ patch는 180 im/s, $5\times5$는 44 im/s까지 떨어진다.

![Figure 5: patch 크기에 따른 정확도-처리량 트레이드오프](fig-1.jpeg)

*Figure 5 (§5.1): ViT-S/ViT-B의 patch 크기별 throughput 대비 $k$-NN 정확도(300 epochs). multi-crop이 "작은 입력을 많이"로 시간당 성능을 얻는 반면, 작은 patch는 "토큰을 많이"로 정확도를 사서 처리량을 잃는다 — 두 다이얼은 반대 방향이다.*

---

## 함께 기억할 맥락

- multi-crop은 프레임워크에 따라 효과가 천차만별이다(Appendix E, ViT-S/16 300 ep, $2\times224^2$ → $+6\times96^2$ linear):

| 방법 | multi-crop 없음 (linear) | multi-crop 있음 (linear) | 변화 |
|---|---|---|---|
| DINO | 72.5 | 75.9 | **+3.4** |
| MoCo-v2 | 71.6 | 73.4 | +1.8 |
| SwAV | 68.5 | 71.8 | +3.3 |
| BYOL | 71.4 | 64.8 | **−6.6** (out-of-the-box로 동작하지 않음) |

  → multi-crop은 아무 방법에나 붙이면 되는 "add-on"이 아니라 **모델의 코어 컴포넌트**이며, DINO가 그 이득을 가장 크게 본다.
- 따라서 Table 8의 시간 이득은 "연산이 싸서" 만으로 설명되지 않는다. 값싼 local view가 만들어내는 **local-to-global 대응**이라는 학습 신호 자체가 DINO에서 특히 잘 먹히기 때문이다.
