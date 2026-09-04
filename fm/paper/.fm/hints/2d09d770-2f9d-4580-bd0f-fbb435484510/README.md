# multi-crop을 반영한 DINO의 최종 손실 함수

## 1. 출발점: 라벨 없는 지식 증류 (식 (2))

DINO는 student $g_{\theta_s}$ 와 teacher $g_{\theta_t}$ 가 같은 이미지에 대해 같은 확률분포를 내놓도록 학습한다. 두 네트워크는 $K$ 차원 출력을 softmax로 정규화해 확률분포 $P_s, P_t$ 를 만든다 (식 (1), temperature $\tau_s, \tau_t$).

$$P_s(x)^{(i)} = \frac{\exp\left(g_{\theta_s}(x)^{(i)}/\tau_s\right)}{\sum_{k=1}^{K}\exp\left(g_{\theta_s}(x)^{(k)}/\tau_s\right)}$$

teacher를 고정한 상태에서 student 파라미터에 대해 cross-entropy를 최소화하는 것이 논문 식 (2)이다.

$$\min_{\theta_s} H\big(P_t(x), P_s(x)\big), \qquad (2)$$

여기서 $H(a,b) = -a\log b$ 이다.

식 (2)는 **같은 입력 $x$** 를 두 네트워크에 넣는다. 이 상태로는 self-supervised 신호가 없으므로(teacher가 student의 EMA이니 자명하게 만족될 수 있다), 논문은 여기에 multi-crop augmentation을 붙인다.

![DINO 구조: 두 view를 student/teacher에 넣고 cross-entropy로 맞춘다](fig-1.jpeg)

*Figure 2 (논문). 그림은 단순화를 위해 view 한 쌍 $(x_1, x_2)$ 만 그렸다. teacher 쪽에는 centering + sharpening과 stop-gradient(sg)가 걸리고, teacher 파라미터는 student의 EMA로 갱신된다. 실제 학습에서는 이 구조가 아래 식 (3)의 18개 항으로 확장된다.*

## 2. 논문 §3.1 식 (3): multi-crop 손실

> "from a given image, we generate a set $V$ of different views. This set contains two *global* views, $x_1^g$ and $x_2^g$ and several *local* views of smaller resolution. All crops are passed through the student while only the *global* views are passed through the teacher, therefore encouraging 'local-to-global' correspondences. We minimize the loss:"

$$\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \;\; \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big). \qquad (3)$$

> "This loss is general and can be used on any number of views, even only 2."

### 각 기호의 정확한 의미

| 기호 | 의미 |
|---|---|
| $V$ | 한 장의 원본 이미지에서 만든 **모든 view의 집합** |
| $x_1^g, x_2^g$ | $224^2$ 해상도의 **global view 2개**. 원본의 넓은 영역(예: 50% 초과)을 덮는다 |
| local views | $96^2$ 해상도의 **작은 crop들**. 원본의 좁은 영역(예: 50% 미만)만 덮는다 |
| 바깥 합 $x \in \{x_1^g, x_2^g\}$ | **teacher에 들어가는 view**. global view만 teacher를 통과한다 |
| 안쪽 합 $x' \in V,\ x' \neq x$ | **student에 들어가는 view**. 모든 crop이 student를 통과한다 |
| $H(P_t(x), P_s(x'))$ | teacher가 view $x$ 에서 낸 분포를 정답 삼아, student가 view $x'$ 에서 낸 분포에 매기는 cross-entropy |

핵심 비대칭: **student는 전부, teacher는 global만.** 이 비대칭이 "작은 부분만 보고도 전체를 본 표현을 맞춰라"는 **local-to-global correspondence**를 만든다.

## 3. $V$ 의 크기와 항의 개수를 실제로 세어보기

DINO의 기본 설정(공식 구현 `main_dino.py`의 `--local_crops_number` 기본값 8)은

$$|V| = \underbrace{2}_{\text{global } 224^2} + \underbrace{8}_{\text{local } 96^2} = 10.$$

이제 식 (3)의 항 수를 센다.

- **바깥 합**: $x$ 가 $x_1^g$, $x_2^g$ 2가지 → **2개**
- **안쪽 합**: $x'$ 는 $V$ 의 10개 중 $x$ 자신 하나를 뺀 → **9개**

$$\text{항의 개수} = 2 \times 9 = \boxed{18}$$

18개 항의 내역을 풀어 쓰면:

| teacher view $x$ | student view $x'$ | 항 수 | 종류 |
|---|---|---|---|
| $x_1^g$ | $x_2^g$ | 1 | global→global |
| $x_1^g$ | $l_1, \dots, l_8$ | 8 | **local→global** |
| $x_2^g$ | $x_1^g$ | 1 | global→global |
| $x_2^g$ | $l_1, \dots, l_8$ | 8 | **local→global** |
| | 합계 | **18** | |

즉 18항 중 2항이 global끼리의 대칭 쌍이고, 나머지 16항이 multi-crop이 새로 추가한 local-to-global 항이다. (부록 E의 일부 실험처럼 local crop을 6개로 쓰면 $|V| = 8$, 항의 수는 $2 \times 7 = 14$가 된다. 논문 Tab. 8에는 $2\times224^2 + 10\times96^2$ 설정도 나오며 이때는 $2\times11 = 22$항이다.)

## 4. 왜 $x' \neq x$ 조건이 필요한가

$x' = x$ 를 허용하면 그 항은

$$H\big(P_t(x), P_s(x)\big)$$

가 되어 **완전히 같은 crop을 teacher와 student에 넣고 비교하는 자명한 항**이 된다. 이것이 나쁜 이유는:

1. **학습 신호가 없다.** teacher는 student의 EMA(지수이동평균)이므로 두 네트워크는 거의 같은 함수다. 같은 입력을 주면 출력도 거의 같고, 손실은 이미 최솟값 근처라 gradient가 거의 0이다. 계산만 낭비된다.
2. **augmentation invariance를 학습하지 못한다.** DINO가 배우려는 것은 "같은 이미지의 *서로 다른* 왜곡·crop이 같은 표현을 갖는다"는 성질이다. 똑같은 입력끼리 맞추는 항은 이 성질에 대해 아무것도 가르치지 않는다.
3. **붕괴(collapse) 쪽으로 밀 수 있다.** 자명 항은 "출력을 아무 상수로나 두어도 만족되는" 방향의 항이므로, centering/sharpening으로 막으려는 붕괴를 오히려 도와준다.

여기서 주의할 점 하나: $x' \neq x$ 는 **crop 인스턴스가 다르다**는 뜻이지, "픽셀 내용이 반드시 다르다"는 뜻이 아니다. $x_1^g$ 와 $x_2^g$ 는 둘 다 global이지만 서로 다른 랜덤 crop + 서로 다른 색 왜곡/블러를 거친 다른 view이므로 $x_1^g \neq x_2^g$ 이고, 따라서 (teacher $x_1^g$, student $x_2^g$) 항은 살아남는다.

공식 구현에서는 이 조건이 인덱스 비교로 그대로 나타난다 (`/home/sungwoo/projects/swcho/dino/main_dino.py`, `DINOLoss.forward`):

```python
for iq, q in enumerate(teacher_out):          # teacher: global 2개 (chunk(2))
    for v in range(len(student_out)):         # student: 전체 crop (chunk(ncrops))
        if v == iq:
            # we skip cases where student and teacher operate on the same view
            continue
        loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        total_loss += loss.mean()
        n_loss_terms += 1
total_loss /= n_loss_terms
```

crop 리스트가 `[global_1, global_2, local_1, ..., local_8]` 순서이므로 `v == iq`가 정확히 $x' = x$ 인 두 경우(0-0, 1-1)를 걸러낸다. 그리고 `n_loss_terms`(=18)로 나누므로, 실제 구현의 손실은 식 (3)의 **합이 아니라 평균**이다. 최적화 지점은 같고 학습률 스케일만 view 수에 무관하게 유지된다.

## 5. "2개 view만으로도 동작한다"의 의미

논문은 식 (3) 직후에 "This loss is general and can be used on any number of views, even only 2"라고 못박는다. 즉 식 (3)은 multi-crop 전용 공식이 아니라 **일반형**이고, view 수를 줄이면 기존 방법들의 표준 손실로 자연스럽게 **환원(reduce)** 된다.

$V = \{x_1^g, x_2^g\}$ 인 경우($|V| = 2$, local crop 0개)를 식 (3)에 대입하면 항의 수는 $2 \times 1 = 2$ 이고

$$\min_{\theta_s}\; H\big(P_t(x_1^g), P_s(x_2^g)\big) + H\big(P_t(x_2^g), P_s(x_1^g)\big)$$

이 된다. 이것이 바로 BYOL/SimSiam 계열이 쓰는 **표준 2-view 대칭 손실**이다. 논문 Algorithm 1의 "DINO PyTorch pseudocode w/o multi-crop"이 정확히 이 형태다.

```python
loss = H(t1, s2)/2 + H(t2, s1)/2
```

(1/2는 위 §4의 `n_loss_terms`로 나누는 평균화에 해당한다. 참고로 논문에서 번호가 붙은 식 (4)는 이 2-view 손실이 아니라 teacher 출력의 center $c$ 를 EMA로 갱신하는 식이다 — 2-view 손실은 번호 없이 Algorithm 1의 의사코드로 제시된다.)

따라서 정리하면:

- 식 (3)은 $|V|$ 에 대한 **일반형**이며, $|V|=2$ 로 두면 표준 대칭 2-view 손실, $|V|=10$ 으로 두면 18항짜리 multi-crop 손실이 된다.
- multi-crop은 손실 함수의 형태를 바꾸는 것이 아니라, **같은 손실에 항을 더 채워 넣는 것**이다.

## 6. 왜 이렇게까지 하는가 (효과)

- 논문 §5.1 ablation(Table 7, row 4·5)에서 multi-crop 제거는 성능을 크게 떨어뜨린다. 부록 E에서 DINO는 multi-crop으로 linear eval **+3.4%** 이득을 보아 비교 프레임워크 중 이득이 가장 크다 (ViT-S/16, 300 epoch: $2\times224^2$ 에서 72.5% → $2\times224^2+6\times96^2$ 에서 75.9%).
- 연산 효율도 좋아진다. local crop은 $96^2$ 라 싸다. Table 8: multi-crop 없이 46시간 학습해 72.5%인 반면, $2\times224^2+10\times96^2$ 설정은 **24시간에 74.6%** — 시간 절반에 +2%. 다만 peak memory는 9.3G → 15.4G로 늘어난다.
- 단, multi-crop은 아무 프레임워크에나 붙이면 되는 "add-on"이 아니다. 부록 E에서 BYOL에 multi-crop을 붙이면 오히려 성능이 떨어진다(66.6 → 59.8 k-NN). 즉 식 (3)의 local-to-global 항이 잘 먹히려면 DINO의 나머지 요소(momentum teacher, centering+sharpening, cross-entropy 손실)와의 조합이 필요하다.

## 7. 한 줄 요약

식 (3)은 "**teacher는 global view만 보고, student는 그것을 제외한 모든 view로 그 출력을 맞춘다**"를 이중 합으로 쓴 것이다. 기본 설정 $|V| = 2+8 = 10$ 에서 항은 $2\times 9 = 18$ 개이며, $x'\neq x$ 는 gradient가 없는 자명한 항을 배제한다. $|V|=2$ 로 줄이면 표준 대칭 2-view 손실로 그대로 환원되므로, 이 손실은 view 수에 대해 일반적이다.
