# `if v == iq: continue` — 같은 view 쌍을 버리는 한 줄

## Q

`MiniDINOLoss`의 이중 루프에서 `if v == iq: continue`가 하는 일은?

## A

teacher와 student가 **같은 view**를 보는 쌍을 건너뛴다. 서로 다른 crop 사이에서만 교차엔트로피를 계산해야 의미 있는 학습이 되기 때문이다.

---

## 1. 루프가 무엇을 도는가

`.fm/assets/dino_collapse_cross_entropy.py`의 `MiniDINOLoss.forward` (원본 `main_dino.py`의 `DINOLoss.forward`와 줄 단위로 동일):

```python
student_out = (student_output / self.student_temp).chunk(self.ncrops)   # 길이 ncrops
teacher_out = F.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
teacher_out = teacher_out.detach().chunk(2)                             # 길이 2 (global만)

total_loss, n_terms = 0.0, 0
for iq, q in enumerate(teacher_out):        # iq ∈ {0, 1}
    for v in range(len(student_out)):       # v  ∈ {0, ..., ncrops-1}
        if v == iq:
            continue                        # ← 이 줄
        loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        total_loss += loss.mean()
        n_terms += 1
total_loss /= n_terms
```

- `iq` — **teacher**가 본 global crop의 인덱스. teacher는 global 2개만 보므로 `chunk(2)`, 즉 `iq ∈ {0, 1}`.
- `v` — **student**가 본 crop의 인덱스. student는 모든 crop을 보므로 `chunk(ncrops)`, 즉 `v ∈ {0, …, ncrops-1}`.
- 한 항은 $H(P_t(\text{crop}_{iq}),\, P_s(\text{crop}_v)) = -\sum_k q_k \log p_k$.

핵심 전제는 **두 chunk 리스트의 앞 2칸이 같은 crop을 가리킨다**는 것이다. 인덱스가 정렬되어 있지 않다면 `v == iq`라는 정수 비교로 "같은 view"를 판정할 수 없다.

## 2. 인덱스 정렬이 코드에서 어떻게 보장되나

세 곳이 맞물려서 보장한다.

**(a) 증강 결과의 순서** — `main_dino.py`의 `DataAugmentationDINO.__call__`이 global 2개를 리스트 맨 앞에 넣는다:

```python
crops = []
crops.append(self.global_transfo1(image))   # index 0
crops.append(self.global_transfo2(image))   # index 1
for _ in range(self.local_crops_number):
    crops.append(self.local_transfo(image)) # index 2 ...
return crops
```

**(b) 같은 리스트를 잘라서 두 네트워크에 넣는다** — `main_dino.py:318-320`:

```python
teacher_output = teacher(images[:2])   # only the 2 global views pass through the teacher
student_output = student(images)
loss = dino_loss(student_output, teacher_output, epoch)
```

`images[:2]`는 student가 받는 `images`의 앞 2개와 **같은 텐서 객체**다. 별도로 증강을 다시 돌리지 않는다.

**(c) `MultiCropWrapper.forward`가 순서를 보존한다** (`utils.py:609-628`):

```python
idx_crops = torch.cumsum(torch.unique_consecutive(
    torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
start_idx, output = 0, torch.empty(0).to(x[0].device)
for end_idx in idx_crops:
    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
    output = torch.cat((output, _out))
    start_idx = end_idx
return self.head(output)
```

해상도가 같은 crop끼리 묶어 배치로 만들지만(224짜리 2개, 96짜리 8개), `torch.cat`을 **리스트 순서대로** 이어 붙이므로 출력 행 순서는 `[g1 배치 | g2 배치 | l1 배치 | … | l8 배치]`가 된다. 그래서 `chunk(ncrops)`의 $i$번째 조각이 $i$번째 crop이 된다. (`unique_consecutive`를 쓰기 때문에 "같은 해상도끼리 연속"이라는 (a)의 순서가 깨지면 이 보장도 깨진다.)

노트북 토이는 같은 구조를 더 단순하게 재현한다 (`train`):

```python
views = torch.cat([augment(X[idx]), augment(X[idx])])   # global crop 2개
s_out = student(views)
with torch.no_grad():
    t_out = teacher(views)
loss = loss_fn(s_out, t_out)
```

`augment`를 두 번 따로 호출했으므로 crop A와 crop B는 서로 다른 노이즈를 받은 서로 다른 view다. 그리고 **student와 teacher가 완전히 동일한 `views` 텐서**를 받는다 → 인덱스 0은 양쪽 모두 crop A, 인덱스 1은 양쪽 모두 crop B.

따라서 `v == iq`는 정확히 **"teacher와 student가 같은 이미지 조각을 봤다"** 를 뜻한다.

## 3. 건너뛰지 않으면 무슨 일이 일어나나

`continue`를 지우면 $x = x'$인 항, 즉

$$H\big(P_t(x),\, P_s(x)\big)$$

가 loss에 들어온다. 논문 식 (2) 그 자체 형태다. 문제는 이 항이 **증강 불변성을 전혀 요구하지 않는다**는 것이다.

- 다른 crop 쌍의 항 $H(P_t(x), P_s(x'))$는 student에게 "$x'$만 보고도 $x$를 본 teacher와 같은 답을 내라"고 요구한다. 이게 표현 학습의 신호다 — 서로 다른 조각을 같은 코드로 매핑해야 하므로 view에 불변인 의미를 잡아야 한다.
- 반면 $H(P_t(x), P_s(x))$는 "같은 입력에 대해 teacher를 그대로 복사하라"만 요구한다. student가 teacher의 함수를 그대로 흉내내면 만족되므로, 입력의 어떤 구조도 배울 필요가 없다.

게다가 DINO에서 teacher는 student의 EMA다 (`main_dino.py:346`, 노트북에서도 `pt.mul_(m).add_((1 - m) * ps)`). 즉 파라미터가 이미 $\theta_t \approx \theta_s$이므로 $P_t(x) \approx P_s(x)$이고, 이 항은 **학습 시작 시점부터 거의 최소값에 앉아 있다**. 정확히는 온도가 다르므로($\tau_t = 0.04$ vs $\tau_s = 0.1$) 0은 아니고 teacher 분포의 엔트로피 근처에 머물지만, gradient가 작고 방향도 "student를 자기 자신의 과거로 되돌려라"에 가깝다.

정리하면 이 항은 **학습 신호가 아니라 희석(dilution)** 이다. 뒤에서 `n_terms`로 나누므로, 쓸모없는 항이 섞이면 진짜 신호가 되는 항들의 가중치가 그만큼 줄어든다. 계산량도 공짜가 아니다 — 건너뛰지 않으면 항이 18개에서 20개로 늘어난다.

## 4. "같은 crop"이 정말 같은 입력인가

원칙적으로는 의심해 볼 만하다. 증강이 stochastic하고 dropout 같은 확률적 레이어가 있으면, 같은 crop 인덱스라도 두 forward의 입력·활성이 달라져 항이 완전히 자명하지는 않을 수 있다.

하지만 이 코드에서는 실제로 동일하다.

- 증강은 **한 번만** 수행되어 `crops` 리스트에 담기고, `images[:2]`는 그 리스트를 슬라이스한 것이다. teacher용으로 증강을 다시 돌리지 않는다.
- 노트북도 마찬가지로 `views`를 한 번 만들어 student/teacher 양쪽에 그대로 넘긴다.
- 노트북의 `mlp`는 `Linear/GELU`만으로 이루어져 dropout이 없다. 원본 DINO의 ViT backbone도 학습 시 stochastic depth(drop path)를 쓰지만 teacher는 `torch.no_grad()` + eval 성격으로 도는 EMA 사본이고, 어차피 이 항의 "student가 teacher를 복사하면 끝"이라는 성질은 바뀌지 않는다.

그러니 "완전히 무의미한 항은 아닐 수도 있다"는 여지는 남지만, 실질적으로는 버리는 게 맞다.

## 5. 항의 개수 산수

이중 루프는 $2 \times \texttt{ncrops}$개의 (iq, v) 쌍을 만들고, 그중 $v = iq$인 쌍 2개(즉 $(0,0)$, $(1,1)$)를 뺀다.

$$n_{\text{terms}} = 2 \cdot \texttt{ncrops} - 2$$

- **기본 설정** (global 2 + local 8, `main_dino.py:112`의 `--local_crops_number` 기본값 8 → `ncrops = 10`, `main_dino.py:217`에서 `local_crops_number + 2`로 전달):
  $$2 \times 10 - 2 = 18 \text{ 항}$$
  내역: global↔global 2항(A→B, B→A) + global→local $2 \times 8 = 16$항.
- **노트북 토이** (`ncrops=2`):
  $$2 \times 2 - 2 = 2 \text{ 항}$$
  살아남는 두 쌍은 $(iq=0, v=1)$과 $(iq=1, v=0)$, 즉 "teacher가 본 crop A → student가 본 crop B"와 "teacher가 본 crop B → student가 본 crop A"의 **양방향** 이다.

논문 식 (3)과 정확히 대응한다:

$$\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \;\; \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)$$

| 논문 | 코드 |
|---|---|
| $x \in \{x_1^g, x_2^g\}$ | `for iq, q in enumerate(teacher_out)` — `chunk(2)`이므로 global 2개 |
| $x' \in V$ | `for v in range(len(student_out))` — `chunk(ncrops)`이므로 전체 crop |
| $x' \neq x$ | `if v == iq: continue` |
| $H(P_t(x), P_s(x'))$ | `torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)` |

즉 이 한 줄이 식 (3)의 $x' \neq x$ 제약을 그대로 구현한 것이다. 논문 본문이 말하는 "local-to-global correspondences"를 강제하는 장치이기도 하다 — 작은 조각만 본 student가 전체를 본 teacher와 같은 답을 내야 한다.

## 6. 대칭성

살아남는 항들은 방향에 대해 대칭이다. global 2개에 대해

$$H(P_t(x_1^g), P_s(x_2^g)) \;+\; H(P_t(x_2^g), P_s(x_1^g))$$

두 방향을 **모두** 넣는다. 한 방향만 넣어도 학습은 되지만, 두 방향을 다 쓰면 배치당 gradient 신호가 두 배가 되고 crop 순서에 대한 임의의 편향이 사라진다.

이건 self-supervised 계열의 공통 관용구다. BYOL은 "we symmetrize the loss by separately feeding $v'$ to the online network and $v$ to the target network", SimSiam은 `L = D(p1, z2)/2 + D(p2, z1)/2`로 같은 일을 한다. DINO의 이중 루프는 이 symmetrized loss를 multi-crop으로 일반화한 것이라고 보면 된다 — global 2개가 각각 teacher 역할을 한 번씩 맡고, 나머지 모든 crop이 student 쪽에 붙는다.

참고로 이 대칭성은 **collapse를 막아주지 않는다**. 노트북 4절이 보여주듯 "입력과 무관하게 항상 같은 원-핫"을 내면 모든 항이 0이 되어 완벽한 점수를 받는다. 붕괴 방지는 centering과 sharpening이 담당하고, 이 루프는 "무엇을 맞출 것인가"만 정한다.

## 7. `n_terms`로 나누는 이유

```python
total_loss /= n_terms
```

항의 개수는 crop 구성에 따라 2개(`ncrops=2`)에서 18개(`ncrops=10`)까지 변한다. 나누지 않으면 **loss와 gradient의 스케일이 crop 수에 비례해 커진다** — local crop을 8개에서 6개로 줄이는 것만으로 실효 learning rate가 바뀌는 셈이다.

`n_terms`로 나누면 loss가 "항 하나당 평균 교차엔트로피"가 되어, crop 구성을 바꿔도

- loss 값이 같은 척도에 머물러 실험 간 비교가 가능하고,
- learning rate / weight decay 스케줄을 재조정할 필요가 없다.

논문이 `--local_crops_number 0`부터 8까지 여러 설정을 같은 하이퍼파라미터로 돌려 비교할 수 있는 이유가 여기 있다. `loss.mean()`이 배치 차원을 평균 내는 것과 같은 취지의 정규화가 crop 차원에도 적용된 것이다.

---

## 한 줄 요약

`if v == iq: continue`는 논문 식 (3)의 $x' \neq x$ 조건을 구현하며, "같은 crop을 보고 teacher를 복사하기"라는 정보가 없는 항 2개를 제거해 $2\cdot\texttt{ncrops}-2$개의 **crop 간 교차** 항만 남긴다.

## 관련 파일

- `/home/sungwoo/projects/swcho/dino/fm/entropy/.fm/assets/dino_collapse_cross_entropy.py` — `MiniDINOLoss` (151-187행), `train` (302-336행)
- `/home/sungwoo/projects/swcho/dino/main_dino.py` — `DINOLoss` (362-415행), teacher/student forward (318-320행), `DataAugmentationDINO.__call__` (457-463행)
- `/home/sungwoo/projects/swcho/dino/utils.py` — `MultiCropWrapper.forward` (609-628행)
- `/home/sungwoo/projects/swcho/dino/paper/2104.14294v2.md` — 식 (2), (3)
