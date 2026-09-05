# crop 리스트가 해상도별로 "연속" 정렬돼야 하는 이유

## 한 줄 답

`MultiCropWrapper.forward`가 `torch.unique_consecutive`로 **인접한 같은 해상도끼리만** 묶어 backbone을 호출하기 때문이다. 정렬이 깨지면 그룹이 잘게 쪼개져 backbone forward 횟수가 늘어난다 — 그런데 **에러도 안 나고 결과 텐서도 정상**이라, 학습은 그냥 조용히 느려진다.

---

## 1. 문제의 코드

`/home/sungwoo/projects/swcho/dino/utils.py` 의 `MultiCropWrapper.forward` (609–628행):

```python
def forward(self, x):
    if not isinstance(x, list):
        x = [x]
    idx_crops = torch.cumsum(torch.unique_consecutive(
        torch.tensor([inp.shape[-1] for inp in x]),
        return_counts=True,
    )[1], 0)
    start_idx, output = 0, torch.empty(0).to(x[0].device)
    for end_idx in idx_crops:
        _out = self.backbone(torch.cat(x[start_idx: end_idx]))   # 그룹 단위 forward
        if isinstance(_out, tuple):
            _out = _out[0]
        output = torch.cat((output, _out))
        start_idx = end_idx
    return self.head(output)                                     # head는 마지막에 한 번만
```

읽는 순서는 이렇다.

1. crop 리스트에서 **마지막 차원(width)** 만 뽑아 키 텐서를 만든다 → `[224,224,96,...,96]`
2. `unique_consecutive(..., return_counts=True)[1]` 로 **연속 구간 길이**를 얻는다 → `counts`
3. `cumsum` 으로 슬라이스 경계를 만든다 → `idx_crops`
4. 경계마다 `torch.cat` 으로 붙여서 backbone을 **그룹 수만큼만** 호출한다

즉 **backbone forward 횟수 = `len(counts)` = 연속 구간의 개수**다. 이 값이 곧 성능이다.

---

## 2. `torch.unique_consecutive` vs `torch.unique`

이게 전부의 원인이다.

| | 동작 | `[224,96,224,96,...]` 입력 시 |
|---|---|---|
| `torch.unique` | **전역**으로 중복 제거 (정렬 후 유일값) | uniq `[96,224]`, counts `[8,2]` — 순서 무관 |
| `torch.unique_consecutive` | **인접해서 같은 값**인 구간만 묶음 | uniq `[224,96,224,96]`, counts `[1,1,1,7]` — 순서에 민감 |

`unique`를 썼다면 순서가 어떻든 그룹은 2개였을 것이다. 하지만 그러면 `x[start:end]` 슬라이스로 crop을 모을 수 없다 — **연속 슬라이스로 잘라야** `torch.cat` 한 번에 배치를 만들 수 있으므로, DINO는 의도적으로 `unique_consecutive` + "리스트가 정렬돼 있다"는 **암묵적 계약**을 택했다. 속도를 위해 안전장치를 포기한 설계다.

---

## 3. 정상 흐름: `[224,224,96×8]` → counts `[2,8]` → cumsum `[2,10]`

```python
import torch

def groups(res):
    t = torch.tensor(res)
    u, c = torch.unique_consecutive(t, return_counts=True)
    print(f"{res}\n  uniq={u.tolist()} counts={c.tolist()} "
          f"cumsum={torch.cumsum(c,0).tolist()} → forward {len(c)}회")

groups([224, 224] + [96] * 8)                      # 정렬됨
groups([224, 96, 224, 96, 96, 96, 96, 96, 96, 96]) # 섞임
groups([96] * 8 + [224, 224])                      # 오름차순 정렬도 OK
```

실제 출력:

```
[224, 224, 96, 96, 96, 96, 96, 96, 96, 96]
  uniq=[224, 96] counts=[2, 8] cumsum=[2, 10] → forward 2회
[224, 96, 224, 96, 96, 96, 96, 96, 96, 96]
  uniq=[224, 96, 224, 96] counts=[1, 1, 1, 7] cumsum=[1, 2, 3, 10] → forward 4회
[96, 96, 96, 96, 96, 96, 96, 96, 224, 224]
  uniq=[96, 224] counts=[8, 2] cumsum=[8, 10] → forward 2회
```

정렬된 경우 `cumsum = [2,10]` 이므로 슬라이스는 `x[0:2]`(224 두 장)과 `x[2:10]`(96 여덟 장), backbone 호출은 **2회**다. `B=4` 라면 실제로 들어가는 텐서는

$$
\underbrace{(8, 3, 224, 224)}_{2 \times B}, \qquad \underbrace{(32, 3, 96, 96)}_{8 \times B}
$$

두 개뿐이다. 워크스루 §5 의 `register_forward_pre_hook` 실험이 정확히 이걸 찍어서 보여준다 — "crop 10개가 아니라 2회".

> 세 번째 예시가 중요하다: 계약은 "**내림**차순"이 아니라 "**같은 해상도끼리 연속**"이다. `[96×8, 224×2]` 도 forward 2회로 똑같이 최적이다. (워크스루 §14 함정 5번의 "내림차순"은 DINO 기본 구현의 관례를 말하는 것이고, 그룹핑 이득의 실제 조건은 연속성이다.)

---

## 4. 섞였을 때: 그룹이 쪼개지는 과정

키 텐서를 왼쪽부터 스캔하며 "이전 값과 같으면 카운트 +1, 다르면 새 그룹"이라고 생각하면 된다.

```
[224, 96, 224, 96, 96, 96, 96, 96, 96, 96]
  224            → 새 그룹 (count 1)
       96        → 값 바뀜 → 새 그룹 (count 1)
           224   → 값 바뀜 → 새 그룹 (count 1)
                96 96 96 96 96 96 96 → 같은 값 연속 → 한 그룹 (count 7)
counts = [1, 1, 1, 7]  →  cumsum = [1, 2, 3, 10]  →  forward 4회
```

배치 크기 $B$ 기준 backbone에 들어가는 텐서가 `(B,3,224,224)`, `(B,3,96,96)`, `(B,3,224,224)`, `(7B,3,96,96)` 넷으로 흩어진다. GPU는 작은 커널을 여러 번 부르는 걸 싫어하므로, **연산량(FLOPs)은 동일한데 처리량만 떨어진다.**

극단으로 가면 **인접한 두 crop의 해상도가 항상 다른 배치**에서 `counts = [1,1,...,1]` 이 되어 forward가 crop 개수 그대로 **10회**가 된다. 다만 그건 개수 구성에 달려 있다:

| crop 리스트 | forward 횟수 |
|---|---|
| `[224,224,96×8]` (정렬) | 2 |
| `[224,96,224,96×7]` | 4 |
| 2 global + 8 local 무작위 셔플 | 평균 4.2, 최대 5 |
| `[224,96,224,96,...]` 완전 교차 (5 global + 5 local 같은 균형 구성) | 10 |

2+8 구성에서는 224가 두 장뿐이라 최대 5그룹까지만 쪼개지지만, 그것만으로도 **backbone 호출이 2회 → 5회로 2.5배**다. 학습의 대부분 시간이 backbone forward/backward이므로 그대로 wall-clock에 얹힌다.

---

## 5. 왜 "조용한" 실패인가

순서를 섞어도 **아무것도 터지지 않는다.** 이유는 셋이다.

1. **출력 행 순서가 보존된다.** 그룹은 항상 원본 리스트의 *연속 슬라이스*이고, 루프는 `start_idx` 를 왼쪽에서 오른쪽으로 진행하며 `output = torch.cat((output, _out))` 로 append한다. 따라서 어떻게 쪼개지든 `output` 의 $k$ 번째 블록은 `x[k]` 의 특징이다.
2. **최종 shape이 동일하다.** $(2+N)\times B$ 행, 각 `out_dim` 차원. `head` 는 어차피 마지막에 concat된 전체를 한 번 통과하므로(convnet 설정에서 head BatchNorm 통계가 모든 crop에 걸쳐 잡히게 하려는 의도) 그룹 분할과 무관하다.
3. **손실 계산이 그대로 맞는다.** `DINOLoss.forward` 는 `student_output.chunk(ncrops)`, `teacher_output.chunk(2)` 로 행을 나누는데, 행 순서가 보존되므로 chunk 대응도 보존된다.

그래서 loss 곡선도, 최종 정확도도, 로그도 정상이다. **틀린 건 오직 초당 iteration 수**뿐이다. 이런 버그는 "언제부터 느려졌지?"라는 질문으로만 발견되기 때문에 몇 주를 갈 수 있다.

> **주의할 예외 하나**: 그룹핑 키는 `inp.shape[-1]`, 즉 **width만** 본다. width는 같고 height가 다른 crop(예: `96×224` 와 `224×224`)을 인접시키면 같은 그룹으로 묶여 `torch.cat` 이 **런타임 에러**를 낸다. 정사각 crop만 쓰는 기본 설정에서는 안 만나지만, 비정사각 증강을 넣을 때는 조용하지 않게 터진다.

---

## 6. 계약을 지키는 쪽: `DataAugmentationDINO.__call__`

`/home/sungwoo/projects/swcho/dino/main_dino.py` (457–463행):

```python
def __call__(self, image):
    crops = []
    crops.append(self.global_transfo1(image))   # 224, blur p=1.0
    crops.append(self.global_transfo2(image))   # 224, blur p=0.1 + solarize p=0.2
    for _ in range(self.local_crops_number):
        crops.append(self.local_transfo(image)) # 96 × 8
    return crops
```

`append` 순서 자체가 계약의 이행부다. global 224 두 개를 **먼저** 넣고 local 96 여덟 개를 뒤에 몰아넣기 때문에 키 텐서가 `[224,224,96×8]` 로 나오고, `unique_consecutive` 가 `counts=[2,8]` 을 잡는다. 데이터 증강 코드와 모델 래퍼 코드가 파일도 클래스도 다른데 **순서라는 문서화되지 않은 규약으로 결합**되어 있다 — 이게 워크스루 §3 말미가 "암묵적 계약"이라 부르는 지점이다.

그리고 이 순서는 **또 다른, 더 무서운 계약**도 동시에 떠받치고 있다. `train_one_epoch` (318–319행):

```python
teacher_output = teacher(images[:2])   # global 2개만 교사에 통과
student_output = student(images)       # 전체 10개
```

`images[:2]` 는 "앞의 두 개가 global"임을 하드코딩한 것이다. 두 계약을 구분해서 기억하는 게 좋다.

| 깨지는 계약 | 증상 |
|---|---|
| **연속성** (같은 해상도 인접) | 조용한 성능 저하. 결과·손실 모두 정상 |
| **위치** (앞 2개가 global) | 교사가 local crop을 보게 됨 → local→global 예측이라는 DINO의 핵심 비대칭이 무너져 **학습 자체가 망가진다** (에러는 여전히 안 남) |

예를 들어 리스트를 `[96×8, 224×2]` 로 뒤집으면 그룹핑 이득은 그대로 2회로 유지되지만(연속성 OK), `images[:2]` 가 local 두 장을 집어 교사에 넣게 되어 두 번째 계약이 깨진다. 반대로 `[224, 96, 224, 96×7]` 은 위치 계약은 절반만 지키고 연속성은 깨뜨린다. **"global 2개를 먼저, 그다음 local 전부"** 라는 한 가지 순서가 두 계약을 동시에 만족시키는 유일하게 안전한 형태다.

---

## 7. 정리

- 그룹핑 키: `inp.shape[-1]` (width)
- `unique_consecutive` 는 **인접** 중복만 묶으므로, forward 횟수 = 연속 구간 개수
- `[224,224,96×8]` → counts `[2,8]` → cumsum `[2,10]` → 슬라이스 `x[0:2]`, `x[2:10]` → **2회**
- 섞이면 counts가 잘게 갈라져 최대 crop 개수만큼 forward (2+8 구성은 최대 5회, 균형 구성이면 10회)
- 출력 순서·shape·손실이 모두 보존되므로 **에러 없는 성능 저하**, 곧 조용한 실패
- 계약 이행부는 `DataAugmentationDINO.__call__` 의 `append` 순서 (global 2개 먼저)
- 별개 계약인 `images[:2]` (교사는 global만) 는 순서가 깨지면 성능이 아니라 **학습 정확성**을 잃는다
