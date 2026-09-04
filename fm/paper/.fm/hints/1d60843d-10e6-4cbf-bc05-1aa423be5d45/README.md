# multi-crop의 스케일 범위 파라미터 $s$

## 한 줄 정리

$s$는 **crop이 원본 이미지 면적에서 차지하는 비율**의 경계값이다. global view 2개는 `RandomResizedCrop(224, scale=(s, 1))`, local view 6개(구현 기본값 8개)는 `RandomResizedCrop(96, scale=(0.05, s))`로 뽑는다. 즉 $s$ 하나가 global 분포의 **하한**과 local 분포의 **상한**을 동시에 정하며, 두 범위는 겹치지 않는다. 논문 부록 E의 스윕 결과 최적값은 약 $0.3$ 근처이고, 이는 SwAV가 쓰는 $0.14$보다 높다.

---

## 1. `RandomResizedCrop(size, scale=(a, b))`이 정확히 하는 일

가장 흔한 오해는 $s$를 **픽셀 길이의 비율**로 읽는 것이다. 아니다. torchvision의 `RandomResizedCrop`은 **면적(area) 비율**을 샘플링한다.

원본 이미지 크기를 $H \times W$라 하면 절차는 다음과 같다.

1. **목표 면적 샘플링**: $\alpha \sim \mathcal{U}(a, b)$를 뽑아 목표 면적을 $A_{\text{target}} = \alpha \cdot HW$로 정한다.
2. **종횡비 샘플링**: 종횡비 $r$을 `ratio` 범위(기본 $(3/4,\ 4/3)$)에서 로그 균등하게 뽑는다. 즉 $\log r \sim \mathcal{U}(\log \tfrac34, \log \tfrac43)$.
3. **crop 크기 결정**: $w = \sqrt{A_{\text{target}} \cdot r}$, $h = \sqrt{A_{\text{target}} / r}$로 반올림한다.
4. **위치 샘플링**: $w \le W$, $h \le H$이면 이미지 안에서 좌상단 위치를 균등하게 뽑아 잘라낸다. (실패하면 최대 10회 재시도하고, 그래도 안 되면 종횡비를 클램프한 **center crop**으로 폴백한다.)
5. **리사이즈**: 잘라낸 영역을 목표 해상도 `size`로 리사이즈한다. DINO는 `interpolation=Image.BICUBIC`을 쓴다.

핵심은 **4번까지가 "어디를 얼마나 넓게 보는가"이고, 5번은 그 영역을 항상 고정 해상도로 맞춘다**는 점이다. 그래서 $s$는 해상도와 무관한 "시야(field of view)" 파라미터다. $224^2$와 $96^2$라는 숫자는 네트워크에 들어가는 텐서 크기일 뿐, crop이 원본에서 얼마나 큰 영역이었는지와는 별개다.

> 감각을 잡기 위한 산수: 전형적인 ImageNet 이미지가 $500 \times 375$($\approx 1.9\times10^5$ px$^2$)라고 하면
> - $\alpha = 0.32$ → 면적 $\approx 6.0\times10^4$ px$^2$ → 대략 $245 \times 245$ 영역. $224^2$로 리사이즈하면 거의 원본 배율.
> - $\alpha = 0.08$ → 면적 $\approx 1.5\times10^4$ px$^2$ → 대략 $122 \times 122$ 영역. $224^2$로 만들려면 **약 1.8배 업샘플**(= 흐릿한 확대).
> - $\alpha = 0.05$ → 대략 $97 \times 97$ 영역 → $96^2$로 리사이즈하면 거의 원본 배율.

---

## 2. $s$가 정하는 경계 구조

| 뷰 | 개수 | `scale` 범위 | 출력 해상도 | 통과 경로 |
|---|---|---|---|---|
| global | 2 | $(s,\ 1)$ | $224^2$ | student **와** teacher 모두 |
| local | 6 (부록 E) / 8 (구현 기본) | $(0.05,\ s)$ | $96^2$ | student만 |

한 개의 스칼라 $s$가 두 분포의 경계를 공유하므로, **$s$를 움직이면 global과 local이 반대 방향으로 함께 움직인다.**

- $s$가 **커지면**: global의 하한이 올라가 global view는 항상 더 넓은 영역($\ge s$)만 보게 되고, local의 상한도 올라가 local view가 꽤 넓은 영역($\le s$)까지 허용된다. 극단적으로 $s \to 1$이면 global은 사실상 전체 이미지 고정, local은 "작은 뷰"라는 성격을 잃는다.
- $s$가 **작아지면**: global view가 원본의 아주 작은 영역까지 포함하게 되고(업샘플된 좁은 패치가 teacher 타깃이 됨), local view는 정말 극단적으로 좁은 영역($0.05 \sim s$)에만 머문다.

> 카드/문헌에서 "$s$가 크면 global이 더 작은 영역까지 본다"처럼 서술된 것을 보면 방향이 뒤집힌 것이다. $(s, 1)$에서 $s$는 **하한**이므로, $s$↑ = global이 더 커진다(= 더 넓은 영역만 본다).

논문은 이 비겹침(non-overlapping) 설계가 필연이 아님을 명시한다:

> "Note that we arbitrarily choose to have non-overlapping scaling range for the global and local views following the original design of SwAV. However, the ranges could definitely be overlapping and experimenting with finer hyperparameters search could lead to a more optimal setting." (부록 E)

이 구조 때문에 DINO의 손실이 "local-to-global" 대응을 학습하게 된다. teacher는 global view만 보고 타깃 분포를 만들고, student는 local view까지 포함해 그 타깃을 맞춘다.

$$\min_{\theta_s} \sum_{x \in \{x_1^g,\, x_2^g\}} \ \sum_{\substack{x' \in V \\ x' \neq x}} H\big(P_t(x),\, P_s(x')\big)$$

![DINO 개요 (multi-crop 없이 뷰 2개만 그린 단순화 버전)](fig-1.jpeg)

*Figure 2 (논문 p.2). 여기서는 $x \to x_1, x_2$ 두 뷰만 그려져 있지만, 실제 학습에서는 $x_2$ 쪽(teacher)에는 global view 2개만, $x_1$ 쪽(student)에는 global 2개 + local 6~8개가 들어간다.*

---

## 3. 부록 E의 $s$ 스윕 (수치 복원함)

ViT-S/16, 100 epoch 사전학습, ImageNet $k$-NN top-1:

| $(0.05, s),\ (s, 1)$의 $s$ | 0.08 | 0.16 | 0.24 | **0.32** | 0.48 |
|---|---|---|---|---|---|
| $k$-NN top-1 | 65.6 | 68.0 | 69.7 | **69.8** | 69.5 |

> **복원 사실 명시**: asset의 `2104.14294v2.md`는 마크다운 변환 과정에서 이 표가 손상되어 `0.08 / 0.16 / 0.24 / 0.32 / 0.45` → `65.6 / 98.0 / 69.7 / 69.8 / 99.2`로 적혀 있다(`98.0`, `99.2`는 ImageNet $k$-NN 값으로 불가능한 수치, 마지막 열 헤더도 `0.45`로 오독). 위 표는 arXiv 원문(ar5iv HTML)에서 확인한 값 — 마지막 열은 $s = 0.48$이고 $k$-NN은 각각 65.6 / 68.0 / 69.7 / 69.8 / 69.5 — 로 복원한 것이다.

읽는 법:

- $s = 0.08$은 명확히 나쁘다(−4.2%p). global view가 원본 면적의 8%까지 내려갈 수 있어 teacher 타깃이 무너지는 구간.
- $0.24 \sim 0.48$은 사실상 평평하다(69.5–69.8). 그래서 논문도 정확한 최적값을 주장하지 않고 "**the optimum to be around 0.3**"이라고만 쓴다. $s$는 예민한 하이퍼파라미터가 아니라 **너무 작지만 않으면 되는** 파라미터다.
- 즉 이 카드에서 외울 것은 "0.3 정확히"가 아니라 **"하한을 충분히 높게 잡아야 한다, 0.08 같은 값은 안 된다"**이다.

---

## 4. 왜 $s \approx 0.3$이 SwAV의 $0.14$보다 높은가

**논문이 실제로 말하는 것은 여기까지다** (부록 E, 원문 인용):

> "In this table, we vary the parameter $s$ that controls the range of scales used in multi-crop and find the optimum to be around 0.3 in our experiments. We note that this is higher than the parameter used in SwAV which is of 0.14."

즉 논문은 **사실만 보고하고 이유는 설명하지 않는다.** 아래는 해석이며, 논문 본문에 명시된 근거가 아니다(단, 코드 기본값이라는 간접 증거는 있다 — 5절 참고).

**해석 (1) teacher 타깃의 품질이 $s$의 하한에 직접 걸려 있다.**
DINO에서 teacher는 **global view만** 본다. teacher의 출력 $P_t$가 곧 student가 맞춰야 할 타깃이므로, global view가 원본의 극히 작은 영역이 되어버리면 "타깃 자체가 이미지 전체에 대한 의미를 담지 못한" 상태가 된다. self-distillation은 라벨이 없어 타깃 품질을 보정해 줄 외부 신호가 없으므로, global의 하한 $s$를 높게 두는 것이 타깃 붕괴를 막는 직접적인 방어다. SwAV는 Sinkhorn-Knopp으로 배치 전체에 걸쳐 균등 분할 제약을 걸어 프로토타입 할당을 정규화하므로, 개별 뷰가 작아 생기는 노이즈에 상대적으로 덜 민감하다.

**해석 (2) ViT는 패치 단위 표현이라 아주 작은 crop에서 남는 구조가 적다.**
ViT-S/16에 들어가는 $224^2$ 입력은 $14 \times 14 = 196$개 패치다. 원본 면적의 8%짜리 영역($\approx 122 \times 122$)을 $224^2$로 **업샘플**해서 넣으면 패치 토큰들은 보간으로 만들어진 저주파 정보를 나눠 갖게 되고, 패치 간 self-attention이 잡을 만한 실제 구조(경계, 부분-전체 관계)가 희박해진다. convnet은 지역 필터를 계단식으로 쌓아 저해상도 입력에서도 비교적 잘 버티지만, 패치 임베딩은 crop이 뭉개지면 그 손실이 첫 층에서 바로 고정된다. 논문이 본문에서 "small patches with ViTs"의 중요성을 강조하는 것과 방향이 같은 이야기다(그 서술 자체는 patch size에 관한 것이고 $s$에 관한 것은 아님).

**해석 (3) local view의 성격 차이.**
$s$가 local의 상한이기도 하므로 $s$↑는 local view도 덜 극단적으로 만든다. $96^2$ 해상도에 5% 면적($\approx 97 \times 97$)이면 거의 원본 배율이지만, 32% 면적($\approx 245 \times 245$)을 $96^2$로 줄이면 2.5배 다운샘플이다. $s \approx 0.3$은 local view가 "원본 배율의 작은 조각 ~ 축소된 중간 크기 영역"을 폭넓게 커버하게 만들어, local-to-global 과제를 적당한 난이도로 유지한다.

---

## 5. 구현 기본값과 논문 서술의 차이 (반드시 짚어둘 것)

`main_dino.py`의 기본값은 부록 E 서술과 두 군데에서 다르다.

| 항목 | 논문 부록 E | 코드 기본값 (`main_dino.py`) |
|---|---|---|
| local view 개수 | **6**개 | `--local_crops_number 8` → **8**개 |
| $s$ | 최적값 "약 0.3" | `--global_crops_scale 0.4 1.` / `--local_crops_scale 0.05 0.4` → **$s = 0.4$** |

```python
# main_dino.py
parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.))
parser.add_argument('--local_crops_number', type=int, default=8)
parser.add_argument('--local_crops_scale',  type=float, nargs='+', default=(0.05, 0.4))
```

- **local 개수 6 vs 8**: 논문 Table 8(compute requirements)은 $2\times224^2$, $+2\times96^2$, $+6\times96^2$, $+10\times96^2$를 비교하며 $6\times96^2$에서 300 epoch linear 75.9%를 보고한다. 부록 E의 프레임워크 비교 표도 "two $224^2$ crops and six $96^2$ crops"로 통일해 공정 비교를 한다. 반면 공개 코드의 기본 학습 설정은 8개다. **논문 표의 설정 = 6, 릴리스된 기본 레시피 = 8**로 기억하면 된다(Table 8이 보여주듯 6→10 사이의 차이는 작고, 개수를 늘릴수록 시간·메모리가 선형에 가깝게 늘어난다: 6개 20.3h/12.9G vs 10개 24.2h/15.4G, 100 epoch 2×8 GPU 기준).
- **$s = 0.3$ vs $0.4$**: 3절 표에서 0.24–0.48 구간이 평평했으므로 코드가 0.4를 고른 것은 스윕 결과와 모순되지 않는다.

**추가 증거 — 아키텍처에 따라 $s$를 바꾼다.** 같은 저장소의 README는 ResNet-50으로 DINO를 학습할 때 명시적으로 SwAV의 값을 되돌려 쓴다.

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch resnet50 \
  --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 \
  --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 ...
```

즉 **ViT는 $s = 0.4$, ResNet-50은 $s = 0.14$**. 4절 해석 (2)("ViT가 작은 crop에 더 취약하다")를 뒷받침하는 실무적 정황이다. 또한 인자 도움말은 **multi-crop을 끌 때**(`--local_crops_number 0`) global 범위를 더 넓게 — `--global_crops_scale 0.14 1.` — 쓰라고 권한다. crop이 2개뿐이면 그 2개가 다양성을 전부 감당해야 하므로 하한을 낮춰 스케일 다양성을 확보하는 것이다.

---

## 6. 자주 틀리는 지점 체크리스트

- ❌ "$s$는 crop 한 변의 길이 비율" → ✅ **면적 비율**. 한 변 기준으로 환산하면 $\sqrt{s}$ 쪽에 가깝다($s=0.32$ ⇒ 정사각 기준 한 변 약 0.57배).
- ❌ "global은 $224^2$이니까 항상 원본의 큰 영역" → ✅ 해상도와 면적 비율은 별개. $s$가 작으면 작은 영역을 **업샘플**해 $224^2$로 채운다.
- ❌ "$s$가 커지면 global이 더 작은 영역도 본다" → ✅ 반대. $(s,1)$에서 $s$는 하한이므로 $s$↑ ⇒ global은 **더 넓은 영역만** 본다.
- ❌ "global과 local 범위가 겹친다" → ✅ 기본 설계는 $(0.05, s)$와 $(s, 1)$로 **겹치지 않음**. 단 논문은 이것이 SwAV를 따른 임의의 선택이며 겹쳐도 된다고 명시.
- ❌ "0.3이 정답" → ✅ 0.24–0.48이 거의 동등(69.5–69.8). 논문 표현도 "around 0.3"이고, 코드는 0.4를 쓴다.
- ❌ "local view도 teacher에 들어간다" → ✅ **teacher는 global view 2개만.** local은 student 전용이며, 이것이 $s$의 하한이 왜 중요한지의 이유이기도 하다.

---

## 참고 위치

- 논문 §3.1 (multi-crop 정의, Eq. 3): global 2개 $224^2$가 "50% 이상", local $96^2$가 "50% 미만" 영역이라는 대략적 서술.
- 논문 §3.2 Implementation details: BYOL의 증강(color jittering, Gaussian blur, solarization) + multi-crop, position embedding은 bicubic 보간으로 스케일에 맞춤.
- 논문 Table 8 (§5.4): local crop 개수별 정확도/시간/메모리.
- 논문 **부록 E "Multi-crop"**: $s$ 스윕 표, "optimum around 0.3", SwAV 0.14 언급, 비겹침 설계에 대한 코멘트, 프레임워크별 multi-crop 효과 비교.
- 코드: `main_dino.py`의 `get_args_parser()` multi-crop 인자와 `DataAugmentationDINO.__init__` / `__call__`, README의 ResNet-50 학습 명령.
