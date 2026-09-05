# 두 global crop의 비대칭 증강 — "저수준 통계 지름길" 차단

## 질문과 답

**Q.** 두 global crop의 증강을 서로 다르게 만든 의도는 무엇인가?

**A.** BYOL에서 온 설계로, 두 view의 **저수준 통계를 다르게** 만들어 색·주파수 같은 단서로 쉽게 매칭되는 **지름길(shortcut)을 막는다**. 표현 붕괴를 추가로 억제하는 효과가 있다.

---

## 1. 코드에서 비대칭이 어디에 있나

`main_dino.py:419-464` 의 `DataAugmentationDINO` 는 세 개의 파이프라인을 만든다. 앞부분 `flip_and_color_jitter` 와 `normalize` 는 **세 개가 완전히 공유**한다.

```python
flip_and_color_jitter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
])
```

차이는 **꼬리 두 줄**뿐이다.

| crop | 해상도 | `scale` | 꼬리 증강 |
|---|---|---|---|
| global 1 (`global_transfo1`) | 224 | $(0.4,\ 1.0)$ | `GaussianBlur(1.0)` |
| global 2 (`global_transfo2`) | 224 | $(0.4,\ 1.0)$ | `GaussianBlur(0.1)` + `Solarization(0.2)` |
| local × 8 (`local_transfo`) | 96 | $(0.05,\ 0.4)$ | `GaussianBlur(0.5)` |

인자로 들어간 숫자는 **적용 확률**이지 세기가 아니다 (`utils.py:36-68`):

```python
class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p                      # ← 첫 인자가 확률
    def __call__(self, img):
        if random.random() > self.prob: return img
        return img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(self.radius_min, self.radius_max)))
```

즉 global 1은 **항상** 블러가 걸리고 (radius $\sim \mathcal{U}(0.1,\ 2.0)$), global 2는 **10번에 1번만** 블러가 걸리며 대신 5번에 1번 solarize 된다. `Solarization` 은 `ImageOps.solarize` 의 기본 threshold 128을 쓰므로, 128보다 밝은 픽셀 $v$ 를 $255 - v$ 로 뒤집는다 — **밝은 영역의 색이 반전**되는, 히스토그램에 대한 매우 폭력적인 연산이다.

정리하면 두 global view의 관계는:

$$
x_1^{g} = t_1(x),\qquad x_2^{g} = t_2(x),\qquad t_1 \sim \mathcal{T}_1,\ \ t_2 \sim \mathcal{T}_2,\ \ \mathcal{T}_1 \neq \mathcal{T}_2
$$

두 view가 **같은 증강 분포에서 i.i.d. 로 뽑히지 않는다**. 이것이 "비대칭 증강(asymmetric augmentation)"이다.

---

## 2. "지름길(shortcut)"이란 정확히 무엇인가

자기지도 학습의 목적은 "같은 이미지의 두 view는 같은 표현을 갖게 하라"이다. 문제는 **그 목적을 만족시키는 방법이 여러 가지**라는 것이다.

$$
\text{목표: } f(x_1^g) \approx f(x_2^g)
$$

이 등식을 만족시키는 방법은 두 부류다.

1. **의미(semantic) 경로** — "이건 강아지 얼굴이다"를 인식해서 같은 값을 낸다. 우리가 원하는 것.
2. **저수준 통계 경로** — 두 crop이 **원본이 같으므로 공유할 수밖에 없는 값**을 읽어서 같은 값을 낸다. 예: 평균 RGB, 색 히스토그램, 채도, 고주파 에너지량.

2번이 훨씬 **싸다**. 컨볼루션 한 층이면 색 히스토그램의 근사치를 뽑을 수 있는데, 의미를 알아내려면 네트워크 전체가 필요하다. 경사하강법은 손실을 가장 빨리 낮추는 방향으로 가므로, 2번이 열려 있으면 **거기로 먼저 간다**. 이렇게 학습 신호를 우회해서 만족시키는 값싼 경로가 지름길이다.

### SimCLR의 관찰

Chen et al. 2020 (SimCLR) 논문 §3의 그림(Fig. 6 부근)은 이걸 그림으로 보여준다. 한 이미지에서 crop 여러 개를 떼어내 각각의 픽셀 강도 히스토그램을 겹쳐 그리면 **거의 포개진다**. 다른 이미지에서 뜬 crop들은 히스토그램이 확연히 다르다. 그래서 논문은 이렇게 적는다:

> "Neural nets may exploit this shortcut to solve the predictive task. Therefore, it is critical to compose cropping with color distortion."

crop만 쓴 SimCLR의 linear probe 정확도는 형편없고, **crop + color distortion** 조합에서 성능이 뛴다. 색 정보 자체가 쓸모없어서가 아니라 — **너무 쉬운 정답지라서** 일부러 망가뜨리는 것이다.

### DINO/BYOL은 여기서 한 걸음 더

SimCLR의 color jitter는 **두 view에 같은 분포로** 걸린다 ($p=0.8$ 씩). 이건 "색 통계 자체를 노이즈로 만든다"까지는 하지만, 두 view가 여전히 **같은 종류의 왜곡**을 겪으므로 잔여 통계가 상관을 유지한다. 특히:

- **주파수 통계**: color jitter는 픽셀값을 바꿔도 **엣지의 선명도는 건드리지 않는다**. 두 crop의 고주파 에너지량은 여전히 거의 같다.
- **밝기 정렬**: jitter의 brightness는 곱셈 스케일이라 "어느 쪽이 더 밝은가"의 순서 관계가 자주 보존된다.

DINO의 비대칭 증강은 남은 두 축을 정확히 겨냥한다.

| 지름길 축 | 차단 장치 | 어떻게 |
|---|---|---|
| **주파수 / 선명도** | blur $p{=}1.0$ vs $p{=}0.1$ | view 1은 항상 흐리고 view 2는 거의 항상 선명 → 고주파 에너지량이 체계적으로 어긋난다 |
| **밝기 / 색 히스토그램** | solarize $p{=}0.0$ vs $p{=}0.2$ | view 2에서 5번에 1번 밝은 픽셀이 반전 → 히스토그램이 완전히 다른 모양이 된다 |

핵심은 **"세게" 망가뜨리는 게 아니라 "다르게" 망가뜨리는 것**이다. 두 view에 똑같이 blur $p{=}0.5$ 를 걸면 평균적 왜곡 강도는 비슷하지만, 두 view의 블러 정도는 여전히 상관되어 있다 — 지름길이 살아 있다. $1.0$ 과 $0.1$ 로 **갈라놓아야** "이 view가 흐린가"라는 특징이 상대 view를 예측하는 데 무용해진다.

수식으로 말하면, 저수준 통계 $s(\cdot)$ 에 대해 두 view의 상호정보량을 낮추는 것이다:

$$
I\big(s(x_1^{g});\ s(x_2^{g})\big)\ \downarrow
\quad\Longrightarrow\quad
\text{손실을 낮추려면 } I\big(\text{semantic}(x_1^{g});\ \text{semantic}(x_2^{g})\big) \text{ 를 써야 한다}
$$

증강 설계란 곧 **"어떤 정보를 view 간에 공유되지 않게 만들 것인가"의 선택**이고, 남겨진 공유 정보가 표현이 학습하는 불변량(invariant)이 된다.

---

## 3. BYOL 부록의 표 — DINO가 그대로 가져온 수치

Grill et al. 2020 (BYOL) 부록 A "Image augmentations"의 표가 원본이다. 두 view $t$, $t'$ 에 대해:

| 파라미터 | $t$ | $t'$ |
|---|---|---|
| random crop 확률 | 1.0 | 1.0 |
| flip 확률 | 0.5 | 0.5 |
| color jitter 확률 | 0.8 | 0.8 |
| brightness / contrast / saturation / hue 최대 세기 | 0.4 / 0.4 / 0.2 / 0.1 | 동일 |
| color dropping (grayscale) 확률 | 0.2 | 0.2 |
| **Gaussian blur 확률** | **1.0** | **0.1** |
| **solarization 확률** | **0.0** | **0.2** |

**두 줄만 다르다.** DINO의 `global_transfo1` / `global_transfo2` 는 이 표의 $t$ / $t'$ 를 글자 그대로 옮긴 것이다. jitter 세기 `(0.4, 0.4, 0.2, 0.1)`, grayscale `p=0.2`, flip `p=0.5` 까지 전부 일치한다.

BYOL이 이 비대칭을 넣은 맥락도 참고할 만하다. BYOL은 negative pair가 없어서 "모든 입력에 같은 값" 이라는 자명해가 열려 있는 구조인데, 논문의 ablation은 **비대칭 증강을 제거하면 성능이 떨어진다**고 보고한다. DINO도 negative가 없는 같은 계열(self-distillation)이므로 이 설계가 그대로 유효하다.

> 참고: BYOL 표기 관례에서 online network가 $t$, target network가 $t'$ 를 받는다. DINO에서는 두 global view가 **양쪽 다** 교사·학생을 거치므로 (`teacher(images[:2])`, 그리고 학생은 전부) 방향성보다 **분포가 두 종류라는 사실 자체**가 본질이다.

---

## 4. "표현 붕괴를 추가로 억제" — centering/sharpening과 다른 층위

DINO의 붕괴 방지 장치를 층별로 나열하면 이렇다. 여기서 헷갈리기 쉬운 게, **증강 비대칭과 centering/sharpening이 같은 문제를 다른 방식으로 푸는 게 아니라, 서로 다른 실패 모드를 막는다**는 점이다.

| 층위 | 장치 | 무엇을 막나 | 작동 지점 |
|---|---|---|---|
| 0 | `weight_norm` 의 $g_k{=}1$ 고정 | 프로토타입 노름 폭주 | 모델 구조 |
| **입력** | **비대칭 증강** | **지름길 해(shortcut solution)** | **데이터 파이프라인** |
| 출력 | centering ($z_t - c$) | 단일 프로토타입 collapse ($H(P_t)\to 0$) | 손실 함수 |
| 출력 | sharpening ($\tau_t < \tau_s$) | uniform collapse ($P_t \to 1/K$) | 손실 함수 |

**centering / sharpening은 출력 분포의 모양을 직접 규제한다.** $P_t$ 가 한 차원을 독식하거나 평평해지는 것을 EMA center를 빼고 온도를 낮춰 **사후적으로** 교정한다. 이 둘은 서로 반대 방향으로 밀며, 균형이 유지되는 한 붕괴가 안 일어난다 (논문 Fig. 5).

**비대칭 증강은 출력 분포를 전혀 건드리지 않는다.** 대신 **문제 자체를 어렵게 만들어** 값싼 해를 없앤다. 그래서 성격이 다르다.

- centering/sharpening 없이는 → 붕괴한다. **필수**.
- 비대칭 증강 없이도 → 학습은 돌아간다. 다만 **표현의 질이 떨어진다**. 손실은 잘 내려가는데 linear probe / k-NN 정확도가 안 오르는 형태로 나타난다.

두 번째가 위험한 이유는 **손실 곡선만 보면 정상으로 보이기 때문**이다. 지름길 해는 목적함수 관점에서 "성공"이다 — 두 view의 표현이 실제로 일치한다. 다만 그 일치가 의미가 아니라 색 통계에서 온다. 이건 centering이 감지할 수 있는 것도 아니고, 엔트로피 $H(P_t)$ 도 건강해 보일 수 있다. 진단하려면 downstream 지표를 봐야 한다.

또 하나의 차이: centering/sharpening은 **하이퍼파라미터 균형**($\tau_t$, `center_momentum`)에 민감해서 스케줄까지 필요하지만 (warmup teacher temp $0.04 \to$ `teacher_temp`), 비대칭 증강은 고정된 확률 두 개일 뿐 스케줄이 없다. 튜닝 부담이 없는 무료 방어선에 가깝다.

---

## 5. 세 종류의 증강 분포 — multi-crop까지 포함하면

local crop 8개까지 세면 DINO의 증강 분포는 **두 종류가 아니라 세 종류**다.

$$
\mathcal{T}_1\ (\text{global 1}),\qquad \mathcal{T}_2\ (\text{global 2}),\qquad \mathcal{T}_{\text{loc}}\ (\text{local} \times 8)
$$

local은 blur $p{=}0.5$ 로 **두 global의 중간**에 놓여 있고, solarize는 없다. 그리고 결정적으로 **해상도(96 vs 224)와 `scale` 범위가 다르다**:

$$
\text{local: } \mathrm{scale} \in (0.05,\ 0.4), \qquad \text{global: } \mathrm{scale} \in (0.4,\ 1.0)
$$

`RandomResizedCrop` 의 `scale` 은 원본 **면적 대비 비율**이므로, local의 상한 $0.4$ 와 global의 하한 $0.4$ 가 맞닿아 **local crop은 항상 global 이하의 영역을 본다**. 이건 §2의 "저수준 통계 지름길 차단"과는 또 다른 축의 장치다 — local-to-global 대응, 즉 "부분을 보고 전체를 예측"을 강제한다.

세 분포가 손실 항 안에서 어떻게 만나는지 보면 그림이 완성된다. 목적함수의 항은 $|\mathcal{N}| = 2(2+N) - 2 = 18$ 개 ($N=8$):

| 교사 view | 학생 view | 항 수 | 어긋나는 축 |
|---|---|---|---|
| $x_1^{g}$ | $x_2^{g}$ | 1 | blur/solarize (**비대칭 증강**) |
| $x_2^{g}$ | $x_1^{g}$ | 1 | blur/solarize (**비대칭 증강**) |
| $x_1^{g}$ | local × 8 | 8 | 해상도 + 영역 (**multi-crop**) |
| $x_2^{g}$ | local × 8 | 8 | 해상도 + 영역 (**multi-crop**) |

18개 중 **global↔global 2개만이 비대칭 증강의 수혜자**이고, 나머지 16개는 multi-crop이 만든 local-to-global 항이다. 두 장치가 서로 다른 항을 담당한다.

그래서 이렇게 읽으면 된다: local crop들은 해상도와 crop 영역이 이미 크게 달라서 저수준 통계 지름길이 자연히 약하다 (96px로 리사이즈되면서 주파수 특성도 바뀐다). **문제는 해상도도 scale 범위도 똑같은 두 global crop**이다 — 여기만 지름길이 활짝 열려 있고, 그래서 blur/solarize 비대칭이 **정확히 여기에만** 들어간다.

---

## 6. 정리

- **지름길**: 같은 원본에서 뜬 두 crop은 색 히스토그램·주파수 에너지 같은 저수준 통계를 공유한다. 네트워크는 의미 대신 이 통계로 두 view를 매칭할 수 있고, 그게 더 싸므로 실제로 그쪽으로 간다 (SimCLR의 관찰).
- **차단 원리**: 두 view에 **서로 다른** blur/solarize 확률을 주면 이 통계의 상관이 깨진다. 세기가 아니라 **비대칭성**이 본질이다.
- **출처**: BYOL 부록 A 표의 $t$/$t'$ — blur $1.0/0.1$, solarize $0.0/0.2$. DINO가 수치까지 그대로 가져왔다.
- **붕괴 억제와의 관계**: centering/sharpening이 출력 분포를 사후 교정하는 **필수** 장치라면, 비대칭 증강은 입력 단에서 값싼 해를 없애는 **다른 층위**의 장치다. 없어도 학습은 돌지만 표현의 질이 조용히 떨어진다.
- **세 종류 분포**: global 1 / global 2 / local $\times$ 8. 비대칭 증강은 global↔global 2개 항을, multi-crop은 나머지 16개 local→global 항을 담당한다.

---

## 참고

- `main_dino.py:419-464` — `DataAugmentationDINO`
- `utils.py:36-68` — `GaussianBlur`, `Solarization` (첫 인자가 **확률**)
- `.fm/assets/dino_training_walkthrough.py` §3 (150-166행) — 증강 표와 의도, §7 (446-536행) — centering/sharpening 균형 실험
- [BYOL: Bootstrap your own latent (Grill et al., 2020)](https://arxiv.org/pdf/2006.07733) / [부록 A "Image augmentations" 표](https://proceedings.neurips.cc/paper_files/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Supplemental.pdf)
- [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709) — color histogram shortcut 관찰, "crop 단독 대신 crop + color distortion"
