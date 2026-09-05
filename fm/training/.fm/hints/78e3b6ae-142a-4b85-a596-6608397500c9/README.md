# `eval_knn.py`의 k-NN이 val 100장 미만에서 죽는 이유

> **Q.** `eval_knn.py`의 `knn_classifier`가 val 이미지 100장 미만에서 죽는 이유는?
>
> **A.** `imgs_per_chunk = num_test_images // 100`이 0이 되어 `range()`의 step이 0이 되기 때문이다.
> `ValueError: range() arg 3 must not be zero`로 실패한다.

---

## 1. 문제의 3줄

`eval_knn.py`의 `knn_classifier` 도입부(파일 기준 146–149행):

```python
num_test_images, num_chunks = test_labels.shape[0], 100   # num_chunks 하드코딩
imgs_per_chunk = num_test_images // num_chunks            # 정수 나눗셈 ← 여기가 0이 될 수 있다
retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
for idx in range(0, num_test_images, imgs_per_chunk):     # step=0 → ValueError
```

- `num_chunks = 100`은 **인자도 옵션도 아닌 리터럴**이다. CLI에서 바꿀 방법이 없다.
- `//`는 버림 나눗셈이므로 $N < 100$이면 $\lfloor N/100 \rfloor = 0$.
- `range(start, stop, 0)`은 파이썬 언어 차원에서 금지 — step 0이면 무한 루프이므로 인터프리터가 즉시 거부한다.

따라서 실패 조건은 정확히

$$
N_{\text{test}} < 100 \quad\Longleftrightarrow\quad \left\lfloor \frac{N_{\text{test}}}{100} \right\rfloor = 0
\quad\Longrightarrow\quad \texttt{ValueError}
$$

이고, $N_{\text{test}} = 0$부터 $99$까지 **전부** 죽는다. 모델·가중치·특징 추출과는 무관하게, 특징을 다 뽑고 GPU에 다 올린 **맨 마지막 단계**에서 터진다는 점이 특히 얄밉다.

### 1줄 재현 (모델 없이)

```python
>>> list(range(0, 50, 50 // 100))
ValueError: range() arg 3 must not be zero
```

실제로 위 한 줄이 스택트레이스의 마지막 프레임과 동일한 예외다. 노트북(`.fm/assets/dino_training_walkthrough.py`) §12 말미와 §14 "실전 함정" 2번이 지목하는 바로 그 지점이다.

---

## 2. 왜 굳이 100개로 쪼개는가 — 청킹의 의도

버그를 없애려고 청킹 자체를 지우면 안 된다. 청킹은 **메모리 폭발 방지**가 목적이다.

k-NN 분류의 핵심 연산은 val 전체와 train 전체 사이의 유사도 행렬이다:

$$
S = Z_{\text{test}} Z_{\text{train}}^{\top} \in \mathbb{R}^{N_{\text{test}} \times N_{\text{train}}}
$$

특징이 L2 정규화되어 있으므로 내적 = 코사인 유사도다($S_{ij} = \cos(z_i, z_j)$). 문제는 이 행렬을 **한 번에** 만들면 크기가 특징 차원 $d$와 무관하게 $N_{\text{test}} \times N_{\text{train}}$이라는 것.

ImageNet 기준으로 계산해 보면:

| 대상 | 형태 | fp32 크기 |
|---|---|---|
| 전체 한 번에 | $50{,}000 \times 1{,}281{,}167$ | $\approx 2.39 \times 10^{11}$ B = **238.6 GiB** |
| 100 청크 중 1개 | $500 \times 1{,}281{,}167$ | $\approx 2.39$ GiB |

$$
50000 \times 1281167 \times 4\ \text{B} = 238.6\ \text{GiB}
\qquad\longrightarrow\qquad
\frac{238.6}{100} = 2.39\ \text{GiB}
$$

238 GiB는 어떤 단일 GPU에도 안 올라간다. 반면 2.39 GiB 청크는 A100 한 장에 충분히 들어간다. 즉 `num_chunks = 100`은 **"ImageNet val 50k를 500장씩 자르면 대충 GPU에 맞더라"** 는, 저자의 워크로드에 맞춰진 경험적 상수다.

부수적으로 `topk`, `scatter_`, `gather`, `torch.mul` 브로드캐스트가 만드는 중간 텐서 — 특히 `retrieval_one_hot.view(B, k, C) * distances.view(B, k, 1)` 이 $B \times k \times C$ (예: $500 \times 200 \times 1000$ = 1억 원소) — 도 같이 청크 크기에 비례해 줄어든다.

**요약: 청킹은 val 쪽 축 $N_{\text{test}}$를 잘라 피크 메모리를 100분의 1로 낮추는 장치다. 상수 100 자체에는 아무 의미가 없다.**

---

## 3. "100의 배수가 아니면 마지막 몇 장이 조용히 누락된다"? — **아니다**

이 코드를 처음 보면 자연스럽게 드는 두 번째 의심이 있다: $N$이 100의 배수가 아니면 나머지 $N \bmod 100$장이 루프에서 빠지는 것 아닌가? 결론부터: **누락은 없다.** 슬라이스가 `min(...)`으로 잘려 있기 때문이다.

```python
features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
targets  = test_labels [idx : min((idx + imgs_per_chunk), num_test_images)]
```

증명은 짧다. $s = \lfloor N/100 \rfloor \ge 1$일 때 `range(0, N, s)`의 마지막 값은

$$
\text{idx}_{\text{last}} = s \cdot \left\lfloor \frac{N-1}{s} \right\rfloor
$$

이고, 정의상 $\text{idx}_{\text{last}} + s \ge N$이다. 따라서 마지막 반복의 상한은 `min(idx_last + s, N) = N`이 되어 **꼬리가 통째로 마지막 청크에 흡수된다**. 각 청크는 서로 겹치지 않고 이어 붙으므로 커버리지는 항상 정확히 $N$.

실제로 돌려서 확인한 값:

| $N$ | step $s=\lfloor N/100\rfloor$ | 반복 수 | 커버된 이미지 | 누락 | 마지막 청크 크기 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0 | — | — | — | **ValueError** |
| 99 | 0 | — | — | — | **ValueError** |
| 100 | 1 | 100 | 100 | 0 | 1 |
| 150 | 1 | 150 | 150 | 0 | 1 |
| 199 | 1 | 199 | 199 | 0 | 1 |
| 250 | 2 | 125 | 250 | 0 | 2 |
| 299 | 2 | 150 | 299 | 0 | 1 |
| 1005 | 10 | 101 | 1005 | 0 | 5 |
| 49999 | 499 | 101 | 49999 | 0 | 99 |
| 50000 | 500 | 100 | 50000 | 0 | 500 |

`total += targets.size(0)`로 누적한 분모 `total`도 항상 $N$과 일치하므로, 정확도 $\text{top1} \times 100 / \text{total}$ 역시 편향되지 않는다.

**대신 진짜로 어긋나는 것은 "청크 개수"다.** 이름이 `num_chunks = 100`인데, $N$이 100의 배수가 아니면 반복 횟수는 100이 아니라 **최대 199회**까지 늘어난다($N=199$일 때 $s=1$ → 199회). 즉 상수 100은 "청크 개수"가 아니라 "청크 크기를 정하는 나눗셈의 분모"일 뿐이고, 실제 청크 개수는 $\lceil N/\lfloor N/100 \rfloor \rceil$이다. $N$이 100~199 구간이면 이미지 1장씩 GPU 커널을 100번 넘게 부르게 되어 **극도로 비효율적**이지만, 결과는 맞다.

정리하면 이 함수의 결함은 하나뿐이다: **$N < 100$에서의 하드 크래시.** "조용한 누락"은 존재하지 않는다.

---

## 4. 노트북 §12는 어떻게 우회했는가

`.fm/assets/dino_training_walkthrough.py` §12는 **`knn_classifier`를 재사용하지 않고 청킹 없는 버전을 직접 재정의**한다. 주석이 의도를 명시한다 — `# eval_knn.knn_classifier 와 동일 로직 (val<100장에서 죽는 chunk 버그만 제거)`:

```python
@torch.no_grad()
def knn_top1(ftr, ltr, fte, lte, ncls, k=20, T=0.07):
    sim = fte @ ftr.t()                                  # 청킹 없이 전체 한 번에
    d, idx = sim.topk(min(k, ftr.shape[0]), dim=-1)      # k > train 크기 방어
    nb = ltr[idx]
    w = (d / T).exp()
    probs = torch.zeros(fte.shape[0], ncls).scatter_add_(1, nb, w)
    return (probs.argmax(-1) == lte).float().mean().item() * 100
```

여기서 눈여겨볼 두 가지:

1. **청킹 제거가 정당한 이유** — 이 노트북의 데이터셋은 CIFAR-10 부분집합(train 600 / val 400) 규모다. 유사도 행렬이 $400 \times 600 \times 4\,\text{B} \approx 0.9$ MB에 불과하므로 쪼갤 이유가 전혀 없다. 청킹은 대규모에서만 필요한 장치이고, 소규모에서는 **버그만 남고 이득은 0**이다.
2. **`min(k, ftr.shape[0])`** — 같은 성격의 두 번째 소규모 함정을 미리 막은 것이다. `eval_knn.py`의 `--nb_knn` 기본값은 `[10, 20, 100, 200]`이라 train이 200장 미만이면 `similarity.topk(200, ...)`이 `RuntimeError: selected index k out of range`로 죽는다. §12는 $k=20$을 쓰면서도 train 크기로 한 번 더 clamp한다.

(대안은 val 부분집합을 100장 이상으로 맞추는 것이지만, 그건 "데이터를 코드에 맞추는" 회피책이고 train 크기 문제는 여전히 남는다. 노트북은 함수 재정의 쪽을 택했다.)

---

## 5. 안전한 패치

원본 함수를 고친다면 한 줄이면 된다.

```python
import math
# 기존: imgs_per_chunk = num_test_images // num_chunks
imgs_per_chunk = max(1, math.ceil(num_test_images / num_chunks))
```

- $N \ge 100$: $\lceil N/100 \rceil$은 청크 개수를 **항상 100 이하**로 유지한다($\lfloor \cdot \rfloor$은 최대 199개까지 늘어났다). 이름 `num_chunks`의 의미가 비로소 맞아떨어진다.
- $N < 100$: `max(1, ...)`이 step을 1로 고정 → 한 장씩 $N$번, 느리지만 **죽지 않는다**.
- 슬라이스의 `min(...)`은 그대로 두면 되고, 커버리지는 §3의 논증에 의해 여전히 완전하다.

더 파이썬다운 대안은 인덱스 산술을 아예 없애는 것이다:

```python
for f_chunk, t_chunk in zip(test_features.split(imgs_per_chunk),
                            test_labels.split(imgs_per_chunk)):
    ...
```

`torch.split`은 마지막 조각이 짧아도 알아서 처리하고, `split(0)`은 `range(...,0)`처럼 조용한 무한 루프가 아니라 즉시 명확한 에러를 낸다. 다만 step 0 자체는 여전히 위쪽 `max(1, ...)`로 막아야 한다.

`num_chunks`를 하드코딩 대신 `--num_chunks` CLI 인자로 빼는 것도 같이 하면 좋다. 어차피 이 값은 "내 GPU에 몇 GiB가 남았나"에 따라 달라지는 값이지, 알고리즘 상수가 아니다.

---

## 6. 같은 패턴: "대규모 기본값 가정이 소규모 실험에서 깨진다"

이 버그의 본질은 산술 실수가 아니라 **설계 가정의 스코프**다. DINO 저장소는 ImageNet(train 1.28M / val 50k, 8 GPU, 100 epoch)을 전제로 쓰였고, 그 전제를 벗어난 순간 곳곳이 깨진다. 노트북 §14의 "실전 함정" 목록이 사실상 이 패턴의 카탈로그다.

| 위치 | 대규모 가정 | 소규모에서 벌어지는 일 |
|---|---|---|
| `eval_knn.knn_classifier` | val이 100장보다 훨씬 많다 | $N<100$ → `ValueError: range() arg 3 must not be zero` |
| `eval_knn` `--nb_knn` 기본 `[10,20,100,200]` | train이 200장보다 많다 | `topk(200)` → `RuntimeError: selected index k out of range` |
| `knn_classifier(..., num_classes=1000)` | 클래스가 1000개 (main에서 인자로 안 넘김) | 2~10 클래스 데이터에서도 $k \times 1000$ one-hot을 잡음. 결과는 맞지만 낭비이고, top5의 의미가 흐려짐 |
| `utils.cosine_scheduler`의 `assert len(schedule) == epochs * niter_per_ep` | `epochs`(기본 100) $>$ `warmup_epochs`(기본 10) | 2~3 epoch 스모크 테스트에서 `warmup_epochs > epochs` → `np.arange(음수)`가 빈 배열 → assert 실패. `--warmup_epochs 0`을 반드시 같이 줘야 한다 |
| `--freeze_last_layer 1` (epoch 단위) | 한 epoch이 수천 iteration | 전체가 2 epoch이면 학습의 **절반**이 last layer 동결 상태. 스모크 테스트 결과가 정상 학습과 질적으로 달라진다 |
| `eval_linear.py` | 러너 스크립트가 디렉터리를 만들어 줌 | `output_dir`을 스스로 만들지 않아 실패 — 미리 `mkdir -p` |

교훈 두 가지:

1. **정수 나눗셈 `//`가 등장하면 "분자가 분모보다 작을 때"를 항상 물어보라.** 특히 그 결과가 `range`의 step, 텐서의 chunk 크기, 나눗셈의 분모로 흘러들어갈 때.
2. **논문 저장소의 "기본값"은 논문의 실험 설정이지 안전한 기본값이 아니다.** 소규모로 파이프라인을 검증할 때는 배치·epoch뿐 아니라 이런 **숨은 규모 상수**(청크 수, $k$, 클래스 수, warmup)까지 함께 스케일 다운했는지 확인해야 한다.

---

## 참고 위치

- `/home/sungwoo/projects/swcho/dino/eval_knn.py` — `knn_classifier` (143–182행), 문제의 청킹은 146–149행
- `/home/sungwoo/projects/swcho/dino/utils.py` — `cosine_scheduler` (187행~), 197행 assert
- `/home/sungwoo/projects/swcho/dino/fm/training/.fm/assets/dino_training_walkthrough.py` — §12 k-NN 우회 구현(`knn_top1`), §14 실전 함정 목록
