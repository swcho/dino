# `eval_knn.py`가 학습하는 파라미터 개수는?

## 답

**0개**다. backbone을 얼린 뒤 CLS 특징만 뽑아 코사인 유사도로 이웃을 찾으므로 학습 대상이 전혀 없다.

`eval_knn.py`에는 `optimizer`도, `loss.backward()`도, `requires_grad=True`로 켜는 텐서도 없다.
스크립트 전체가 **순전파 한 번 + 행렬곱 한 번**으로 끝난다.

---

## 1. `eval_knn.py`의 전체 흐름

DINO 사전학습(`main_dino.py`)에는 검증 루프가 없다. loss는 붕괴하지 않아도 계속 $\log K$ 근처를
맴돌기 때문에 "표현이 좋아졌는가"는 **따로** 재야 하고, 그 가장 싼 측정기가 k-NN이다.

### 단계 1 — 모델 로드, head는 버린다

```python
model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)   # eval_knn.py:60
model.cuda()
utils.load_pretrained_weights(model, args.pretrained_weights,
                              args.checkpoint_key, args.arch, args.patch_size)  # :71
model.eval()                                                                     # :72
```

- `num_classes=0` → `VisionTransformer.__init__`에서
  `self.head = nn.Linear(...) if num_classes > 0 else nn.Identity()` (vision_transformer.py:159)
  이므로 **분류 head 자체가 만들어지지 않는다**.
- `--checkpoint_key`의 기본값은 `"teacher"`다. 즉 평가 대상은 학생이 아니라 **EMA teacher**다
  (teacher가 항상 학생보다 표현이 좋다).
- `load_pretrained_weights`(utils.py:71~82)는 `module.` / `backbone.` 접두사를 벗긴 뒤
  `load_state_dict(..., strict=False)`로 넣는다. 체크포인트에 들어 있던
  `head.mlp.*`, `head.last_layer.*` 키는 모델에 대응 파라미터가 없으므로 **조용히 무시**된다
  (반환 `msg`의 `unexpected_keys`에 찍힌다). 이것이 "DINOHead 폐기"의 실체다.
- `--pretrained_weights` 경로가 없으면 아키텍처에 맞는 공식 가중치를 torch.hub에서 자동으로 받는다.

### 단계 2 — `model.eval()` + `@torch.no_grad()`

```python
@torch.no_grad()                                  # eval_knn.py:95
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    ...
    feats = model(samples).clone()                # :105
```

- `model.eval()`은 dropout / drop-path를 끄고 정규화를 추론 모드로 바꾼다.
- `@torch.no_grad()`는 autograd 그래프를 아예 만들지 않는다 → gradient가 존재할 수 없다.
- 두 줄이 합쳐져 "backbone 동결"이 된다. `requires_grad=False`를 명시적으로 걸지 않아도
  `no_grad` 안에서는 어차피 grad가 생기지 않는다.

### 단계 3 — train / val 특징 추출: $(N, D)$

`VisionTransformer.forward`의 마지막 줄은 `return x[:, 0]` — 즉 마지막 블록 + `norm` 이후의
**[CLS] 토큰 하나**다. patch 토큰도, 중간 블록도 쓰지 않는다.

$$
Z_{\text{train}} \in \mathbb{R}^{N_{\text{tr}} \times D},\qquad
Z_{\text{val}} \in \mathbb{R}^{N_{\text{te}} \times D},\qquad
D = \texttt{embed\_dim}\ (\text{ViT-S} = 384,\ \text{ViT-B} = 768)
$$

`ReturnIndexDataset`(:185)이 `(img, idx)`를 돌려주는 이유는 분산 환경에서 `all_gather` 후
`features.index_copy_(0, index_all, ...)`(:136)로 **원래 순서 자리에 되꽂기** 위해서다.
DataLoader의 `DistributedSampler`가 순서를 흩뜨려도 인덱스를 같이 들고 다니면 복원할 수 있다.

### 단계 4 — L2 정규화

```python
train_features = nn.functional.normalize(train_features, dim=1, p=2)   # eval_knn.py:81
test_features  = nn.functional.normalize(test_features,  dim=1, p=2)   # :82
```

$$
z \leftarrow \frac{z}{\lVert z \rVert_2}
$$

정규화해 두면 이후 내적이 곧 코사인 유사도다:

$$
\langle z_x, z_i\rangle = \frac{z_x^\top z_i}{\lVert z_x\rVert\,\lVert z_i\rVert} = \cos(z_x, z_i) \in [-1, 1]
$$

### 단계 5 — 유사도 행렬과 top-$k$

```python
train_features = train_features.t()                                  # :145  → (D, N_tr)
similarity = torch.mm(features, train_features)                      # :158  (B, N_tr)
distances, indices = similarity.topk(k, largest=True, sorted=True)   # :159  (B, k)
```

변수명이 `distances`지만 실제로는 **유사도**다(클수록 가깝다). 이름에 속지 말 것.

val을 100개 chunk로 잘라 도는 이유는 $(N_{\text{te}} \times N_{\text{tr}})$ 행렬을 통째로
올리지 않기 위해서다 — ImageNet이면 $50\text{k} \times 1.28\text{M}$ = 약 256 GB라 불가능하다.

### 단계 6 — 온도 가중 투표

```python
retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)     # :164
distances_transform = distances.clone().div_(T).exp_()                # :165  exp(sim / T)
probs = torch.sum(retrieval_one_hot.view(B, -1, C) *
                  distances_transform.view(B, -1, 1), 1)              # :166
_, predictions = probs.sort(1, True)                                  # :173
```

수식으로 쓰면:

$$
\hat{y}(x) = \arg\max_{c}\ \sum_{i \in \mathcal{N}_k(x)}
\mathbb{1}[y_i = c]\cdot \exp\!\left(\frac{\cos(z_x, z_i)}{T}\right),
\qquad T = 0.07
$$

- 이웃 레이블을 one-hot으로 펼친 뒤 $\exp(\text{sim}/T)$ 가중치를 곱해 클래스별로 더하고 argmax.
- $T = 0.07$은 아주 낮은 온도다. $\cos$ 차이 0.07이면 가중치가 $e$배 벌어지므로
  **가장 가까운 이웃 몇 개가 사실상 표를 독점**한다. 즉 단순 다수결이 아니라 "거리로 급격히
  기운 다수결"이다.
- `probs.sort` 후 상위 5개까지 보고 top-5도 함께 리포트한다(`k < 5`면 의미 없음, :178 주석).

---

## 2. "학습 파라미터 0개"를 코드로 확인하는 법

grep 관점에서 `eval_knn.py`에 **없는 것들**:

| 찾을 것 | `eval_knn.py` | `eval_linear.py` |
|---|---|---|
| `torch.optim` / `optimizer` | 없음 | `optim.SGD(...)` (:103) |
| `.backward()` | 없음 | `loss.backward()` (:180) |
| `.step()` | 없음 | `optimizer.step()` (:183) |
| `requires_grad = True` | 없음 | (LinearClassifier 기본값) |
| `nn.Module` 서브클래스 정의 | `ReturnIndexDataset`(Dataset) 뿐 | `LinearClassifier` (:237) |
| `.train()` 호출 | 없음 (`model.eval()`만) | `linear_classifier.train()` (:154) |
| `--epochs` 인자 | 없음 | 기본 100 (:265) |

```bash
grep -nE "optimizer|backward|\.step\(|requires_grad|\.train\(\)" eval_knn.py   # → 결과 없음
```

`extract_features`와 `knn_classifier` **둘 다** `@torch.no_grad()`가 붙어 있다(:95, :142).
스크립트 안에서 grad가 살아 있는 구간이 물리적으로 존재하지 않는다.

---

## 3. 왜 "비모수(non-parametric)" 평가인가

통계학에서 **비모수 모델**은 고정된 개수의 파라미터로 데이터를 요약하지 않고,
**데이터 자체를 모델로 삼는** 방법이다. k-NN이 교과서적 예시다.

- 학습 = 특징을 저장하는 것. 그게 전부다. "적합(fit)"이라 부를 절차가 없다.
- 결정 경계는 저장된 train 특징들의 배치가 암묵적으로 정의한다.
- 조정할 것은 **$k$와 $T$ 두 개의 하이퍼파라미터뿐**이고, 이마저도 학습되는 값이 아니다
  (`--nb_knn` 기본값 `[10, 20, 100, 200]` 전부를 한 번의 특징 추출로 동시에 평가한다 — :238).

### 장점

1. **재현성 / 무편향**: 학습률·에폭·초기화·데이터 증강 같은 튜닝 여지가 없다.
   숫자가 나쁘면 그건 표현 탓이지 평가 프로토콜 탓이 아니다.
   linear probe는 lr을 잘못 주면 몇 %가 왔다 갔다 한다(eval_linear.py의 lr 도움말도
   "체크포인트마다 lr을 조정하길 권장"이라고 실토한다).
2. **빠르다**: 특징 추출 1 epoch + 행렬곱. linear probe의 100 epoch SGD와 비교가 안 된다.
   사전학습 중간중간 붙여 모니터링용으로 쓸 수 있다.
3. **표현의 기하를 직접 측정한다**: "같은 클래스가 코사인 거리 기준으로 뭉쳐 있는가"를
   그대로 묻는다. 선형 변환으로 구제해 주지 않는다.

### 한계

1. **메모리**: train 특징 전체를 들고 있어야 한다.
   ImageNet ViT-S/16이면 $1{,}281{,}167 \times 384 \times 4\ \text{bytes} \approx 1.97\ \text{GB}$ (fp32).
   ViT-B/16은 $D = 768$이라 약 4 GB. GPU에 올리다 OOM 나면 `--use_cuda false`.
2. **선형 분리 가능성은 못 본다**: 클래스가 선형 초평면으로는 깔끔히 갈리지만
   국소적으로 얽혀 있는 표현은 k-NN에서 과소평가된다. 반대도 가능하다.
3. **코사인 거리 하나에 전부를 건다**: 특징 스케일 정보는 정규화로 버려진다.
4. **레이블 불균형/노이즈에 민감**: 이웃 후보가 많은 클래스가 유리하다.

---

## 4. linear probe(`eval_linear.py`)와의 대비

| | `eval_knn.py` | `eval_linear.py` |
|---|---|---|
| 학습 파라미터 | **0** | $D_{\text{eff}} \times C + C$ |
| ViT-S/16 기준 실제 수 | 0 | `n_last_blocks=4` → $D_{\text{eff}} = 384 \times 4 = 1536$, $C=1000$ → **약 1.54 M** |
| backbone | frozen (`eval()` + `no_grad`) | frozen (`eval()`, grad는 `no_grad`로 특징만 추출) |
| 쓰는 특징 | 마지막 블록 CLS **1개** | 마지막 $n$개 블록의 CLS를 concat (+ 옵션으로 patch avgpool) |
| 옵티마이저 | 없음 | SGD(momentum 0.9, wd 0) + CosineAnnealingLR |
| 학습 시간 | 특징 추출 1 pass | **100 epoch** (`--epochs` 기본값) |
| 튜닝 대상 | $k$, $T$ | lr, epochs, `n_last_blocks`, `avgpool_patchtokens` |
| 증강 | CenterCrop만 | train에 RandomResizedCrop + flip |
| 코드 근거 | `@torch.no_grad()` × 2 | `optimizer.step()`, `loss.backward()` |

`LinearClassifier`는 이름 그대로 `nn.Linear(dim, num_labels)` 딱 하나다(eval_linear.py:237~251).
backbone은 여기서도 얼려 있지만 — 그 위에 **학습되는 층이 하나 얹힌다**는 점이 결정적 차이다.

> 실전 함정: `eval_linear.py`는 `output_dir`을 만들어 주지 않으니 미리 `mkdir -p` 해야 하고,
> 저장되는 체크포인트는 best가 아니라 **last**다.

---

## 5. DINO 논문에서 k-NN이 특별한 이유

자기지도 학습 평가에서 보통 linear probe가 k-NN보다 한참 높다. 그런데 DINO에서는:

| 방법 (ViT-S/16) | linear | k-NN | 격차 |
|---|---|---|---|
| DINO | 77.0% | **74.5%** | 2.5%p |
| (대조군) 여러 이전 SSL 방법 | — | — | 통상 5~10%p 이상 |

**격차가 작다는 것 자체가 결과다.** linear probe는 $D \times C$짜리 학습된 변환으로 특징을
"교정"할 기회를 갖는다. k-NN은 그 기회가 없다. 그런데도 2.5%p밖에 안 벌어진다는 건
**특징 공간이 이미 의미적으로 군집해 있다**는 뜻이다 — 같은 클래스 이미지들이 별도 학습 없이도
코사인 거리 기준으로 이미 서로 가깝다.

이 성질이 DINO의 다른 유명한 결과들과 직결된다:
- CLS 어텐션이 객체 경계를 따라간다(§13)
- 이미지 검색(copy detection, Oxford/Paris retrieval)에 특징을 그대로 꽂아 쓸 수 있다
- 특징에 k-means를 돌리면 클래스 구조가 나온다

논문이 k-NN 수치를 항상 linear 옆에 나란히 싣는 이유가 이것이다.
"우리 표현은 downstream 학습 없이도 쓸 만하다"는 주장의 증거로 쓰인다.

---

## 6. 노트북 §12의 실측

워크스루 §12는 `eval_knn.knn_classifier`와 동일한 로직을 `knn_top1(k=20, T=0.07)`으로
재현해서 CIFAR-10 부분집합(train 600 / val 400)에 돌린다:

| 백본 | 20-NN top1 |
|---|---|
| random init ViT-Tiny | ~24% (chance 10% 대비 약간 위) |
| 미니 학습(수십 step) teacher | ~24% (랜덤과 구별 안 됨) |
| **공식 DINO ViT-S/16 (frozen)** | **87.0%** |

> **이 숫자를 믿지 말 것** — 노트북이 직접 경고한다. 수십 step 학습한 ViT-Tiny와 랜덤 초기화는
> 둘 다 chance 근처다. DINO는 ImageNet ViT-S/16 8 GPU 기준 100 epoch에 약 1.75일이 걸린다.
> 의미 있는 비교 대상은 공식 사전학습 가중치뿐이다.

랜덤 초기화 백본을 비교 기준으로 함께 돌리는 습관이 중요하다.
"87%"가 인상적인 이유는 같은 파이프라인에서 랜덤이 24%를 내기 때문이다.

### 함정: val < 100장이면 죽는다

```python
num_test_images, num_chunks = test_labels.shape[0], 100
imgs_per_chunk = num_test_images // num_chunks          # eval_knn.py:147
for idx in range(0, num_test_images, imgs_per_chunk):   # :149
```

val 이미지가 100장 미만이면 `imgs_per_chunk == 0`이 되어
`ValueError: range() arg 3 must not be zero`로 죽는다.
노트북은 이 chunking을 제거한 버전을 쓴다. 소규모 데이터로 실험할 땐
`imgs_per_chunk = max(1, num_test_images // num_chunks)` 정도로 고쳐야 한다.

---

## 7. 실전 팁

### `--use_cuda`

```
--use_cuda True   (기본)  특징 행렬을 GPU에 상주 → 행렬곱이 빠름
--use_cuda False          CPU RAM에 저장 → 느리지만 OOM 회피
```

도움말이 직접 "OOM 만나면 False로 두길 권장"이라고 적어 놓았다(:199~200).
ImageNet ViT-B/8 같은 조합에서는 특징 행렬만 수 GB인 데다 chunk 유사도 행렬이 겹쳐서
쉽게 터진다. 정확도는 동일하고 속도만 바뀐다.

### `--dump_features` / `--load_features`

특징 추출은 전체 시간의 대부분이지만 **$k$나 $T$와 무관**하다. 그래서 한 번 뽑아 저장해 두고
재사용하는 흐름이 표준이다:

```bash
# 1) 특징 추출 + 저장 (오래 걸림)
python -m torch.distributed.launch --nproc_per_node=4 eval_knn.py \
    --data_path /path/to/imagenet --arch vit_small --patch_size 16 \
    --dump_features out/knn_feats

# 2) 이후 실험은 저장된 특징만 로드 (초 단위)
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
    --load_features out/knn_feats --temperature 0.04 --nb_knn 5 10 20 50
```

저장 파일은 `trainfeat.pth`, `testfeat.pth`, `trainlabels.pth`, `testlabels.pth` 4개다(:88~91).
`--load_features` 경로에서 이 이름 그대로 찾으므로 파일명을 바꾸면 안 된다.
저장되는 특징은 **이미 L2 정규화된 상태**다(정규화가 :81~82, 저장이 :88 순서).

추가로:
- `--nb_knn`은 리스트를 받으므로 한 번의 실행으로 $k \in \{10, 20, 100, 200\}$을 전부 본다.
  DINO 저자들은 "20이 보통 제일 잘 된다"고 도움말에 적어 두었다.
- `--checkpoint_key teacher`가 기본이지만, 학생을 재보고 싶으면 `student`로 바꿀 수 있다
  (거의 항상 teacher가 낫다).
- `eval_knn.py`는 `utils.init_distributed_mode`를 부르므로 GPU 1장이라도
  `torch.distributed.launch`(또는 `torchrun`)로 띄워야 한다.
- `extract_features`에 `multiscale=True` 옵션이 있지만 CLI로는 노출되지 않았다
  (`utils.multi_scale`을 쓰는 경로, :102~103). 필요하면 코드를 직접 고쳐야 한다.

---

## 한 줄 정리

`eval_knn.py`는 **분류기가 아니라 자(ruler)** 다. 파라미터 0개, 순전파 1회, 행렬곱 1회로
"이 표현 공간에서 같은 클래스끼리 실제로 뭉쳐 있는가"를 직접 잰다.
그래서 학습이 개입할 여지가 없고, 그래서 표현 품질의 정직한 지표가 된다.
