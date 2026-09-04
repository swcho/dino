# linear classification 평가 프로토콜의 세부 설정

## 한눈에 보기

| 항목 | 설정 |
|---|---|
| 백본 | frozen (gradient 없음), `model.eval()` |
| projection head | **제거** |
| 분류기 | `nn.Linear(embed_dim, 1000)` 하나 |
| 옵티마이저 | SGD (momentum 0.9), batch size 1024 |
| epoch | ImageNet 100 epoch |
| weight decay | **0 (미적용)** |
| 스윕 대상 | 학습률(lr) **하나만** |
| 학습 증강 | `RandomResizedCrop(224)` + `RandomHorizontalFlip` 만 |
| 평가 | central crop, top-1 accuracy |

논문 부록 F.2의 원문:

> "Following common practice in self-supervised learning, we evaluate the representation quality with a linear classifier. **The projection head is removed, and we train a supervised linear classifier on top of frozen features.** This linear classifier is trained with **SGD and a batch size of 1024 during 100 epochs on ImageNet. We do not apply weight decay. For each model, we sweep the learning rate value.** During training, we apply only **random resizes crops** (with default parameters from PyTorch `RandomResizedCrop`) and **horizontal flips** as data augmentation. **We report central-crop top-1 accuracy.**"

---

## 1. projection head를 제거하는 이유

![DINO 학습 구조: backbone + projection head](fig-2.jpeg)

DINO에서 네트워크는 $g = h \circ f$ 로 분해된다. $f$ 는 backbone(ViT 또는 ResNet), $h$ 는 projection head다. 부록 C에 따르면 head는 "3-layer MLP(hidden 2048d, GELU) → $\ell_2$ normalization bottleneck($d=256$) → weight-normalized FC($K = 65536$)" 구조다.

이 head는 **사전학습 목적함수 전용 구조**다. 출력 $K$ 차원은 "prototype"에 대한 확률분포이고, temperature softmax·centering·sharpening과 짝을 이루어 collapse를 피하도록 설계된 것이다. 즉 head의 출력은 "이미지의 범용 표현"이 아니라 "self-distillation 손실을 계산하기 좋은 좌표계"다. 반면 다운스트림에서 표현이라고 부르는 것은 관례적으로 backbone 출력 $f(x)$ 이므로, 평가도 그 정의를 따라야 한다.

head를 남기면 무엇이 달라지나:

- 평가 대상이 backbone 표현이 아니라 "backbone + 4개 선형층 + $\ell_2$ 병목 + 65536차원 사영"이 된다. head가 이미 비선형 변환을 여러 겹 수행하므로 **"선형 분리 가능성"이라는 측정 자체가 오염된다** (linear probe가 사실상 얕은 MLP probe가 된다).
- $\ell_2$ bottleneck이 표현을 256차원으로 압축하고 정규화해서 노름 정보를 버린다.
- 65536차원 출력에 선형층을 붙이면 분류기 파라미터가 폭증해 다른 방법과 공정 비교가 불가능해진다.
- 방법마다 head 구조/출력 차원이 달라 표 2 같은 **cross-method 비교의 공통 축이 사라진다.**

## 2. backbone 동결(frozen feature)

목적은 "사전학습이 만들어낸 표현이 이미 얼마나 좋은가"를 재는 것이다. backbone에 gradient가 흐르면 그건 fine-tuning 평가이고, 표현의 질과 백본의 적응 능력이 뒤섞인다. 논문 §3.2도 두 프로토콜을 명확히 구분한다.

> "Standard protocols for self-supervised learning are to either learn a linear classifier on frozen features or to finetune the features on downstream tasks. ... For finetuning evaluations, we initialize networks with the pretrained weights and adapt them during training."

구현상으로는 특징 추출을 `torch.no_grad()` 안에서 한다 — 공개 구현 `eval_linear.py`의 학습 루프:

```python
with torch.no_grad():
    if "vit" in args.arch:
        intermediate_output = model.get_intermediate_layers(inp, n)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        ...
output = linear_classifier(output)
```

포인트 셋:

- `model.eval()` + `no_grad()`: gradient가 backbone으로 흐르지 않고, dropout/stochastic depth 같은 학습 모드 동작도 끈다.
- optimizer에 넘기는 파라미터가 `linear_classifier.parameters()` 뿐이다 — backbone 파라미터는 애초에 옵티마이저에 등록되지 않는다.
- 표현의 정의(부록 F.2): **ViT-S는 마지막 $l=4$ 개 블록의 [CLS] token을 concat** ($384 \times 4 = 1536$ 차원), **ViT-B는 최종 layer만** 쓰되 patch token의 global average pooling을 [CLS]에 concat ($768 \times 2 = 1536$). convnet은 관례대로 최종 feature map에 global average pooling.

## 3. weight decay를 쓰지 않는 이유

`eval_linear.py`가 주석까지 달아 명시한다:

```python
optimizer = torch.optim.SGD(
    linear_classifier.parameters(),
    args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
    momentum=0.9,
    weight_decay=0,  # we do not apply weight decay
)
```

학습 가능한 것은 선형층 하나($\mathbf{W} \in \mathbb{R}^{1000 \times d}$, bias)뿐이고 입력 특징은 고정이다. 이런 볼록(convex)에 가까운 문제에서는 100 epoch 동안 특징이 변하지 않으므로 과적합 여력이 작고, ImageNet-1k 학습 세트는 1.28M 장으로 파라미터 수에 비해 충분히 크다. 여기에 $\lambda \lVert \mathbf{W} \rVert^2$ 를 넣으면 **비교하고 싶지 않은 자유도가 하나 더 생긴다**: 모델마다 최적 $\lambda$ 가 다르면 "표현이 좋아서 점수가 높은지, $\lambda$ 를 잘 골라서 높은지"를 구분할 수 없다. $\lambda = 0$ 으로 고정하면 프로토콜이 단순해지고, 스윕할 축이 lr 하나로 줄어 재현·비교가 쉬워진다.

## 4. 학습률만 스윕하는 이유 — 그리고 그 한계

남는 자유도가 lr뿐이라는 것은 곧 **스윕 비용이 1차원**이라는 뜻이다. 게다가 특징 스케일은 모델마다 다르다(ViT-S/16 vs ViT-B/8 vs ResNet-50, concat 개수도 다름). 고정된 lr 하나로는 어떤 모델은 발산하고 어떤 모델은 수렴이 덜 된 상태로 끝나므로, 최소한 lr은 모델별로 맞춰줘야 공정하다. 그래서 논문은 "For each model, we sweep the learning rate value"라고 쓴다.

공개 구현의 기본값도 스윕을 전제로 한다:

```
--lr default 0.001  # reference batch size 256 기준. 실제 lr = 0.001 * 1024/256 = 0.004
                    # "We recommend tweaking the LR depending on the checkpoint evaluated."
--epochs default 100
--batch_size_per_gpu default 128   # README 권장 실행이 8 GPU → 총 batch 1024 (논문과 일치)
scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
```

그런데 논문은 lr만 스윕해도 프로토콜 자체가 하이퍼파라미터에 민감하다는 점을 §3.2에서 스스로 지적한다.

> "However, **both evaluations are sensitive to hyperparameters, and we observe a large variance in accuracy between runs when varying the learning rate** for example. We thus also evaluate the quality of features with a simple weighted nearest neighbor classifier ($k$-NN)."

이 문장이 이 카드의 핵심 맥락이다. weight decay를 없애고 증강을 최소화해 자유도를 줄여도, **남은 lr 하나만으로도 run 간 정확도 분산이 크다.** 그래서 DINO는 linear probe를 유일한 지표로 쓰지 않고 $k$-NN을 병기하는 쪽으로 갔고, "linear 77.0 vs $k$-NN 74.5" 처럼 두 숫자를 항상 나란히 보고한다(표 2). 표 12의 관찰 — "the ranking of the frameworks depends on the evaluation protocol considered" — 도 같은 취지다: 프로토콜 하나에 순위를 맡기지 말라는 것.

## 5. 증강 최소화 + central crop 평가 (학습·평가의 비대칭)

학습 증강은 `RandomResizedCrop(224)` + `RandomHorizontalFlip`이 전부다. 평가는 `Resize(256, bicubic)` → `CenterCrop(224)`. 실제 구현:

```python
train_transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(...)])
val_transform   = Compose([Resize(256, interpolation=3), CenterCrop(224), ToTensor(), Normalize(...)])
```

증강을 이 둘로 제한하는 이유:

- **평가하려는 대상이 표현이지 증강 레시피가 아니다.** color jitter, blur, solarization, RandAugment, mixup 같은 강한 증강을 넣으면 정확도가 올라갈 수 있지만, 그 이득이 표현에서 온 건지 증강 튜닝에서 온 건지 분리할 수 없다. 특히 DINO 사전학습이 이미 BYOL식 color jitter/blur/solarization을 쓰기 때문에, 평가에서 같은 증강을 쓰면 사전학습 증강과의 궁합까지 점수에 섞인다.
- crop + flip은 **label-preserving하고 파라미터가 사실상 없는(PyTorch 기본값) 최소 정규화**다. 선형층 하나짜리 문제에 필요한 최소한의 데이터 다양성만 제공한다.
- 선행 SSL 논문들(SimCLR, MoCo, BYOL, SwAV)이 모두 이 crop+flip 조합을 linear probe 기본으로 써왔기 때문에, **표 2의 인용 수치와 직접 비교 가능**하다.

평가는 왜 central crop인가 — 학습/평가 증강의 비대칭은 의도된 것이다. 학습 시 랜덤 crop은 매 epoch 다른 view를 보여주는 **정규화 수단**이지만, 평가는 **결정적(deterministic)이고 재현 가능**해야 한다. 이미지마다 랜덤 crop을 뽑으면 같은 체크포인트를 두 번 재도 숫자가 달라진다. 그래서 관례적으로 "짧은 변을 256으로 resize → 중앙 224 crop → top-1" 이라는 단일 결정적 view를 쓴다. 10-crop/multi-crop TTA로 평균 내면 점수는 올라가지만, 그것 역시 튜닝 축이 되므로 쓰지 않는다. "central-crop top-1"이라는 명시는 곧 **TTA 없음**의 선언이다.

## 6. SGD / batch 1024 / 100 epoch은 어디서 왔나

세 값 모두 발명이 아니라 **선행 SSL 프로토콜과의 호환**을 위한 선택이다.

- **SGD + momentum 0.9**: linear probe는 사실상 다항 로지스틱 회귀이고, SGD + cosine decay는 이 문제의 표준 해법이다. Adam/AdamW를 쓰면 파라미터별 적응 스케일 때문에 특징 스케일에 대한 민감도가 달라지고, 인용해야 할 선행 수치들과 옵티마이저가 어긋난다. 사전학습에는 AdamW를 쓰지만(§3의 pretraining setup) 평가에는 SGD를 쓰는 비대칭이 여기서 나온다.
- **batch 1024 + linear scaling rule**: Goyal et al.의 $lr = \text{base\_lr} \times \dfrac{\text{batchsize}}{256}$ 규칙을 그대로 쓴다. DINO 사전학습도 batch 1024와 $lr = 0.0005 \times \text{batchsize}/256$ 을 쓰므로, 평가 배치를 1024로 맞추면 8-GPU 노드 하나(GPU당 128)에서 그대로 돌아간다. 논문 README 권장 실행이 `--nproc_per_node=8`에 `--batch_size_per_gpu 128`(기본값)이라 총 1024가 된다.
- **100 epoch**: SSL linear-probe의 사실상 표준 예산이다. 특징이 고정이라 수렴이 빠르고, 100 epoch면 cosine decay가 충분히 끝난다. 이보다 짧으면 모델별로 수렴 정도가 달라 비교가 불공정해지고, 길게 가도 얻는 것이 거의 없다(사전학습 예산 300~800 epoch에 비하면 무시할 수 있는 추가 비용).

## 7. $k$-NN 평가와의 비교

같은 frozen feature를 재는 두 프로토콜이지만 성격이 정반대다. **자유도 측면**: linear probe는 최소화해도 lr 스윕이 남고, 100 epoch × 1.28M 장의 SGD 학습 결과가 초기화·데이터 순서·증강 난수에 따라 흔들린다(§3.2의 "large variance in accuracy between runs"). $k$-NN은 부록 F.1대로 [CLS] token을 저장하고 $\alpha_i = \exp(T_i x / \tau)$, $\tau = 0.07$ (튜닝 안 함)로 가중 투표하며, $k$ 하나만 고르면 되고 그마저 $k = 20$ 이 거의 항상 최적이었다 — 즉 **"does not require any other hyperparameter tuning, nor data augmentation"**. **비용 측면**: linear probe는 100 epoch 분량의 forward를 반복해야 하는데, $k$-NN은 다운스트림 데이터셋을 **단 한 번 통과(one pass)**해 특징을 저장하면 끝이다. 대신 $k$-NN은 추론 시 학습 특징 전체를 들고 있어야 하고, 표현이 이미 코사인 거리 기준으로 클러스터링돼 있어야 잘 나온다 — 그래서 보통 linear보다 낮다(ViT-S/16: linear 77.0 vs $k$-NN 74.5). DINO+ViT의 인상적인 점은 이 격차가 유독 작다는 것이고(다른 SSL 방법이나 ResNet-50에서는 나타나지 않음), 논문은 두 숫자를 늘 함께 보고해 "프로토콜에 따라 방법 순위가 뒤바뀔 수 있다"는 위험을 방어한다.
