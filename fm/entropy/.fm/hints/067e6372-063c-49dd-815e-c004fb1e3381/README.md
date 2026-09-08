# loss가 붕괴를 못 잡는다면 — 실전 모니터링 플레이북

붕괴한 DINO 모델의 학습 손실은 건강한 모델과 같거나 **더 낮다**.
그래서 `log.txt`의 `train_loss` 곡선만 보고 있으면, 표현이 이미 무너진 뒤 며칠치 GPU를 태우고 나서야 알게 된다.

실전에서 해야 할 일은 두 가지다.

1. **바깥 지표**: 손실과 무관한 다운스트림 신호 — `eval_knn.py`를 주기적으로 돌린다.
2. **안쪽 지표**: teacher 출력 분포의 통계 — 이미 메모리에 있는 텐서에서 몇 줄로 뽑아 매 iteration 로깅한다.

1번은 정확하지만 무겁고 늦다. 2번은 거의 공짜고 즉시 반응한다. **둘 다** 필요하다.

---

## 1. `eval_knn.py`가 실제로 하는 일

`/home/sungwoo/projects/swcho/dino/eval_knn.py`, 242줄. 파이프라인은 세 단계다.

### (a) feature 추출 — `extract_feature_pipeline` (L29–91)

```python
model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
model.cuda()
utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, ...)
model.eval()
```

- **어느 가중치인가**: `--checkpoint_key`의 기본값이 `"teacher"`다. 즉 기본 동작은 **teacher backbone** 평가다. DINO 논문에서도 teacher가 student보다 일관되게 좋다.
- **어느 레이어, 어느 토큰인가**: `num_classes=0`이므로 head는 `nn.Identity`이고, `VisionTransformer.forward`(`vision_transformer.py` L208–213)는

  ```python
  x = self.prepare_tokens(x); [blk(x) for blk in self.blocks]; x = self.norm(x)
  return x[:, 0]
  ```

  **마지막 블록의 최종 LayerNorm을 통과한 CLS 토큰 하나**다. ViT-S/16이면 $d=384$, ViT-B/16이면 $d=768$ (논문 F.1과 일치).
- **`--n_last_blocks`는 여기 없다.** 그 인자는 `eval_linear.py`(L256, 기본 4)에만 있고, 거기서는 `model.get_intermediate_layers(inp, n)`로 마지막 $n$개 블록의 CLS를 concat해 $384 \times 4 = 1536$차원을 만든다. k-NN 평가는 **마지막 블록 CLS 하나만** 쓴다 — 이걸 헷갈리면 재현이 안 된다.
- **DINO head는 평가에 전혀 쓰이지 않는다.** `out_dim=65536` 프로토타입 공간은 학습용 보조 장치일 뿐, 우리가 원하는 표현은 backbone feature다. 붕괴 진단이 head 통계(2절)와 backbone 품질(k-NN)로 **나뉘는** 이유가 이것이다.
- 변환은 `Resize(256) → CenterCrop(224) → ToTensor → Normalize`, 즉 augmentation 없는 중앙 크롭 1패스다.

### (b) L2 정규화 + cosine similarity

```python
train_features = nn.functional.normalize(train_features, dim=1, p=2)
test_features  = nn.functional.normalize(test_features,  dim=1, p=2)
```

두 쪽 모두 단위 벡터로 만들어 두기 때문에, `knn_classifier` 안의

```python
similarity = torch.mm(features, train_features)   # train_features는 이미 .t() 된 상태
```

내적이 그대로 **cosine similarity**가 된다. 별도의 거리 함수가 없다 — 정규화가 곧 거리 정의다.

### (c) 가중 투표 — `knn_classifier` (L141–181)

```python
distances, indices = similarity.topk(k, largest=True, sorted=True)
retrieved_neighbors = torch.gather(candidates, 1, indices)
retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
distances_transform = distances.clone().div_(T).exp_()
probs = torch.sum(retrieval_one_hot.view(B, -1, C) * distances_transform.view(B, -1, 1), 1)
```

이웃 $i$의 표에 $\alpha_i = \exp(s_i / \tau)$ 가중치를 준다. $s_i$는 cosine similarity, $\tau$는 `--temperature`.

클래스 $c$의 총점은

$$\text{score}(c) = \sum_{i \in \mathcal{N}_k} \exp\!\left(\frac{s_i}{\tau}\right)\mathbf{1}_{c_i = c}$$

인자 두 개의 역할:

| 인자 | 기본값 | 역할 |
|---|---|---|
| `--nb_knn` | `[10, 20, 100, 200]` | 이웃 수 $k$. **리스트**로 받아 `for k in args.nb_knn:` 루프를 돈다. feature는 한 번만 뽑고 투표만 다시 하므로 $k$를 4개 보는 비용은 사실상 공짜다. 논문 F.1과 argparse help 모두 **$k=20$이 가장 안정적**이라고 말한다. |
| `--temperature` | `0.07` | 투표 가중치의 온도. $\tau \to 0$이면 1-NN에 수렴하고, $\tau$가 크면 $k$개 이웃의 단순 다수결에 가까워진다. 논문은 **튜닝하지 않았다**고 명시(`[73]`에서 그대로 가져옴). |

top-5는 `correct.narrow(1, 0, min(5, k))`로, $k < 5$일 때는 의미가 없으니 잘라 쓴다.

### 왜 linear probe보다 가벼운 진단인가

| | `eval_knn.py` | `eval_linear.py` |
|---|---|---|
| 학습 | **없음** (`@torch.no_grad()` 두 함수뿐) | SGD 100 epoch (`--epochs` 기본 100) |
| 튜닝할 하이퍼파라미터 | $k$, $\tau$ — 논문이 안 건드림 | `--lr` 스윕 필수, `--n_last_blocks`, `--avgpool_patchtokens` |
| 데이터 통과 | train+val **1패스** | train 100패스 + val 100패스 |
| 결과 해석 | feature 공간의 **국소 이웃 구조**를 직접 측정 | 선형 분리가능성 — 학습이 잘 됐는지와 섞임 |

논문 F.1의 표현 그대로: *"This evaluation protocol does not require hyperparameter tuning, nor data augmentation and can be run with only one pass over the downstream dataset."*

붕괴 진단 관점에서 결정적인 건 마지막 줄이다. **linear probe는 나쁜 feature 위에서도 분류기가 어떻게든 맞춰버릴 수 있지만**, k-NN은 feature 자체가 이웃을 옳게 모으지 못하면 곧바로 무너진다. 붕괴 감지에는 k-NN이 더 예민한 센서다.

캐싱 인자도 있다 — `--dump_features DIR`로 저장해 두면 `--load_features DIR`로 forward 없이 투표만 다시 돌릴 수 있다.

---

## 2. 주기와 비용 — 매 에폭은 과하다

### 대략의 비용 감각 (ViT-S/16, 224px 글로벌 크롭 forward 1회 = 1 unit)

DINO 한 에폭이 이미지 한 장당 하는 일:

- teacher forward: 글로벌 2장 → $2$
- student forward+backward (backward $\approx 2\times$ forward): 글로벌 2장 + 로컬 8장.
  로컬은 96px → 토큰 $ (96/16)^2 + 1 = 37$ vs 글로벌 $197$, 즉 $\approx 0.19$ unit.
  $3 \times (2 + 8 \times 0.19) \approx 10.5$

합 $\approx 12.5$ unit/이미지.

k-NN 평가는 이미지 한 장당 **정확히 1 unit**(inference 1패스)이다. 따라서

- **풀 ImageNet(train 1.28M + val 50k)** k-NN ≈ 학습 1에폭의 $1.33/12.5 \approx$ **11%**
- **train 10% 서브샘플(128k) + val 50k** ≈ **1.4%**
- 이걸 **10에폭마다** 돌리면 전체 학습 오버헤드 **0.15% 미만**

(자릿수 감각을 위한 추정치다. 실제 시간은 GPU 수·데이터로더 처리량에 좌우된다 — 실측하면 8×V100에서 풀 ImageNet feature 추출이 수 분, 10% 서브샘플은 1분 안쪽 수준이다.)

### 실용적 타협

```
--saveckp_freq 5          # 롤백 지점을 촘촘히 (기본 20은 붕괴 대응에 너무 성김)
k-NN: 10에폭마다, train 10% 서브샘플, val 전체, --nb_knn 20
```

서브샘플 만드는 법: `eval_knn.py`는 `datasets.ImageFolder`로 `data_path/train`, `data_path/val`을 읽는다. 클래스별로 균등하게 골라 **심볼릭 링크 트리**를 한 번 만들어 두고 그 경로를 `--data_path`로 주면 코드 수정이 전혀 없다.

```bash
python -m torch.distributed.launch --nproc_per_node=8 eval_knn.py \
  --pretrained_weights $OUT/checkpoint0050.pth --checkpoint_key teacher \
  --arch vit_small --patch_size 16 \
  --nb_knn 20 --data_path /path/to/imagenet_subset
```

### ⚠️ 서브샘플링 함정

`knn_classifier`는 검증셋을 100조각으로 쪼갠다.

```python
num_test_images, num_chunks = test_labels.shape[0], 100
imgs_per_chunk = num_test_images // num_chunks
for idx in range(0, num_test_images, imgs_per_chunk):
```

**val 이미지가 100장 미만이면 `imgs_per_chunk == 0`**이 되어 `range()` step 0으로 `ValueError`가 난다. val은 넉넉히(최소 수천 장) 남기고, 줄일 거면 **train 쪽**을 줄여라 — 비용의 대부분도 train feature 추출이다.

k-NN이 참조 집합을 줄이면 절대 top-1은 당연히 떨어진다. 우리가 보는 건 **절대값이 아니라 같은 서브셋 위에서의 시간에 따른 추세**다. 한 번 정한 서브셋은 학습 내내 고정해야 비교가 성립한다.

---

## 3. 안쪽 지표 — teacher 출력 통계를 매 iteration 로깅

k-NN은 10에폭마다다. 그 사이에 붕괴가 시작되면 최대 10에폭을 낭비한다. teacher 분포 통계는 그 틈을 메운다.

`main_dino.py`의 `train_one_epoch`(L300–359)은 이미 다음을 손에 쥐고 있다.

- `teacher_output` — shape `(2 * batch_size_per_gpu, out_dim)`, **raw logit**. 글로벌 크롭 2장만 teacher를 통과한다.
- `dino_loss.center` — shape `(1, out_dim)` 버퍼
- `dino_loss.teacher_temp_schedule[epoch]` — 현재 온도

손실이 실제로 보는 분포를 그대로 재구성하는 건 한 줄이다.

```python
q = F.softmax((teacher_output - dino_loss.center) / temp, dim=-1)
```

### 3-1. 매 iteration용 스칼라 (거의 공짜)

`train_one_epoch`의 loss 계산 직후, `metric_logger.update(loss=...)` 근처에 끼워 넣는다.

```python
# --- collapse probes (main_dino.py, train_one_epoch 안, loss 계산 직후) ---
if it % 50 == 0:                       # 50 iteration마다면 충분
    with torch.no_grad():
        temp = dino_loss.teacher_temp_schedule[epoch]
        # fp16 autocast 밖에서, float32로 계산해야 log가 안정적이다
        q = F.softmax((teacher_output.float() - dino_loss.center.float()) / temp, dim=-1)

        h_each = -(q * torch.log(q + 1e-12)).sum(-1).mean()      # 샘플별 엔트로피의 평균
        qbar   = q.mean(0)                                        # 배치 평균 분포
        h_mean = -(qbar * torch.log(qbar + 1e-12)).sum()          # 그 분포의 엔트로피

        codes  = q.argmax(-1)
        counts = torch.bincount(codes, minlength=q.shape[-1])
        top_share  = counts.max().float() / codes.numel()         # 최빈 코드 점유율
        n_codes    = (counts > 0).sum()                           # 이 배치에서 쓰인 코드 수

        c = dino_loss.center
        metric_logger.update(
            h_each=h_each.item(), h_mean=h_mean.item(),
            top_share=top_share.item(), n_codes=n_codes.item(),
            center_norm=c.norm().item(), center_std=c.std().item(),
        )
```

**비용**: `q`는 `(128, 65536)` float32 ≈ 32MB, reduce 몇 번. ViT-S forward 하나에 비하면 무시할 수준이고, `it % 50` 게이팅이면 더더욱 그렇다. `metric_logger.update`는 `**kwargs`로 스칼라를 받아 `SmoothedValue`에 누적하고(`utils.py` L318–323), 에폭 끝에 `synchronize_between_processes()` → `global_avg`로 `log.txt`에 그대로 실린다. **로깅 인프라를 새로 만들 필요가 없다.**

> **순서 주의**: `DINOLoss.forward`는 마지막 줄에서 `self.update_center(teacher_output)`을 호출한다. 위 스니펫을 `loss = dino_loss(...)` **뒤**에 두면 이미 갱신된 center를 쓰는 셈이다. `center_momentum=0.9`라 한 스텝 차이는 무시할 만하지만, 정확히 맞추려면 `dino_loss` 호출 **전**에 계산하라.

### 3-2. 에폭 누적 히스토그램 (임계값 판정은 반드시 이쪽으로)

**`h_mean`과 `n_codes`를 배치 단위로 재면 안 된다.** 샘플 $N$개의 평균 분포는 각 샘플이 완전 one-hot이어도 엔트로피가 최대 $\log N$까지밖에 안 오른다. `batch_size_per_gpu=64` → `teacher_output`은 128행 → $h_{\text{mean}} \le \log 128 = 4.85$. $\log K = 11.09$와 비교하는 건 **원리적으로 불가능**하다.

$K = 65536$짜리 히스토그램을 에폭 내내 누적하면(float32로 256KB — 무료다) 이 문제가 사라진다.

```python
# train_one_epoch 시작 부분
prob_acc = torch.zeros(args.out_dim, device='cuda')   # 확률 질량 누적
n_acc = 0

# --- 위 probe 블록 안에 두 줄 추가 ---
        prob_acc += q.sum(0)
        n_acc += q.shape[0]

# --- 루프가 끝난 뒤, metric_logger.synchronize_between_processes() 전후 ---
dist.all_reduce(prob_acc)
n_total = n_acc * dist.get_world_size()
p = prob_acc / n_total                                 # 에폭 전체의 평균 teacher 분포
h_mean_epoch = -(p * torch.log(p + 1e-12)).sum().item()
used_codes   = (p > 1.0 / (100 * args.out_dim)).sum().item()   # 균등의 1% 이상 쓰인 코드 수
top_share_epoch = p.max().item()
if utils.is_main_process():
    print(f"[epoch {epoch}] h_mean_epoch={h_mean_epoch:.3f} "
          f"(log K={math.log(args.out_dim):.3f})  used={used_codes}  top={top_share_epoch:.4f}")
```

`prob_acc`는 `q.sum(0)` 하나만 더하므로 iteration당 비용이 사실상 0이다. `it % 50` 게이팅 대신 매 iteration 누적하면 통계가 더 정확해지고, 그래도 여전히 싸다.

### 세 지표를 짝으로 읽는 법

| | $h_{\text{each}}$ (샘플별 확신) | $h_{\text{mean}}$ (코드 사용 분산) | `top_share` |
|---|---|---|---|
| **건강** | 낮음 | 높음 | 아주 작음 |
| **균등 붕괴** (centering만) | **높음** ($\to \log K$) | 높음 ($\to \log K$) | 작음 |
| **원-핫 붕괴** (sharpening만) | **0에 가까움** | **낮음** | **큼** |

$h_{\text{mean}}$ 하나만 높은 건 **좋은 신호다**. 균등 붕괴와 건강한 상태를 가르는 건 **$h_{\text{each}}$**다. 그래서 비율

$$r = \frac{h_{\text{each}}}{h_{\text{mean}}}$$

를 하나 더 찍어 두면 판정이 쉬워진다. 건강하면 $r \ll 1$, 균등 붕괴면 $r \to 1$ (둘 다 $\log K$).

### `center` 추이

`dino_loss.center`는 teacher logit의 배치 평균에 대한 EMA($m = 0.9$)다. `center_norm`과 `center_std`를 보면:

- **`center_std`가 커지면서 계속 발산** → 특정 차원의 logit이 구조적으로 커지고 있다. centering이 그걸 상쇄하려 애쓰는 중이다. 원-핫 쪽으로 밀리는 전조.
- **`center_std → 0`** → 모든 차원의 평균 logit이 같아졌다. teacher가 입력을 구분 못 하는 중.
- **step 함수처럼 튐** → `update_center`의 `dist.all_reduce`나 배치 크기 문제. 배치가 아주 작으면 EMA가 노이즈에 흔들려 centering 효과가 약해진다.

`center`는 `register_buffer`이므로 `dino_loss.state_dict()`에 들어가고, `main_dino.py` L285의 `save_dict['dino_loss']`로 **모든 체크포인트에 이미 저장돼 있다**. 로깅을 안 걸어 뒀더라도 과거 체크포인트를 열어 사후 분석이 가능하다.

```python
ck = torch.load("checkpoint0040.pth", map_location="cpu")
c = ck["dino_loss"]["center"]          # (1, 65536)
print(c.norm().item(), c.std().item(), c.max().item(), c.min().item())
```

---

## 4. 경보 기준 — `out_dim = 65536` 기준

$$\log K = \log 65536 = 11.0904$$

**아래 숫자는 출발점이지 절대 기준이 아니다.** 데이터셋, `out_dim`, 배치 크기, 아키텍처마다 건강한 범위가 다르다. 실제로 해야 할 일은 **한 번 잘 된 런의 곡선을 baseline으로 저장해 두고 새 런을 거기에 겹쳐 보는 것**이다. 아래는 baseline이 아직 없을 때 쓰는 임시 가드레일이다.

### 균등 붕괴 (uniform collapse)

| 조건 | 판정 |
|---|---|
| $h_{\text{mean,epoch}} \ge 0.9 \log K = 9.98$ **그리고** $h_{\text{each}} \ge 0.9 \log K$ | 🚨 균등 붕괴. teacher가 모든 입력에 사실상 균등분포를 내놓는다. |
| $h_{\text{mean,epoch}} \ge 0.9 \log K$, $h_{\text{each}}$는 낮음 | ✅ 정상. 코드를 골고루 쓰면서 샘플마다 확신하고 있다 — 이게 목표 상태다. |
| $r = h_{\text{each}}/h_{\text{mean}} > 0.9$ | ⚠️ 두 엔트로피가 붙었다. 입력에 따른 차이가 사라지는 중. |

$h_{\text{each}}$가 학습 초반 이후에도 계속 $\log K$의 90% 위에 눌러앉아 있으면, sharpening이 약한 것이다 → `--teacher_temp`를 낮춰라.

### 원-핫 / 집중 붕괴

| 조건 | 판정 |
|---|---|
| `top_share_epoch` > **0.1** | 🚨 코드 하나가 전체 확률 질량의 10%를 먹었다. 균등이면 $1/65536 \approx 1.5\times10^{-5}$다. |
| `top_share_epoch` > **0.01** 이면서 상승 추세 | ⚠️ 관찰 시작. |
| $h_{\text{mean,epoch}}$가 **급락** (예: 몇 에폭 만에 절반 이하) | 🚨 사용 코드가 급격히 줄고 있다. 값 자체보다 **기울기**가 신호다. |
| `used_codes` < **수백** (수천~수만이 정상 범위) | 🚨 $K = 65536$을 놓고 코드 200개만 쓰는 중이면 사실상 200-way 클러스터링이다. |
| $h_{\text{each}} < 0.1$ | ⚠️ teacher가 완전 one-hot이 됐다. centering이 안 먹고 있다는 뜻. |

`used_codes`를 볼 때는 **에폭 누적 히스토그램**을 써라(3-2절). 배치 단위 `n_codes`는 배치 크기가 상한이라 무의미하다.

### 최종 판정은 항상 k-NN이 한다

위 지표는 **조기 경보**다. 확정 진단은 `eval_knn.py`의 20-NN top-1이 내린다.

| k-NN 추세 | 대응 |
|---|---|
| 상승 중 | 계속 |
| 2회 연속 정체 | 관찰. 안쪽 지표 확인 |
| **하락** | 🚨 loss가 어떻게 보이든 붕괴다. 즉시 5절로 |

---

## 5. 붕괴가 시작된 뒤 — 되돌릴 수 있는가

**부분적으로만.** 완전히 무너진 뒤에는 손실 지형의 자명한 최소점에 갇혀 있어 거의 못 나온다. **초기 징후 단계**라면 체크포인트 롤백 + 하이퍼파라미터 조정으로 대개 회복된다. 그래서 조기 감지가 전부다.

### (a) 롤백

```bash
# 마지막으로 k-NN이 정상이었던 지점으로
cp $OUT/checkpoint0040.pth $OUT/checkpoint.pth
```

`train_dino`는 시작 시 **`output_dir/checkpoint.pth`만** 본다(L256, `restart_from_checkpoint`). 그러니 롤백은 파일 복사다.

복원되는 것: `student`, `teacher`, `optimizer`, `fp16_scaler`, `dino_loss`(**`center` 포함**), `epoch`.
복원되지 **않는** 것: **`args`**. 체크포인트에 저장돼 있지만 `restart_from_checkpoint`가 읽지 않는다 — **재개할 때 커맨드라인 하이퍼파라미터를 바꿀 수 있다는 뜻**이고, 이게 복구의 핵심 지렛대다. `lr_schedule`/`teacher_temp_schedule`은 재개 시점의 `epoch` 인덱스부터 새 인자로 다시 만들어진다.

`--saveckp_freq`가 기본 20이면 롤백 지점이 너무 성기다. **처음부터 5 정도로 두라.**

### (b) 손잡이들 — 어느 쪽으로 미는지

| 인자 | 기본값 | 올리면 | 붕괴 방향 |
|---|---|---|---|
| `--teacher_temp` | 0.04 | teacher가 **부드러워짐** (sharpening 약화) | 올리면 균등 쪽, 내리면 원-핫 쪽 |
| `--warmup_teacher_temp` | 0.04 | 초기 teacher가 부드러워짐 | 위와 같은 방향 |
| `--warmup_teacher_temp_epochs` | **0** | 낮은 온도(=날카로움) 구간이 길어짐 | 초기 불안정 완화 |
| `--freeze_last_layer` | 1 | head 마지막 층이 더 오래 고정 | 초기 붕괴 차단 |
| `--warmup_epochs` (lr) | 10 | lr이 천천히 오름 | 초기 불안정 완화 |
| `--clip_grad` | 3.0 | — | 0이면 비활성. 불안정하면 0.3~1.0으로 |
| `--use_fp16` | True | — | help: *"disable ... if the loss is unstable"* |
| `--norm_last_layer` | True | — | help: *"Not normalizing leads to better performance but can make training unstable"*. ViT-S는 false 권장, ViT-B는 true |

> ⚠️ **`--warmup_teacher_temp_epochs`의 기본값은 코드상 `0`인데 help 문자열은 `Default: 30`이라고 말한다** (`main_dino.py` L73–74). 상류 코드의 불일치다. **명시적으로 넘기지 않으면 teacher temp warmup이 아예 없다.** 다만 `warmup_teacher_temp`와 `teacher_temp`가 둘 다 0.04인 기본 설정에서는 스케줄이 어차피 상수라 티가 안 난다. `--teacher_temp 0.07`처럼 올릴 때만 문제가 된다 — README의 권장 조합이 `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`인 이유다.

스케줄의 실제 방향을 코드로 확인하면(`DINOLoss.__init__`, L373–377):

```python
self.teacher_temp_schedule = np.concatenate((
    np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
))
```

`0.04 → 0.07`, 즉 **날카로운 데서 시작해 부드러워진다**. 코드 주석이 이유를 밝힌다: *"a too high temperature makes the training instable at the beginning."* 초기에 teacher가 너무 물렁하면 신호가 없어 불안정하다는 것. `--teacher_temp`를 0.07 위로 올리면 help가 경고하듯 *"anything above 0.07 is unstable"*.

### (c) `--freeze_last_layer`가 존재하는 이유

이 인자가 하는 일은 `utils.py` L143–148에 6줄로 다 있다.

```python
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

`train_one_epoch`에서 `clip_gradients` **뒤**, `optimizer.step()` **앞**에 호출된다(L332, L340). `p.grad = None`이면 AdamW가 그 파라미터를 아예 건너뛴다 — **gradient도 weight decay도 적용 안 된다. 진짜로 안 움직인다.**

`last_layer`는 `DINOHead`(`vision_transformer.py` L275)의

```python
self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
```

즉 256차원 bottleneck → **65536개 프로토타입** 사영이다. 그 앞줄에서 입력이 `normalize(x, dim=-1, p=2)`로 단위 구 위에 올라가므로, 이 층은 사실상 **"단위 벡터와 65536개 프로토타입 방향의 cosine 유사도"**를 계산한다.

**왜 이걸 얼려야 하나.** 학습 시작 시점의 backbone feature는 무작위에 가깝다. 이때 프로토타입까지 자유롭게 움직이면 가장 쉬운 해가 있다 — **프로토타입 몇 개를 모든 입력이 향하는 방향으로 몰아버리는 것**. 그러면 손실은 즉시 떨어지고, 표현은 배운 게 없다. 이게 정확히 원-핫 붕괴의 씨앗이다.

마지막 층을 고정하면 65536개 프로토타입은 초기화된 무작위 방향에 그대로 머문다. 손실을 줄이는 유일한 길이 **backbone이 feature를 움직이는 것**밖에 남지 않는다. 프로토타입이 도망갈 수 없으니 backbone이 먼저 의미 있는 구조를 만들어야 하고, 그 뒤에 프로토타입이 풀려나 정렬된다.

`norm_last_layer=True`일 때 `weight_g.requires_grad = False`로 크기까지 묶는 것도 같은 계열의 안전장치다(방향만 학습, 크기는 1로 고정 — 특정 프로토타입이 norm을 키워 logit을 독식하는 걸 막는다).

argparse help가 실전 조언을 직접 준다: *"Try increasing this value if the loss does not decrease."* 기본 1에서 **2~3**으로 올리는 게 초기 불안정에 대한 첫 대응이다.

### (d) 복구 레시피

```bash
# 40에폭 지점으로 롤백 + 초기 안정화 강화
cp $OUT/checkpoint0040.pth $OUT/checkpoint.pth
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
  --arch vit_small --data_path $IMAGENET/train --output_dir $OUT \
  --teacher_temp 0.04 \                    # 0.07에서 되돌림 (sharpening 강화)
  --warmup_teacher_temp_epochs 30 \        # 명시적으로! 기본 0이다
  --freeze_last_layer 3 \                  # 기본 1 → 3
  --warmup_epochs 15 \                     # lr warmup 연장
  --use_fp16 false \                       # 불안정 의심 시
  --saveckp_freq 5
```

**한 번에 하나씩** 바꿔라. 다섯 개를 동시에 바꾸면 뭐가 들었는지 영영 모른다. 우선순위: `--freeze_last_layer` → `--warmup_teacher_temp_epochs` → `--teacher_temp` → `--use_fp16` → lr.

---

## 6. 혼동하기 쉬운 정상 현상

### 정상: loss가 $\log K$ 근처에서 시작해 천천히 내려온다

초기화 직후 student 출력은 사실상 균등분포이고, teacher도 그렇다. 크로스 엔트로피

$$-\sum_k q_k \log p_k$$

에서 $q$와 $p$가 모두 $\approx 1/K$면 값은 $\approx \log K = 11.09$다.
**첫 iteration의 loss가 11 언저리인 건 완벽하게 정상이다.** 여기서 몇 에폭에 걸쳐 부드럽게 내려온다.

### 정상: 첫 에폭에 loss가 거의 안 움직인다

`--freeze_last_layer 1`이 head 마지막 층을 얼려 뒀다. 의도된 동작이다.

### 정상: k-NN top-1이 초반 몇 에폭 동안 낮게 시작해 계단식으로 오른다

feature가 아직 구조를 못 잡았다. 단조 증가가 아니어도 되고, 몇 % 진동은 서브샘플 노이즈다.

### 정상: $h_{\text{each}}$가 초반에 $\log K$ 근처였다가 내려온다

teacher가 확신을 갖게 되는 정상 과정이다. **내려오기만 한다면.**

### 이상: 구분 기준

| 증상 | 정상 | 이상 |
|---|---|---|
| loss 하강 | $\log K$에서 시작해 **완만하게, 계속** | **급락 후 완전 평탄** — 자명해에 안착했을 가능성 |
| `top_share` | 학습 내내 작게 유지 | 단조 증가하며 $10^{-2}$ 돌파 |
| $h_{\text{each}}$ | 높은 데서 내려와 **중간 수준에 정착** | $0$까지 계속 내려가거나, warmup 끝나고도 $\log K$에 눌러앉음 |
| `used_codes` | 초반 상승 후 **안정** | 단조 감소, 특히 급감 |
| **k-NN top-1** | 상승 또는 정체 | **하락** |
| `center_std` | 완만히 변동 | 단조 발산 또는 $\to 0$ |

**핵심 판정 기준 하나만 고르라면: k-NN top-1이 내려가는데 loss도 같이 내려가고 있다면, 그건 붕괴다.**
loss가 좋아지는데 표현이 나빠지는 건 자기지도학습에서 유일하게 명확한 위험 신호다. 4절에서 본 대로 붕괴한 모델의 loss는 건강한 모델보다 **낮을 수** 있다 — 손실 곡선만 보면 오히려 "잘 되고 있다"고 오독하게 된다.

---

## 요약: 세 줄 체크리스트

1. **`log.txt`에 세 열을 추가하라** — `h_each`, `h_mean`, `top_share` (+ `center_norm`). 3-1절 스니펫, 10줄, 비용 무시 가능. 임계값 판정은 3-2절의 에폭 누적본으로.
2. **k-NN을 10에폭마다, train 10% 서브샘플로 돌려라** — 오버헤드 0.2% 미만, `--checkpoint_key teacher --nb_knn 20`.
3. **`--saveckp_freq 5`로 롤백 지점을 확보하라** — 붕괴 대응의 성공 여부는 얼마나 최근으로 되돌아갈 수 있느냐가 결정한다.

### 파일 참조

| 내용 | 위치 |
|---|---|
| k-NN 파이프라인 | `eval_knn.py` L29–91 (추출), L141–181 (투표), L189–212 (인자) |
| `--n_last_blocks` (k-NN 아님) | `eval_linear.py` L256 |
| CLS 토큰 반환 | `vision_transformer.py` L208–213 |
| `DINOHead.last_layer` | `vision_transformer.py` L275–278 |
| 학습 루프 / probe 삽입 지점 | `main_dino.py` L300–359 |
| `DINOLoss` / `center` / temp 스케줄 | `main_dino.py` L362–415 |
| `freeze_last_layer` 구현 | `utils.py` L143–148 |
| 체크포인트 저장·복원 | `main_dino.py` L255–265, L278–290 |
