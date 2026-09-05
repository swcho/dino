# DINO 사전학습 루프에 검증이 없다 — 실전 함의

> **Q.** DINO 사전학습 루프에 검증이 없다는 사실의 실전 함의는?
>
> **A.** loss / lr / wd 만 로깅하므로 **조기 종료도 best 모델 선택도 불가능**하다.
> 표현 품질을 보려면 학습을 멈추고 `eval_knn.py` 를 따로 돌려야 한다.

---

## 1. 코드로 확인: `train_dino` 의 epoch 루프에 무엇이 있고 무엇이 없나

`main_dino.py:269-295` — 한 epoch 이 끝난 뒤 일어나는 일 **전부**다.

```python
for epoch in range(start_epoch, args.epochs):
    data_loader.sampler.set_epoch(epoch)

    # ============ training one epoch of DINO ... ============
    train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
        data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
        epoch, fp16_scaler, args)

    # ============ writing logs ... ============
    save_dict = {
        'student': ..., 'teacher': ..., 'optimizer': ...,
        'epoch': epoch + 1, 'args': args, 'dino_loss': dino_loss.state_dict(),
    }
    utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
    if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
    if utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
```

**세 줄 요약**: (1) 한 epoch 학습 → (2) `checkpoint.pth` **덮어쓰기**
(+ `saveckp_freq` 마다 `checkpoint{epoch:04}.pth` 스냅샷) → (3) `log.txt` 한 줄 append.
**검증 데이터 로더도, 평가 함수 호출도, 지표 비교도 없다.**

대조군으로 `eval_linear.py:112-148` 의 루프를 보면 있을 것이 다 있다 —
`validate_network(...)` 호출, `best_acc = max(best_acc, test_stats["acc1"])`,
`to_restore = {"epoch": 0, "best_acc": 0.}`. 즉 이 저장소가 검증 루프를 *못 쓰는* 게 아니라,
**사전학습 단계에서는 쓸 수 없어서 안 쓰는** 것이다.

### 실제로 기록되는 스칼라

`train_one_epoch` (`main_dino.py:354-360`) 이 `MetricLogger` 에 넣는 것은 정확히 3개다.

```python
metric_logger.update(loss=loss.item())
metric_logger.update(lr=optimizer.param_groups[0]["lr"])
metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
...
metric_logger.synchronize_between_processes()
return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
```

| 기록됨 | 기록 안 됨 |
|---|---|
| `train_loss` (epoch 전체 global average) | 어떤 종류의 **정확도**도 (k-NN, linear, top-1) |
| `train_lr` (param group 0) | 교사 엔트로피 $H(P_t)$ |
| `train_wd` (param group 0) | 교사 top-1 확률 $\max_k P_t(k)$ |
| `epoch` | argmax 프로토타입 다양성 |
| — | center 노름 $\lVert c \rVert_2$ |
| — | EMA momentum $m$, teacher temp $\tau_t$ (스케줄 값) |
| — | grad 노름 / 클리핑 발생 비율 |

> `wd` 를 `param_groups[0]` 에서만 읽는 것도 의도적이다 — `utils.get_params_groups` 가
> bias·Norm 파라미터를 1번 그룹(not-regularized)으로 빼기 때문에 1번의 wd 는 항상 0 이다.

### `log.txt` 포맷과 읽는 법

JSON Lines, epoch 당 한 줄:

```json
{"train_loss": 8.3174, "train_lr": 0.000241, "train_wd": 0.0407, "epoch": 0}
{"train_loss": 8.2996, "train_lr": 0.000480, "train_wd": 0.0421, "epoch": 1}
```

```bash
# loss 추이만 뽑기
python - <<'PY'
import json, pathlib
for l in pathlib.Path("out/dino_train/log.txt").read_text().splitlines():
    d = json.loads(l)
    print(f"{d['epoch']:4d}  loss={d['train_loss']:.4f}  lr={d['train_lr']:.3e}  wd={d['train_wd']:.4f}")
PY

# 또는 jq
jq -r '[.epoch, .train_loss, .train_lr, .train_wd] | @tsv' out/dino_train/log.txt
```

읽을 때 확인할 것은 **loss 의 절대값이 아니라 정상 범위 안에 있는가**다.
$K = 65536$ 이면 $\log K \approx 11.09$, 노트북의 소형 설정($K=4096$)이면 $\log K \approx 8.32$.
DINO 는 학습 초반 오랫동안 loss 가 $\log K$ **근처에 머무는 것이 정상**이고,
구조는 그 평탄면 위에서 서서히 생긴다. `train_lr` 이 warmup 구간에서 선형 상승 후
cosine 으로 내려가는지, `train_wd` 가 0.04 → 0.4 로 올라가는지 정도가
"스케줄이 의도대로 주입되고 있다"는 확인용이다. 그 이상은 알 수 없다.

---

## 2. 왜 자기지도 사전학습에서 검증이 구조적으로 어려운가

### (a) 레이블이 없다

데이터 로더 자체가 `for it, (images, _) in enumerate(...)` — 레이블 자리를 `_` 로 버린다.
`ImageFolder` 가 폴더명에서 만들어 준 레이블조차 사용하지 않는다. 애초에 사전학습 데이터에
클래스 구조가 있다고 가정하지 않으므로, "held-out 정확도"라는 물건이 존재하지 않는다.

### (b) held-out loss 를 재도 소용없다 — loss 가 붕괴와 학습을 구분하지 못한다

이게 진짜 이유다. 설령 val split 을 잘라 DINO loss 를 계산해도 **의미 있는 신호가 아니다.**

DINO 목적함수는 교차 엔트로피이고, 교차 엔트로피는 다음과 같이 분해된다:

$$
H(P_t, P_s) \;=\; -\sum_k P_t(k)\log P_s(k) \;=\; \underbrace{H(P_t)}_{\text{교사 분포의 엔트로피}} \;+\; \underbrace{D_{\mathrm{KL}}(P_t \,\|\, P_s)}_{\text{두 view 의 정렬}}
$$

우리가 원하는 것은 **둘째 항**을 줄이는 것 — 같은 이미지의 서로 다른 crop 이 같은 분포로 가는 것.
그런데 옵티마이저는 **첫째 항을 깎아도 똑같이 loss 가 내려간다.** $H(P_t) \to 0$,
즉 교사가 모든 입력에 대해 단일 프로토타입만 출력하는 상태가 되면 loss 는 아주 잘 내려간다.
**이게 붕괴(collapse)이고, loss 상으로는 "학습이 잘 되는 것"과 구별되지 않는다.**

워크스루 §11 의 3-설정 실험이 이걸 실측으로 보여준다:

| 설정 | centering | $\tau_t$ | loss | 실제 상태 |
|---|---|---|---|---|
| DINO | O | 0.04 | $\log K$ 근처에 머묾 | **건강** (두 붕괴 영역 사이에 매달림) |
| centering 제거 | X | 0.04 | **세 설정 중 가장 많이 내려감** | **단일 프로토타입 붕괴** |
| sharpening 제거 | O | 0.10 $(=\tau_s)$ | $\log K$ 에서 꼼짝 안 함 | uniform 붕괴 (gradient 소멸) |

> loss 만 보면 **가장 좋아 보이는 실행이 가장 망가진 실행**이다.
> "loss 가 잘 안 내려가네, 뭔가 잘못됐나" 하고 centering 을 끄는 순간
> 로그는 예뻐지고 표현은 죽는다.

붕괴를 보려면 loss 가 아니라 **교사 분포의 모양**을 봐야 한다:

| 진단량 | 정의 | 붕괴 신호 |
|---|---|---|
| 교사 엔트로피 | $H(P_t) = -\sum_k P_t(k)\log P_t(k)$ | $\to 0$ (단일 프로토타입) 또는 $\to \log K$ (uniform) |
| 교사 top-1 확률 | $\max_k P_t(k)$ | $\to 1$ |
| argmax 다양성 | 배치 내 서로 다른 argmax 프로토타입 수 | $\to 1$ |
| center 노름 | $\lVert c \rVert_2$ | 발산 |

그리고 **이 넷 중 어느 것도 기본 루프에서 로깅되지 않는다.**

### (c) 그래서 논문·저장소는 별도 프로토콜로만 평가한다

DINO 논문(Caron et al., ICCV 2021)의 모든 수치는 사전학습이 **끝난 뒤**
frozen backbone 위에서 별도 프로토콜로 측정된다:

- **k-NN** (`eval_knn.py`) — 학습 파라미터 0개. CLS 특징 L2 정규화 후 코사인 유사도로 이웃 투표.
- **linear probe** (`eval_linear.py`) — backbone 동결, 선형 분류기만 학습.
- 그 외 image retrieval, copy detection, video segmentation 등.

사전학습 루프와 평가는 **서로 다른 스크립트, 서로 다른 데이터, 서로 다른 실행**이다.
이건 설계 누락이 아니라 자기지도 학습의 표준 관행이고,
"프리텍스트 태스크 loss ≠ 다운스트림 품질"이라는 전제를 코드 구조에 반영한 것이다.

### 유일한 자동 중단 조건은 NaN 가드뿐

```python
if not math.isfinite(loss.item()):
    print("Loss is {}, stopping training".format(loss.item()), force=True)
    sys.exit(1)
```

수치 폭발은 잡지만 **붕괴는 전혀 잡지 못한다.** 붕괴한 실행은 유한한 loss 로,
심지어 더 낮은 loss 로, 끝까지 멀쩡히 돌아간다.

---

## 3. 실전 함의 정리

| 함의 | 구체적으로 | 근거 |
|---|---|---|
| **조기 종료 불가** | 언제 멈춰야 좋은지 알려주는 신호가 없다. `--epochs` 로 정한 횟수를 그냥 다 돈다 | 루프에 종료 조건 자체가 없음 |
| **best 모델 선택 불가** | `checkpoint.pth` 는 매 epoch **덮어써지므로** 남는 건 **마지막 epoch** 뿐. best 를 고를 기준도, 고른 것을 저장할 코드도 없다 | `main_dino.py:288` |
| **중간 스냅샷은 수동** | `--saveckp_freq N` 을 줘야 `checkpoint{epoch:04}.pth` 가 남는다 (기본 20) | `main_dino.py:289-290` |
| **하이퍼파라미터 튜닝 비용이 폭발** | 한 설정의 좋고 나쁨을 알려면 **전체 학습을 끝내고** k-NN 을 돌려야 한다. ImageNet ViT-S/16 은 8 GPU 로 100 epoch 에 **약 1.75일** | 노트북 §11 |
| **붕괴 감지가 지연된다** | 붕괴는 첫 수백 step 안에 시작될 수 있는데, 로그만 보면 며칠 뒤 평가 시점까지 모른다 | §11 3-설정 실험 |
| **재현 성공 여부를 학습 중 알 수 없다** | "논문 수치를 재현 중인가?"는 끝난 뒤에만 판정 가능 | `docs/analysis/2026-09-04-ml-analysis.md` |

> **`saveckp_freq` 의 함정 두 가지**
> 1. 조건이 `epoch % saveckp_freq == 0` 이므로 **epoch 0 이 저장된다**(`checkpoint0000.pth` — 거의 무학습 상태).
> 2. **마지막 epoch 은 `args.epochs - 1` 이 `saveckp_freq` 의 배수가 아니면 스냅샷으로 남지 않는다.**
>    다만 `checkpoint.pth` 가 항상 마지막이므로 최종 모델을 잃지는 않는다.
> 3. 각 스냅샷은 student + teacher + optimizer 를 전부 담아 **모델 크기의 3배 이상**이다.
>    ViT-S/16 기준 한 파일이 수백 MB. `saveckp_freq` 를 너무 작게 잡으면 디스크가 먼저 죽는다.

관련 함정 하나 더 — `eval_linear.py` 는 `best_acc` 를 **추적해서 출력하지만**
저장하는 체크포인트는 `best` 가 아니라 **last** 다(`eval_linear.py:134` vs `:148`).
즉 리포트된 "Top-1 test accuracy" 와 디스크에 남은 가중치가 **다를 수 있다**.
이 저장소 전반이 "best 를 고르지 않는다"는 성향을 공유한다.

---

## 4. 권장 운영 패턴

### (a) 주기 저장 → 오프라인 k-NN 스윕 (가장 쉬움, 코드 수정 없음)

```bash
# 사전학습: 스냅샷을 촘촘히 남긴다
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
    --arch vit_small --data_path /path/to/imagenet/train \
    --output_dir out/dino_run --saveckp_freq 10
```

```bash
# 학습 중/후에 별도 프로세스로 스냅샷마다 평가
for ck in out/dino_run/checkpoint0*.pth; do
  echo "=== $ck ==="
  python eval_knn.py \
      --pretrained_weights "$ck" --checkpoint_key teacher \
      --arch vit_small --patch_size 16 \
      --data_path /path/to/eval_data \
      --nb_knn 10 20 --temperature 0.07 \
      --batch_size_per_gpu 128 --use_cuda True
done
```

- `--checkpoint_key teacher` 가 **기본값이자 옳은 선택**이다. DINO 의 배포 가중치는 항상 teacher —
  student 의 EMA 라 더 안정적이고 성능도 높다. student 를 평가하고 싶으면 명시적으로 바꿔야 한다.
- head 는 여기서 쓰이지 않는다. k-NN 은 backbone 의 CLS 특징만 본다 (head 는 버려지는 부분).
- `--dump_features <dir>` 로 특징을 저장해 두면 $k$, $T$ 를 바꿔가며 재계산 없이
  `--load_features` 로 재평가할 수 있다.

이렇게 만든 (epoch, k-NN top1) 곡선이 **사실상의 검증 곡선**이다.
학습을 중단시키지 않고 병렬로 돌릴 수 있는 것이 이 방식의 장점.

### (b) 학습 루프에 진단량 로깅 추가 (최소 패치, 강력 권장)

붕괴를 **며칠 뒤가 아니라 수백 step 안에** 잡으려면 이게 필요하다.
`train_one_epoch` 의 EMA 갱신 블록 바로 뒤에 아래를 넣는다:

```python
# main_dino.py, train_one_epoch 안 — EMA teacher 갱신 직후
with torch.no_grad():
    temp = dino_loss.teacher_temp_schedule[epoch]          # 현재 epoch 의 tau_t
    p_t = F.softmax((teacher_output.float() - dino_loss.center) / temp, dim=-1)
    metric_logger.update(H_t=(-(p_t * p_t.clamp_min(1e-12).log()).sum(-1)).mean().item())
    metric_logger.update(top1_t=p_t.max(-1).values.mean().item())
    metric_logger.update(uniq=float(p_t.argmax(-1).unique().numel()))
    metric_logger.update(cnorm=dino_loss.center.norm().item())
```

- 추가 비용은 사실상 0 — `teacher_output` 은 이미 계산돼 있고 `no_grad` 안이다.
- `metric_logger.update(...)` 만 하면 `train_stats` 를 거쳐 `log.txt` 에
  `train_H_t`, `train_top1_t`, `train_uniq`, `train_cnorm` 으로 **자동으로 흘러 들어간다**
  (`return {k: meter.global_avg for k, meter in metric_logger.meters.items()}`).
- DDP 에서는 `synchronize_between_processes()` 가 rank 평균을 내주므로 그대로 맞다.
  단 `uniq` 는 "rank-local 배치 내 고유 argmax 수의 평균"이라는 뜻이 되니 해석에 주의.
- **읽는 법**: $H(P_t)$ 가 $\log K$ 보다 확실히 낮은 값에서 안정되고, $\max_k P_t(k)$ 가
  $1/K$ 보다 크되 1 에서 멀며, `uniq` 가 배치 크기 근처를 유지하면 건강.
  $H(P_t) \to 0$ 이나 `uniq → 1` 이면 즉시 중단하고 설정을 고친다.

이 네 숫자만 있어도 **"돈 낭비하는 실행"을 조기에 죽일 수 있다** — 완전한 검증은 아니지만
투자 대비 효과가 가장 크다.

### (c) 작은 val 부분집합으로 epoch 끝마다 k-NN (제대로 된 검증 곡선)

`train_dino` 의 epoch 루프 안, `log_stats` 를 만들기 전에 넣는다.

```python
if utils.is_main_process() and args.knn_freq and epoch % args.knn_freq == 0:
    knn_top1 = quick_knn(teacher_without_ddp.backbone, knn_train_loader, knn_val_loader)
    log_stats["knn_top1"] = knn_top1
```

고려사항:

- **비용**: (train + val) 전체를 forward 해야 하므로 이미지 수에 비례한다.
  ImageNet 전체로 하면 epoch 시간이 눈에 띄게 늘어난다. **부분집합**을 쓸 것 —
  클래스당 수십 장 수준의 train bank + 몇 천 장 val 이면 상대 비교에는 충분하다.
- **레이블이 필요하다.** 사전학습 데이터와 별개로, 레이블이 있는 작은 프로브 셋을
  준비해야 한다. 이 프로브 셋은 절대 사전학습에 쓰지 않는다.
- **DDP 처리**: main process 에서만 돌리면 다른 rank 가 기다리다 timeout 날 수 있다.
  `dist.barrier()` 로 감싸거나, 아예 전 rank 가 나눠 계산하고 all-gather 하는 편이 안전하다
  (`eval_knn.py` 의 `extract_features` 가 이미 그렇게 되어 있다).
- **`model.eval()` / `model.train()` 복구를 잊지 말 것.** ViT 는 BN 이 없지만
  drop path / dropout 이 있고, backbone 이 아니라 `MultiCropWrapper` 를 통째로 넘기면
  crop 그룹핑 로직이 끼어든다 — `teacher_without_ddp.backbone` 을 직접 넘겨야 한다.
- 이 곡선이 있으면 비로소 **조기 종료와 best 선택이 가능해진다.**

---

## 5. `eval_knn.py` 가 저렴한 이유와, 그 함정

**학습 파라미터가 0개다.** backbone 을 얼리고 CLS 특징만 뽑아
코사인 유사도로 이웃을 찾을 뿐이다:

$$
\hat{y}(x) = \arg\max_{c}\ \sum_{i \in \mathcal{N}_k(x)}
\mathbb{1}[y_i = c]\cdot \exp\!\left(\frac{\cos(z_x, z_i)}{T}\right),
\qquad T = 0.07
$$

특징을 `F.normalize(..., p=2)` 로 L2 정규화한 뒤 내적을 쓰므로 내적 = 코사인 유사도다.
옵티마이저도, backward 도, 하이퍼파라미터 탐색도 없다 —
**forward 한 번 + top-k** 라서 linear probe 보다 훨씬 싸고, 그래서 반복 평가에 적합하다.
`--nb_knn 10 20 100 200` 처럼 여러 $k$ 를 한 번에 재도 특징 추출은 한 번뿐이다.

> **함정** (`eval_knn.py:146-149`): `knn_classifier` 가
> `imgs_per_chunk = num_test_images // 100` 으로 val 셋을 자르기 때문에
> **val 이미지가 100장 미만이면 `ValueError: range() arg 3 must not be zero`** 로 죽는다.
> 스모크 테스트용 소형 데이터셋을 만들 때 val 을 100장 이상 확보할 것.

관련 함정: **`eval_linear.py` 는 `--output_dir` 을 만들어 주지 않는다.**
`log.txt` 를 열 때 `FileNotFoundError` 로 죽으므로 미리 `mkdir -p` 해야 한다.

---

## 6. 실무 체크리스트

- [ ] 장기 실행 전에 **`--saveckp_freq` 를 명시**했는가? (기본 20; 디스크와 상의)
- [ ] 진단량(§4b) 로깅 패치를 넣었는가? 안 넣었다면 **최소한 첫 1~2 epoch 는 지켜보고**
      loss 가 비정상적으로 빨리 내려가지 않는지 확인했는가?
- [ ] `log.txt` 의 `train_loss` 가 $\log K$ 근처에 머무는 것을 **정상으로** 이해하고 있는가?
      (내려가지 않는다고 놀라서 설정을 건드리면 오히려 붕괴시킬 수 있다)
- [ ] 평가용 프로브 셋(레이블 있음, val ≥ 100장)을 사전학습 데이터와 **분리해서** 준비했는가?
- [ ] 평가할 때 `--checkpoint_key teacher` 인가?
- [ ] "best 체크포인트"를 기대하고 있지 않은가? — `checkpoint.pth` 는 **last** 다.

---

## 참고

- 원자료: `.fm/assets/dino_training_walkthrough.py` §10(1 iteration 해부), §11(미니 루프 + 붕괴 진단), §12(k-NN), §14(요약·함정)
- 코드: `main_dino.py:269-295` (epoch 루프), `main_dino.py:354-360` (로깅), `main_dino.py:364-416` (`DINOLoss`), `eval_knn.py:143-160` (`knn_classifier`), `eval_linear.py:112-150` (대조군 검증 루프)
- 분석 문서: `docs/analysis/2026-09-04-ml-analysis.md` (§체크포인트/검증 비교표, "사전학습 검증 부재" 항목)
- 실행 샘플: `SAMPLES.md` §4 (k-NN), §5 (linear probe)
- 논문: [Emerging Properties in Self-Supervised Vision Transformers, arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
