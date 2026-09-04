# DINO 실행 샘플 모음

이 저장소를 실제로 돌려 보기 위한 샘플 커맨드 모음입니다.
아래 커맨드는 전부 다음 환경에서 **직접 실행해 성공을 확인**했습니다.

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24GB) x1 |
| Python | 3.10.21 (conda env `trellis`) |
| PyTorch / torchvision | 2.4.0+cu121 / 0.19.0+cu121 |
| 보조 패키지 | Pillow 10.4, scikit-image 0.25, matplotlib 3.10, opencv 4.11, submitit |

원 저장소는 PyTorch 1.7.1 기준으로 작성됐지만, torch 2.4에서도 코드 수정 없이 동작합니다
(`torch.cuda.amp` FutureWarning만 출력됨).

---

## 0. 준비

추가 설치 없이 현재 `trellis` 환경 그대로 돌아갑니다. 확인만 하려면:

```bash
python -c "import torch, torchvision, cv2, skimage; print(torch.__version__, torch.cuda.is_available())"
# -> 2.4.0+cu121 True
```

스모크 테스트용 소형 데이터셋(ImageFolder 구조)은 헬퍼 스크립트로 만듭니다.

```bash
# 다운로드 없이 즉시 생성: 5개 클래스 x (train 60 / val 40)
python samples/make_tiny_dataset.py --out out/dino_tiny --source synth

# 실제 이미지로 하고 싶으면 CIFAR-10 부분집합 (최초 1회 ~170MB 다운로드)
python samples/make_tiny_dataset.py --out out/dino_cifar --source cifar
```

두 경로 모두 검증했습니다. `cifar` 는 10개 클래스 x (train 60 / val 40) = 1000장을
96x96 로 만들며, 출력 구조는 `synth` 와 동일합니다.
다운로드가 느린 회선에서는 최초 1회에 15분 이상 걸릴 수 있습니다
(이 환경 실측 ~200KB/s). `_cache/cifar-10-python.tar.gz` 를 미리 복사해 두면 건너뜁니다.

생성되는 구조는 `eval_knn.py` / `eval_linear.py`가 기대하는 형태와 같습니다:

```
/tmp/dino_tiny/
  train/{a,b,c,d,e}/*.png
  val/{a,b,c,d,e}/*.png
```

---

## 1. Self-attention 시각화 — 가장 빠른 "돌아간다" 확인 :white_check_mark:

데이터셋도, 체크포인트도 필요 없습니다. 사전학습 가중치(ViT-S/8, 83MB)와 논문 예시 이미지를
자동으로 내려받습니다.

```bash
python visualize_attention.py \
    --arch vit_small --patch_size 8 \
    --image_size 480 480 \
    --threshold 0.6 \
    --output_dir out/dino_attn
```

출력: `img.png`, 헤드별 어텐션 히트맵 `attn-head0..5.png`,
`--threshold` 를 준 경우 마스크 오버레이 `mask_th0.6_head0..5.png`.
자기 이미지로 보려면 `--image_path my.jpg` 를 추가하세요. GPU 없이 CPU로도 돌아갑니다.

## 2. 사전학습 백본 불러오기 (torch.hub)

로컬 [hubconf.py](hubconf.py) 를 통해 바로 특징 추출기로 씁니다.

```bash
python - <<'PY'
import torch
m = torch.hub.load('.', 'dino_vits16', source='local').eval()
with torch.no_grad():
    f = m(torch.randn(1, 3, 224, 224))
print(f.shape)   # -> torch.Size([1, 384])
PY
```

원격에서 받으려면 `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`.
사용 가능한 이름: `dino_vits16`, `dino_vits8`, `dino_vitb16`, `dino_vitb8`,
`dino_xcit_*`, `dino_resnet50`.

## 3. DINO 학습 스모크 테스트 (GPU 1장, 수 초)

ImageNet 전체 학습은 8 GPU로 1.75일 걸립니다. 아래는 **파이프라인이 도는지만 확인**하는 용도입니다.

```bash
python main_dino.py \
    --arch vit_tiny --patch_size 16 \
    --data_path out/dino_tiny/train \
    --output_dir out/dino_train \
    --epochs 2 --warmup_epochs 0 \
    --batch_size_per_gpu 8 --num_workers 2 \
    --local_crops_number 4 --saveckp_freq 1
```

`utils.init_distributed_mode` 가 단일 GPU 실행을 지원하므로 `torchrun` 없이 그냥 `python` 으로 돌아갑니다.
2 epoch / 120장 기준 약 4초, 피크 VRAM 1.4GB. 결과는 `checkpoint.pth` 와 `log.txt`.

> **함정:** `--warmup_epochs`(기본 10) 가 `--epochs` 보다 크면
> [utils.py:197](utils.py#L197) 의 `assert len(schedule) == epochs * niter_per_ep` 에서 죽습니다.
> 짧게 돌릴 땐 `--warmup_epochs 0` 을 반드시 주세요.

실제 ImageNet 학습(8 GPU 노드 1대):

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
    --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/save
```

Slurm 다중 노드는 `python run_with_submitit.py --nodes 2 --ngpus 8 ...` (submitit 설치됨).

## 4. k-NN 평가

```bash
python eval_knn.py \
    --arch vit_small --patch_size 16 \
    --data_path out/dino_tiny \
    --batch_size_per_gpu 32 --num_workers 4 \
    --nb_knn 10 20 --use_cuda True
```

`--pretrained_weights` 를 생략하면 공식 DINO 가중치를 자동으로 씁니다.
`--source cifar` 로 만든 실제 이미지 셋(600 train / 400 val, 10 클래스)에서는
ViT-S/16 frozen feature 기준 **Top1 87.0 / Top5 99.25 (20-NN)** 가 나옵니다 —
합성 데이터의 100%와 달리 이 수치는 백본이 실제로 동작하는지 판단할 근거가 됩니다.
직접 학습한 체크포인트를 쓰려면:

```bash
python eval_knn.py --pretrained_weights /tmp/dino_train/checkpoint.pth \
    --checkpoint_key teacher --arch vit_tiny --data_path /tmp/dino_tiny
```

> **함정:** [eval_knn.py:149](eval_knn.py#L149) 가 val 셋을 `len(val)//100` 크기로 자르기 때문에
> **val 이미지가 100장 미만이면 `ValueError: range() arg 3 must not be zero`** 로 죽습니다.
> 헬퍼 스크립트 기본값(val 200장)은 이 조건을 넘깁니다.

## 5. Linear probe 평가

```bash
mkdir -p /tmp/dino_linear          # <- 필수
python eval_linear.py \
    --arch vit_small --patch_size 16 \
    --data_path /tmp/dino_tiny \
    --output_dir /tmp/dino_linear \
    --epochs 2 --batch_size_per_gpu 32 --num_workers 4 \
    --num_labels 5 --val_freq 1
```

> **함정:** [eval_linear.py:139](eval_linear.py#L139) 는 `output_dir` 을 만들지 않고 바로 `log.txt` 를 열어서,
> 디렉터리가 없으면 첫 epoch 검증 직후 `FileNotFoundError` 로 죽습니다. 미리 `mkdir -p` 하세요.

ImageNet 사전학습 가중치의 성능만 재현하려면 (`--evaluate`):

```bash
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --data_path /path/to/imagenet
```

## 6. Self-attention 영상 생성

```bash
python video_generation.py \
    --arch vit_small --patch_size 8 \
    --input_path /path/to/video.mp4 \
    --output_path /tmp/dino_video \
    --resize 240 --video_format mp4
```

`frames/`(추출 프레임), `attention/`(프레임별 어텐션), `video.mp4` 가 생깁니다.
이미 추출한 프레임 폴더를 `--input_path` 로 줘도 되고,
어텐션 이미지 폴더만 있으면 `--video_only` 로 영상만 만들 수 있습니다.
OpenCV가 `MP4V tag is not supported` 경고를 내지만 `mp4v` 로 폴백해 정상 생성됩니다.

---

## 외부 데이터셋이 필요한 평가들

아래 3개는 스크립트 자체는 정상 임포트/파싱되지만(확인 완료), 별도 데이터 준비가 필요해
이번에 end-to-end 실행은 하지 않았습니다.

| 스크립트 | 필요한 데이터 | 준비 |
|---|---|---|
| [eval_video_segmentation.py](eval_video_segmentation.py) | DAVIS 2017 | `git clone https://github.com/davisvideochallenge/davis-2017 && ./data/get_davis.sh` |
| [eval_image_retrieval.py](eval_image_retrieval.py) | revisited Oxford / Paris | [revisitop](https://github.com/filipradenovic/revisitop) 절차 참고 |
| [eval_copy_detection.py](eval_copy_detection.py) | Copydays (+ YFCC100M distractor) | [INRIA Copydays](https://lear.inrialpes.fr/~jegou/data.php#copydays) |

DAVIS 평가는 원 README에서 "PyTorch 1.7.1이 아니면 결과 재현이 안 된다"고 명시하고 있어,
torch 2.4인 이 환경에서는 수치가 논문과 다를 수 있습니다.

---

## 추천 순서

1. **§1 어텐션 시각화** — 환경이 멀쩡한지 1분 안에 확인
2. **§2 torch.hub** — 백본을 내 코드에 붙일 때의 최소 형태
3. **§0 헬퍼로 소형 데이터 생성 → §3 학습 → §4 k-NN → §5 linear** — 전체 파이프라인 스모크 테스트
4. 그 다음에 실제 ImageNet / DAVIS 등 큰 데이터로 확장
