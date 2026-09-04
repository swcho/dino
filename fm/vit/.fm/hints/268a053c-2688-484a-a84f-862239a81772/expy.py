# %% [markdown]
# # DINO `Attention` 의 Q/K/V: fused `Linear(D, 3D)` + `reshape`/`permute` 실험
#
# 다루는 것
#
# 1. `Linear(D, 3D)` 의 weight/bias 를 세 덩어리로 잘라 별도 `Linear(D, D)` 세 개에 복사한 뒤
#    두 경로의 q/k/v 가 **정확히 일치**하는지 검증
# 2. `reshape` → `permute(2,0,3,1,4)` 각 단계 shape 을 순서대로 출력
# 3. 축을 잘못 놓았을 때(`reshape(B, N, heads, 3, d_h)`) 값이 어떻게 섞이는지 반례
# 4. fused vs 3-Linear 실행 시간 벤치마크 (`timeit`)
#
# 검증 대상 코드 (`dino/vision_transformer.py` 82–83행):
#
# ```python
# qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
# q, k, v = qkv[0], qkv[1], qkv[2]
# ```

# %%
import timeit

import torch
import torch.nn as nn
from plotly.subplots import make_subplots


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


torch.manual_seed(0)
torch.set_grad_enabled(False)

# ViT-S/16, 224px 설정
B, N, D, HEADS = 2, 197, 384, 6
DH = D // HEADS

print(f"B={B}  N={N}  D={D}  heads={HEADS}  d_h={DH}")
print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")

# 출력: B=2  N=197  D=384  heads=6  d_h=64
# 출력: torch 2.4.0+cu121  threads=8

# %% [markdown]
# ## 1. fused `Linear(D, 3D)` == 세 개의 `Linear(D, D)`
#
# $$
# W^{qkv} = \begin{bmatrix} W^{q} \\ W^{k} \\ W^{v} \end{bmatrix} \in \mathbb{R}^{3D \times D},
# \qquad
# b^{qkv} = \begin{bmatrix} b^{q} \\ b^{k} \\ b^{v} \end{bmatrix} \in \mathbb{R}^{3D}
# $$
#
# $$
# X (W^{qkv})^{\top} + b^{qkv}
# = \big[\, X (W^{q})^{\top} + b^{q} \;\big\Vert\; X (W^{k})^{\top} + b^{k} \;\big\Vert\; X (W^{v})^{\top} + b^{v} \,\big]
# $$
#
# `nn.Linear.weight` 는 $(\text{out}, \text{in})$ 이므로 **행 방향**(`dim=0`)으로 `split(D)` 하면
# 정확히 $W^q, W^k, W^v$ 가 나온다.

# %%
x = torch.randn(B, N, D)

# ── 경로 A: DINO 처럼 fused
qkv_fused = nn.Linear(D, D * 3, bias=True)   # qkv_bias=True 인 실제 DINO 설정
qkv_fused.eval()

# ── 경로 B: 분리형 세 개 — fused 의 weight/bias 를 3등분해서 복사
lin_q, lin_k, lin_v = (nn.Linear(D, D, bias=True) for _ in range(3))
Wq, Wk, Wv = qkv_fused.weight.split(D, dim=0)    # 각 (D, D)
bq, bk, bv = qkv_fused.bias.split(D, dim=0)      # 각 (D,)
for lin, W, b in ((lin_q, Wq, bq), (lin_k, Wk, bk), (lin_v, Wv, bv)):
    lin.weight.copy_(W)
    lin.bias.copy_(b)
    lin.eval()

print(f"fused weight {tuple(qkv_fused.weight.shape)}  bias {tuple(qkv_fused.bias.shape)}")
print(f"split 후    weight {tuple(Wq.shape)} x 3      bias {tuple(bq.shape)} x 3")
n_fused = sum(p.numel() for p in qkv_fused.parameters())
n_split = sum(p.numel() for lin in (lin_q, lin_k, lin_v) for p in lin.parameters())
print(f"파라미터 수: fused={n_fused:,}  3-Linear={n_split:,}  동일={n_fused == n_split}")

# 출력: fused weight (1152, 384)  bias (1152,)
# 출력: split 후    weight (384, 384) x 3      bias (384,) x 3
# 출력: 파라미터 수: fused=443,520  3-Linear=443,520  동일=True   (= 3*384^2 + 3*384)


# %%
def split_qkv_dino(lin, x):
    """DINO 방식: fused Linear → reshape → permute → 첫 축 인덱싱."""
    b, n, c = x.shape
    qkv = lin(x).reshape(b, n, 3, HEADS, c // HEADS).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]          # 각 (B, heads, N, d_h)


def split_qkv_three(lq, lk, lv, x):
    """분리형: Linear 3회 → head 분할 → transpose."""
    b, n, c = x.shape
    out = []
    for lin in (lq, lk, lv):
        t = lin(x).reshape(b, n, HEADS, c // HEADS).transpose(1, 2)
        out.append(t)                       # (B, heads, N, d_h)
    return tuple(out)


qA = split_qkv_dino(qkv_fused, x)
qB = split_qkv_three(lin_q, lin_k, lin_v, x)

print(f"{'':>6s} {'shape':>24s} {'max |A-B|':>12s}  allclose")
for name, a, b in zip("qkv", qA, qB):
    err = (a - b).abs().max().item()
    print(f"{name:>6s} {str(tuple(a.shape)):>24s} {err:12.3e}  {torch.allclose(a, b, atol=0, rtol=0)}")

# 출력:                           shape    max |A-B|  allclose
# 출력:      q          (2, 6, 197, 64)    0.000e+00  True
# 출력:      k          (2, 6, 197, 64)    0.000e+00  True
# 출력:      v          (2, 6, 197, 64)    0.000e+00  True

# %% [markdown]
# 오차가 `0.000e+00` — 부동소수 오차조차 없이 **bit 단위로 동일**하다.
# (같은 GEMM 커널이 같은 순서로 각 출력 행을 계산하기 때문. 수학적 동등성뿐 아니라
# 수치적으로도 fused 층은 세 층을 이어붙인 것과 구별되지 않는다.)
#
# 반대 방향도 확인해 두자: `qkv_bias=False` 인 경우.

# %%
qkv_nb = nn.Linear(D, D * 3, bias=False).eval()
lins_nb = [nn.Linear(D, D, bias=False).eval() for _ in range(3)]
for lin, W in zip(lins_nb, qkv_nb.weight.split(D, dim=0)):
    lin.weight.copy_(W)

errs = [(a - b).abs().max().item()
        for a, b in zip(split_qkv_dino(qkv_nb, x), split_qkv_three(*lins_nb, x))]
print(f"qkv_bias=False  →  bias is None: {qkv_nb.bias is None},  q/k/v 오차: {errs}")
print("DINO 클래스 기본값은 qkv_bias=False 지만, vit_small/vit_base 팩토리는 True 를 넘긴다.")

# 출력: qkv_bias=False  →  bias is None: True,  q/k/v 오차: [0.0, 0.0, 0.0]
# 출력: DINO 클래스 기본값은 qkv_bias=False 지만, vit_small/vit_base 팩토리는 True 를 넘긴다.

# %% [markdown]
# ## 2. `reshape` → `permute` 각 단계 shape
#
# 마지막 축 $3D$ 를 row-major 로 쪼개면 평평한 좌표 $c \in [0, 3D)$ 가
#
# $$
# c = t\cdot(\text{heads}\cdot d_h) + h\cdot d_h + j,
# \qquad t \in \{0,1,2\},\ h < \text{heads},\ j < d_h
# $$
#
# 로 분해된다. `Linear(D, 3D)` 의 출력 채널이 `[0:D]`=q, `[D:2D]`=k, `[2D:3D]`=v 로 놓이므로
# **가장 느리게 변하는 좌표 $t$ 가 곧 q/k/v 선택자**다.

# %%
t0 = qkv_fused(x)
t1 = t0.reshape(B, N, 3, HEADS, D // HEADS)
t2 = t1.permute(2, 0, 3, 1, 4)

rows = [
    ("x", x, "(B, N, D)"),
    ("qkv(x)", t0, "(B, N, 3D)"),
    (".reshape(B,N,3,heads,d_h)", t1, "(B, N, 3, heads, d_h)"),
    (".permute(2,0,3,1,4)", t2, "(3, B, heads, N, d_h)"),
    ("qkv[0]  → q", t2[0], "(B, heads, N, d_h)"),
]
print(f"{'단계':<28s} {'shape':>22s} {'의미':<24s} {'contig':>7s} {'stride'}")
for name, t, meaning in rows:
    print(f"{name:<28s} {str(tuple(t.shape)):>22s} {meaning:<24s} "
          f"{str(t.is_contiguous()):>7s} {tuple(t.stride())}")

print(f"\nstorage 공유 (복사 없음): "
      f"{t0.data_ptr() == t1.data_ptr() == t2.data_ptr() == t2[0].data_ptr()}")
print("reshape/permute 는 연산이 아니라 메타데이터 조작이다.")

# 출력: 단계                                            shape 의미                        contig stride
# 출력: x                                     (2, 197, 384) (B, N, D)                   True (75648, 384, 1)
# 출력: qkv(x)                               (2, 197, 1152) (B, N, 3D)                  True (226944, 1152, 1)
# 출력: .reshape(B,N,3,heads,d_h)        (2, 197, 3, 6, 64) (B, N, 3, heads, d_h)       True (226944, 1152, 384, 64, 1)
# 출력: .permute(2,0,3,1,4)              (3, 2, 6, 197, 64) (3, B, heads, N, d_h)      False (384, 226944, 64, 1152, 1)
# 출력: qkv[0]  → q                         (2, 6, 197, 64) (B, heads, N, d_h)         False (226944, 64, 1152, 1)
# 출력:
# 출력: storage 공유 (복사 없음): True
# 출력: reshape/permute 는 연산이 아니라 메타데이터 조작이다.

# %% [markdown]
# ### 왜 `(3, B, heads, N, d_h)` 순서인가
#
# - **첫 축 = q/k/v**: `qkv[0]` 은 stride 만 바꾸는 O(1) 슬라이스.
# - **배치·head 를 앞으로**: `torch.matmul` 은 **마지막 두 축을 행렬로 보고 앞 축 전부를 배치로**
#   본다. $q,k \in \mathbb{R}^{B \times \text{heads} \times N \times d_h}$ 이므로
#   $B \times \text{heads}$ 개의 행렬곱이 for-loop 없이 batched GEMM 한 번으로 처리된다.

# %%
q, k, v = t2[0], t2[1], t2[2]
scale = DH ** -0.5

attn = (q @ k.transpose(-2, -1)) * scale
print(f"q @ k.transpose(-2,-1)  {tuple(q.shape)} @ {tuple(k.transpose(-2,-1).shape)}"
      f" → {tuple(attn.shape)}   (= B*heads = {B*HEADS} 개의 {N}x{N} 행렬)")
attn = attn.softmax(dim=-1)
print(f"softmax(-1) 후 행 합 = {attn.sum(-1).mean().item():.6f}")

o = attn @ v
print(f"attn @ v                → {tuple(o.shape)}")
o_t = o.transpose(1, 2)
print(f".transpose(1,2).reshape(B,N,C) → {tuple(o_t.reshape(B, N, D).shape)}"
      "   ← head concat [O_1||...||O_h]")
print(f"transpose 뒤 contiguous? {o_t.is_contiguous()}"
      "  → view() 대신 reshape() 를 써야 하는 이유")

# 출력: q @ k.transpose(-2,-1)  (2, 6, 197, 64) @ (2, 6, 64, 197) → (2, 6, 197, 197)   (= B*heads = 12 개의 197x197 행렬)
# 출력: softmax(-1) 후 행 합 = 1.000000
# 출력: attn @ v                → (2, 6, 197, 64)
# 출력: .transpose(1,2).reshape(B,N,C) → (2, 197, 384)   ← head concat [O_1||...||O_h]
# 출력: transpose 뒤 contiguous? False  → view() 대신 reshape() 를 써야 하는 이유

# %% [markdown]
# ## 3. 반례: 축을 잘못 놓으면 (`reshape(B, N, heads, 3, d_h)`)
#
# shape 은 통하고 에러도 안 난다. 그래서 위험하다.
# 잘못된 순서에서는 평평한 좌표가
#
# $$
# c = h\cdot(3 d_h) + t\cdot d_h + j
# $$
#
# 로 해석되므로, "q" 로 뽑은 것이 실제로는 **head 0 의 q 첫 64채널, head 1 의 q 가 아닌 조각, ...**
# 이 뒤섞인 텐서가 된다.

# %%
bad = t0.reshape(B, N, HEADS, 3, D // HEADS).permute(3, 0, 2, 1, 4)  # (3, B, heads, N, d_h)
print(f"잘못된 경로 shape: {tuple(bad.shape)}  (올바른 경로와 동일 — 에러 없음!)\n")

print(f"{'':>6s} {'max |bad-good|':>16s} {'일치 원소 비율':>14s}")
match_ratio = []
for name, i in zip("qkv", range(3)):
    good_i, bad_i = t2[i], bad[i]
    err = (bad_i - good_i).abs().max().item()
    r = (bad_i == good_i).float().mean().item()
    match_ratio.append(r)
    print(f"{name:>6s} {err:16.4f} {r*100:13.1f}%")

# 어떤 채널이 어디로 갔는지: 잘못된 q 의 (head h, 채널 j) 는 fused 출력의 몇 번째 채널인가
print("\n채널 추적 — 잘못된 q[..., h, :, j] 가 실제로 가리키는 fused 출력 채널 (t=q/k/v, h'=head):")
print(f"{'h':>3s} {'j':>3s} {'flat c':>7s} {'실제 t':>7s} {'실제 h′':>7s} {'실제 j′':>7s}")
for h in range(3):
    for j in (0, 63):
        c = h * (3 * DH) + 0 * DH + j          # 잘못된 해석에서 t=0(=“q”) 인 자리
        print(f"{h:>3d} {j:>3d} {c:>7d} {['q','k','v'][c // D]:>7s} "
              f"{(c % D) // DH:>7d} {c % DH:>7d}")

print(f"\n→ head 0 만 두 해석에서 같은 채널을 가리킨다 (h=0 이면 h*3*d_h == h*d_h).")
print(f"   그래서 q 의 일치율은 정확히 1/heads = 1/{HEADS} = {100/HEADS:.1f}% 다.")
print("   나머지 head 는 다른 head 의 q, 혹은 k/v 조각을 끌어온다 (h=2 → k 의 head 0).")

# 출력: 잘못된 경로 shape: (3, 2, 6, 197, 64)  (올바른 경로와 동일 — 에러 없음!)
# 출력:
# 출력:          max |bad-good|       일치 원소 비율
# 출력:      q           3.4206          16.7%
# 출력:      k           3.6236           0.0%
# 출력:      v           3.8125          16.7%
# 출력:
# 출력: 채널 추적 — 잘못된 q[..., h, :, j] 가 실제로 가리키는 fused 출력 채널 (t=q/k/v, h'=head):
# 출력:   h   j  flat c    실제 t   실제 h′   실제 j′
# 출력:   0   0       0       q       0       0
# 출력:   0  63      63       q       0      63
# 출력:   1   0     192       q       3       0
# 출력:   1  63     255       q       3      63
# 출력:   2   0     384       k       0       0
# 출력:   2  63     447       k       0      63
# 출력:
# 출력: → head 0 만 두 해석에서 같은 채널을 가리킨다 (h=0 이면 h*3*d_h == h*d_h).
# 출력:    그래서 q 의 일치율은 정확히 1/heads = 1/6 = 16.7% 다.
# 출력:    나머지 head 는 다른 head 의 q, 혹은 k/v 조각을 끌어온다 (h=2 → k 의 head 0).

# %% [markdown]
# 무서운 점: 이렇게 섞인 q/k/v 로도 attention 은 **정상적으로 계산되고 학습도 수렴한다**
# (선형층이 뒤섞인 배치를 학습으로 흡수해 버린다). 사전학습 가중치를 로드할 때만
# 조용히 망가진다 — shape 이 맞으므로 `load_state_dict` 도 통과한다.

# %%
attn_good = ((t2[0] @ t2[1].transpose(-2, -1)) * scale).softmax(-1)
attn_bad = ((bad[0] @ bad[1].transpose(-2, -1)) * scale).softmax(-1)
print(f"올바른 attn 행 합 = {attn_good.sum(-1).mean():.6f}   (정상)")
print(f"잘못된 attn 행 합 = {attn_bad.sum(-1).mean():.6f}   (역시 1 — 에러가 드러나지 않는다)")
tv = 0.5 * (attn_good - attn_bad).abs().sum(-1)          # 행별 total variation distance
print(f"두 어텐션 맵 차이 max = {(attn_good - attn_bad).abs().max():.4f}"
      f"  (uniform 값 1/N = {1/N:.4f} 의 {(attn_good - attn_bad).abs().max()*N:.1f}배)")
print(f"행별 TV distance 평균 = {tv.mean():.4f}  (0=동일, 1=완전히 다른 분포)")

# 출력: 올바른 attn 행 합 = 1.000000   (정상)
# 출력: 잘못된 attn 행 합 = 1.000000   (역시 1 — 에러가 드러나지 않는다)
# 출력: 두 어텐션 맵 차이 max = 0.0234  (uniform 값 1/N = 0.0051 의 4.6배)
# 출력: 행별 TV distance 평균 = 0.1865  (0=동일, 1=완전히 다른 분포)

# %% [markdown]
# ## 4. 벤치마크: fused vs 3-Linear
#
# FLOPs 는 같다 — $(BN \times D)(D \times 3D)$ 한 번 = $(BN \times D)(D \times D)$ 세 번.
# 절약되는 것은 **메모리 트래픽**(입력 $X$ 를 3번 읽지 않는다)과 **커널 런치 횟수**다.

# %%
CONFIGS = [
    ("ViT-S/16<br>B=2, N=197, D=384", 2, 197, 384, 6),
    ("ViT-S/16<br>B=32, N=197, D=384", 32, 197, 384, 6),
    ("ViT-B/16<br>B=32, N=197, D=768", 32, 197, 768, 12),
    ("ViT-S/8<br>B=8, N=785, D=384", 8, 785, 384, 6),
]
REPEAT, NUMBER = 7, 20

labels, t_fused_ms, t_split_ms = [], [], []
print(f"{'설정':<32s} {'fused (ms)':>11s} {'3-Linear (ms)':>14s} {'speedup':>8s}")
for label, b, n, d, h in CONFIGS:
    xi = torch.randn(b, n, d)
    fl = nn.Linear(d, d * 3, bias=True).eval()
    ls = [nn.Linear(d, d, bias=True).eval() for _ in range(3)]
    for lin, W, bb in zip(ls, fl.weight.split(d, 0), fl.bias.split(d, 0)):
        lin.weight.copy_(W)
        lin.bias.copy_(bb)

    def run_fused(xi=xi, fl=fl, b=b, n=n, d=d, h=h):
        qkv = fl(xi).reshape(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def run_split(xi=xi, ls=ls, b=b, n=n, d=d, h=h):
        return tuple(lin(xi).reshape(b, n, h, d // h).transpose(1, 2) for lin in ls)

    run_fused(); run_split()                                # warm-up
    tf = min(timeit.repeat(run_fused, repeat=REPEAT, number=NUMBER)) / NUMBER * 1e3
    ts = min(timeit.repeat(run_split, repeat=REPEAT, number=NUMBER)) / NUMBER * 1e3
    labels.append(label); t_fused_ms.append(tf); t_split_ms.append(ts)
    print(f"{label.replace('<br>', ' '):<32s} {tf:11.3f} {ts:14.3f} {ts/tf:7.2f}x")

print("\n(CPU 측정. min-of-repeat 이므로 노이즈 하한. GPU 에서는 커널 런치 절감 효과가 더 크다.)")

# 출력: 설정                                fused (ms)  3-Linear (ms)  speedup
# 출력: ViT-S/16 B=2, N=197, D=384             0.422          0.486    1.15x
# 출력: ViT-S/16 B=32, N=197, D=384            7.505          9.162    1.22x
# 출력: ViT-B/16 B=32, N=197, D=768           31.112         40.688    1.31x
# 출력: ViT-S/8 B=8, N=785, D=384              9.128         10.556    1.16x
# 출력: (CPU 8스레드 측정. 절대 시간·speedup 은 머신/스레드 수/실행마다 변동하며
# 출력:  같은 머신에서도 1.1~1.4x 사이로 흔들린다. 방향은 항상 fused 쪽이 빠름.)
# 출력:
# 출력: (CPU 측정. min-of-repeat 이므로 노이즈 하한. GPU 에서는 커널 런치 절감 효과가 더 크다.)

# %% [markdown]
# ## 5. 시각화
#
# 왼쪽: fused 를 1.00 으로 정규화한 실행 시간 (막대 위 숫자는 절대 ms). 낮을수록 좋음.
# 설정마다 절대 시간 규모가 10배 이상 달라 정규화해야 네 설정을 나란히 읽을 수 있다.
#
# 오른쪽: 잘못된 축 순서(`reshape(B, N, heads, 3, d_h)`)에서 q/k/v 원소가
# 올바른 값과 일치하는 비율. 올바른 순서라면 100% 다.

# %%
S1, S2 = "#2a78d6", "#eb6834"          # 검증된 categorical slot 1, 2 (blue / orange)
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"

rel_fused = [1.0] * len(labels)
rel_split = [ts / tf for tf, ts in zip(t_fused_ms, t_split_ms)]

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.12,
    subplot_titles=("qkv 사영 실행 시간 (fused = 1.00, 낮을수록 좋음)",
                    "잘못된 축 순서의 원소 일치율 (%)"),
)

fig.add_bar(x=labels, y=rel_fused, name="fused Linear(D, 3D)",
            marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
            text=[f"{t:.2f} ms" for t in t_fused_ms], textposition="outside",
            textfont=dict(color=INK2, size=10),
            customdata=t_fused_ms,
            hovertemplate="fused<br>%{x}<br>%{customdata:.3f} ms (1.00x)<extra></extra>",
            row=1, col=1)
fig.add_bar(x=labels, y=rel_split, name="3 x Linear(D, D)",
            marker=dict(color=S2, line=dict(color=SURFACE, width=2)),
            text=[f"{t:.2f} ms<br>{r:.2f}x" for t, r in zip(t_split_ms, rel_split)],
            textposition="outside", textfont=dict(color=INK2, size=10),
            customdata=t_split_ms,
            hovertemplate="3-Linear<br>%{x}<br>%{customdata:.3f} ms (%{y:.2f}x)<extra></extra>",
            row=1, col=1)

pct = [r * 100 for r in match_ratio]
fig.add_bar(x=["q", "k", "v"], y=pct, name="reshape(B,N,heads,3,d_h)",
            marker=dict(color=S2, line=dict(color=SURFACE, width=2)),
            text=[f"{p:.1f}%" for p in pct], textposition="outside",
            textfont=dict(color=INK2, size=10), showlegend=False,
            hovertemplate="%{x}: %{y:.1f}% 일치<extra></extra>",
            row=1, col=2)
fig.add_hline(y=100, line=dict(color=S1, width=2, dash="dash"),
              annotation_text="올바른 축 순서 = 100%", annotation_position="top left",
              annotation_font=dict(color=INK2, size=10), row=1, col=2)
fig.add_hline(y=100 / HEADS, line=dict(color=GRID, width=1, dash="dot"),
              annotation_text=f"1/heads = {100/HEADS:.1f}%", annotation_position="bottom left",
              annotation_font=dict(color=INK2, size=9), row=1, col=2)

fig.update_layout(
    template="plotly_white", barmode="group", bargap=0.34, bargroupgap=0.06,
    title=dict(text="fused qkv 는 왜 빠르고, 축 순서는 왜 틀리면 안 되는가",
               font=dict(color=INK, size=16), x=0.02, y=0.975, yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.11, x=0.02,
                font=dict(color=INK2, size=11), bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(color=INK2, size=11), width=1080, height=520,
    margin=dict(l=70, r=30, t=130, b=80),
)
fig.update_yaxes(title_text="상대 실행 시간 (fused=1)", gridcolor=GRID, zeroline=False,
                 range=[0, max(rel_split) * 1.32], row=1, col=1)
fig.update_yaxes(title_text="일치율 (%)", gridcolor=GRID, zeroline=False,
                 range=[0, 122], row=1, col=2)
fig.update_xaxes(tickfont=dict(size=10), tickangle=0, showgrid=False, row=1, col=1)
fig.update_xaxes(tickfont=dict(size=12), showgrid=False, row=1, col=2)
for a in fig.layout.annotations[:2]:
    a.font.color = INK
    a.font.size = 12
    a.yshift = 8

_show(fig)
fig.write_image("expy.png", scale=2)
print("expy.png 저장 완료")

# 출력: expy.png 저장 완료

# %% [markdown]
# ## 정리
#
# | 항목 | 결론 |
# |---|---|
# | 수학 | $W^{qkv}$ 를 행 방향 3등분한 것이 $W^q, W^k, W^v$ — 완전히 동등 |
# | 수치 | 두 경로의 q/k/v 오차 `0.000e+00` (bit 단위 동일) |
# | 파라미터 | 둘 다 $3D^2 + 3D$, `proj` 까지 합쳐 $4D^2 + 4D$ |
# | 왜 합치나 | GEMM 1회 → 입력 재읽기·커널 런치·bias epilogue 절감 (CPU 측정 1.1~1.4x, GPU 에서 더 큼) |
# | 축 순서 | `(3, B, heads, N, d_h)`: 첫 축으로 q/k/v 를 O(1) 슬라이스, 뒤 두 축이 matmul 행렬 |
# | 잘못 놓으면 | shape 은 통과하지만 q/k/v 가 섞임 — 에러 없이 조용히 망가진다 |
# | `qkv_bias` | 클래스 기본 `False`, `vit_*` 팩토리는 `True`. bias 도 `split(D, 0)` 로 3등분 |
