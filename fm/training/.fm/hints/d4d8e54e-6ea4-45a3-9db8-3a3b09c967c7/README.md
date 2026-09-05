# `out_dim=65536`일 때 ViT-S의 파라미터 규모는 어떻게 되는가?

DINOHead가 약 22.4M으로 backbone(21.7M)보다 커진다. 마지막 층 하나가 $256 \times 65536 = 16.8$M을 차지하기 때문이다.

## 시각화

![expy 시각화](expy.png)
