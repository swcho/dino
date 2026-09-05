# DINO의 손실 항 개수 $|\mathcal{N}|$은 어떻게 계산되는가?

$|\mathcal{N}| = 2(2+N) - 2$. 교사 view 2개 × 학생 view $(2+N)$개에서 같은 view 쌍 2개를 뺀 값이며,
$N=8$이면 이미지마다 18개 항이 만들어진다.

## 시각화

![expy 시각화](expy.png)
