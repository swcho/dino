# 로짓이 코사인 유사도임을 어떻게 수치로 확인했는가?

`F.normalize(head.last_layer.weight_v, dim=-1)` 과 정규화된 병목 출력의 내적을 직접 계산해
실제 로짓과 비교하면 float32 반올림 오차(카드 기준 `8.9e-08`) 안에서 일치한다.

## 시각화

![expy 시각화](expy.png)
