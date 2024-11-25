### 변수
- $d(x, y)$: AP와 점$(x, y)$사이의 거리  
- $f$: 공유기의 주파수(Mhz)  
- $n$: 총 공유기 수  
- $(x,y)$: 좌표 평면 위의 임의의 점  
- $j$: $j$ 번째 공유기 $(1\leq j\leq n,\,j\in\mathbb{N})$  
- $S$: 지도 픽셀 좌표의 전체 집합  
- $m$: $I_h$의 평균  

### 목적 함수


$min \ \sqrt{\frac{\sum_{i=1}^{n} (m - I_i)^2}{n}}$
$I_{h(a,b)}=\sum_{i=1}^{n}I_i$
$I_{j(a,b)} = 20 \cdot \log d(x_a, y_b) + 20 \cdot \log f - 147.55$
=======
\min \sqrt{\sum_{i=1}^n \left( I(x, y) - a_i \right)^2}



### 제약 조건

$d(x_a, y_b) > 0$
$n\ge j\ge 1$
=======
- $min{(d(x_a, y_b))} > 0$  
- $n\ge  1$  
- $f\gt 0$  

