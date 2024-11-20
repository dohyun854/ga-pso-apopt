### 변수

$d(x, y)$: AP와 점$(x, y)$사이의 거리  

$f$: 공유기의 주파수(Mhz)  

$n$:총 공유기 수  

$(x,y)$: 좌표 평면 위의 임의의 점  

$j$:$j$번째 공유기$(1\leq j\leq n,\,j\in\mathbb{N})$  

$S$: 지도 픽셀 좌표의 전체 집합  

$m$: $I_h$의 평균  



### 목적 함수

  

$\min_{(x, y) \in S} \sqrt{\frac{1}{n} \sum_{i=1}^n (m - I_h)^2}$  

$I_{h} = \displaystyle\sum_{j=1}^{n} I_j(x,y)$  

$I_{j(x,y)} = 20  \cdot  \log d(x, y) + 20  \cdot  \log f - 147.55$  

$h$: 위쪽 줄부터 오른쪽, 오른쪽이 더 이상 없다면 밑줄 순서대로 1부터 1씩 커지게 모든 픽셀들에게 부여된 값  

### 제약 조건

- $min{(d(x_a, y_b))} > 0$  

- $n\ge  1$  

- $f\gt 0$  