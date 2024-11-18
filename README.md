### 변수 정의
- \( n \): 공기 수
- \( x_n \): 공기의 \( x \)좌표
- \( y_n \): 공기의 \( y \)좌표
- \( d \): 최소 거리
- \( f \): 공기의 간섭식

### 목적 함수
\[
\min \sqrt{\sum_{i=1}^n \left( I(x, y) - a_i \right)^2}
\]

### 제약 조건
- \( d > 0 \)
- \( f > 0 \)
- \( n > 0 \)