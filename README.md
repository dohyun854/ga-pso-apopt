# [2024 마포고 탐구 프로젝트 공모전] 유전 알고리즘과 입자 군집 최적화를 활용한 교내 무선 AP의 최적의 설치 위치 탐색
## [탐구 포스터 보기](https://1drv.ms/p/c/1fd85018d07015f9/EfkVcNAYUNgggB9iBgAAAAABuW6lsvetbd7F02LuV_V_3Q?e=GnKkig)
학교 : 마포고등학교  / 참여 학생 : 10705 김도현(조장), 10702 김건우, 10708 김지우, 10710 김현중​

### 변수
- $d(x,y)$: AP와 점 $(x,y)$사이의 거리  
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


### 제약 조건


- $min{(d(x_a, y_b))} > 0$   
