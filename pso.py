import numpy as np
from scipy.spatial.distance import euclidean
from PIL import Image, ImageDraw
from image import extract_wall_and_internal_coordinates

# 신호 세기 계산 함수
def signal_strength(distance, frequency):
    if distance == 0:  # AP와 겹치는 경우 신호 무한대로 간주
        return -float('inf')
    return 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55

# 목적 함수
def objective_function(router_positions, internal_coordinates, frequency):
    signal_map = np.zeros(len(internal_coordinates))
    for idx, (x, y) in enumerate(internal_coordinates):
        signals = [
            signal_strength(euclidean((x, y), router), frequency)
            for router in router_positions
        ]
        signal_map[idx] = np.sum(signals)
    average_signal = np.mean(signal_map)
    return np.sqrt(np.mean((signal_map - average_signal) ** 2))

# PSO 알고리즘
def pso_optimization(internal_coordinates, num_routers, frequency, coverage_radius, max_iter=100, swarm_size=30):
    # 내부 좌표를 넘파이 배열로 변환
    internal_coordinates = np.array(internal_coordinates)

    # 초기화: 랜덤하게 라우터 위치 선택
    swarm = [internal_coordinates[np.random.choice(len(internal_coordinates), num_routers, replace=False)] for _ in range(swarm_size)]
    velocities = [np.random.uniform(-1, 1, (num_routers, 2)) for _ in range(swarm_size)]
    personal_best_positions = swarm.copy()
    personal_best_scores = [objective_function(pos, internal_coordinates, frequency) for pos in swarm]
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    # 이미지 생성 준비
    img_width, img_height = 500, 500  # 이미지 크기 설정
    img = Image.new('RGB', (img_width, img_height), color='white')  # 흰 배경
    draw = ImageDraw.Draw(img)

    # PSO 메인 루프
    for iteration in range(max_iter):
        for i in range(swarm_size):
            # 속도 업데이트
            inertia = 0.5
            cognitive = 1.5
            social = 1.5
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (personal_best_positions[i] - swarm[i])
                + social * r2 * (global_best_position - swarm[i])
            )
            # 위치 업데이트
            swarm[i] = swarm[i] + velocities[i]

            # 제한 조건: 폐곡선 내부만 허용
            valid_positions = [pos for pos in swarm[i] if tuple(pos) in internal_coordinates]
            if len(valid_positions) < num_routers:
                valid_positions = internal_coordinates[np.random.choice(len(internal_coordinates), num_routers, replace=False)]
            swarm[i] = np.array(valid_positions)

            # 점수 계산 및 업데이트
            score = objective_function(swarm[i], internal_coordinates, frequency)
            if score < personal_best_scores[i]:
                personal_best_positions[i] = swarm[i]
                personal_best_scores[i] = score

        # 글로벌 베스트 업데이트
        current_best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[current_best_idx] < global_best_score:
            global_best_position = personal_best_positions[current_best_idx]
            global_best_score = personal_best_scores[current_best_idx]

        # 중간 결과 업데이트
        router_x, router_y = zip(*global_best_position)

        # 이미지 초기화
        img = Image.new('RGB', (img_width, img_height), color='white')  # 배경을 흰색으로
        draw = ImageDraw.Draw(img)

        # 내부 영역 그리기
        for (x, y) in internal_coordinates:
            # 내부 좌표들을 회색으로 표시
            draw.point((x, y), fill='gray')

        # 라우터 위치 그리기
        for rx, ry in zip(router_x, router_y):
            draw.point((rx, ry), fill='red')  # 라우터 위치는 빨간색으로

            # 커버리지 영역 그리기
            for angle in range(0, 360, 10):
                x_offset = int(coverage_radius * np.cos(np.radians(angle)))
                y_offset = int(coverage_radius * np.sin(np.radians(angle)))
                draw.point((rx + x_offset, ry + y_offset), fill='blue')

        # 출력 이미지 저장
        output_image_path = 'optimized_ap_placement.png'
        img.save(output_image_path)

        print(f"Iteration {iteration + 1}, Best Score: {global_best_score}")

    return output_image_path

# 프로그램 실행
if __name__ == "__main__":
    image_path = 'shame.png'
    wall_coords, internal_coords = extract_wall_and_internal_coordinates(image_path)

    # 라우터 배치 최적화 실행
    num_routers = 5
    frequency = 2400  # MHz
    coverage_radius = 50  # arbitrary units

    output_image_path = pso_optimization(internal_coords, num_routers, frequency, coverage_radius)
    print(f"Optimized AP Placement Image saved at: {output_image_path}")
