import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
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
    velocities = [np.random.uniform(-1, 1, (num_routers, 2)) for _ in range(swarm_size)]
    personal_best_positions = swarm.copy()
    personal_best_scores = [objective_function(pos, internal_coordinates, frequency) for pos in swarm]
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)
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
    return global_best_position

# 프로그램 실행
if __name__ == "__main__":
    wall_coords, internal_coords = extract_wall_and_internal_coordinates(image_path)

    # 라우터 배치 최적화 실행
    num_routers = 5
    frequency = 2400  # MHz
    coverage_radius = 50  # arbitrary units

    best_positions = pso_optimization(internal_coords, num_routers, frequency, coverage_radius)
    # 결과 시각화
    x, y = zip(*internal_coords)
    plt.scatter(x, y, s=1, label="Internal Area", color="gray")

    router_x, router_y = zip(*best_positions)
    plt.scatter(router_x, router_y, color="red", label="Router Positions")

    for router in best_positions:
        circle = plt.Circle(router, coverage_radius, color='blue', alpha=0.3)
        plt.gca().add_artist(circle)

    plt.legend()
    plt.show()
