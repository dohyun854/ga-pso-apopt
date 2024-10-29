from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import cv2
import logging

# Flask 앱 생성
app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)


# 신호 강도 계산 함수
def signal_strength(position, coverage_range):
    """주어진 위치와 커버리지 범위를 기반으로 신호 강도를 계산합니다."""
    distance = np.linalg.norm(position)  # 위치에서 원점까지의 거리 계산
    return coverage_range / (1 + distance)  # 거리 기반으로 신호 강도 계산


# 신호 감소 함수
def signal_decay(signal, distance, wall_penalty):
    """벽에 부딪히거나 거리가 멀어질 때 신호가 감소합니다."""
    return signal * (1 - wall_penalty / (1 + distance))


# AP와 디바이스 간의 신호 교환
def exchange_signals(ap_positions, device_positions, coverage_range,
                     wall_penalty):
    """AP와 디바이스 간의 신호 교환을 처리합니다."""
    signals_received = []
    for device in device_positions:
        received_signal = 0
        for ap in ap_positions:
            distance = np.linalg.norm(device - ap)
            if distance <= coverage_range:
                strength = signal_strength(ap, coverage_range)
                # 벽의 영향을 반영하여 신호 감소
                strength = signal_decay(strength, distance, wall_penalty)
                received_signal += strength
        signals_received.append(received_signal)
    return signals_received


# 적합도 함수
def fitness_function(ap_positions, coverage_range, binary_img, bandwidth,
                     num_users, wall_penalty):
    """AP의 위치에 대한 적합도를 평가합니다."""
    fitness = 0
    # 무작위 디바이스 위치 생성
    device_positions = np.random.uniform(0, binary_img.shape, (num_users, 2))
    signals_received = exchange_signals(ap_positions, device_positions,
                                        coverage_range, wall_penalty)

    # 신호 강도의 평균을 적합도로 사용
    fitness = np.mean(signals_received)
    return fitness  # 총 적합도 반환


# 유전 알고리즘 함수
def genetic_algorithm(num_generations, population_size, bounds, num_routers,
                      coverage_range, binary_img, bandwidth, num_users,
                      wall_penalty):
    """유전 알고리즘을 사용하여 최적의 AP 위치를 찾습니다."""
    population = np.random.uniform(bounds[0], bounds[1],
                                   (population_size, 2))  # 초기 인구 생성

    for generation in range(num_generations):
        fitness = np.array([fitness_function(population, coverage_range,
                                             binary_img, bandwidth, num_users,
                                             wall_penalty)
                            for _ in range(population_size)])
        selected = population[np.argsort(fitness)[-population_size //
                                                  2:]]  # 가장 적합한 개체 선택

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = selected[np.random.choice(len(selected), 2,
                                                         replace=False)]  # 부모 선택
            alpha = np.random.rand()  # 교배 비율
            child = alpha * parent1 + (1 - alpha) * parent2  # 자식 생성
            child += np.random.uniform(-1, 1, size=child.shape) * 0.1  # 돌연변이 추가
            child = np.clip(child, bounds[0], bounds[1])  # 경계 제한
            new_population.append(child)

        population = np.array(new_population)  # 새로운 세대로 갱신

    best_idx = np.argmax(fitness)  # 가장 적합한 개체 인덱스 찾기
    return population[best_idx]  # 최적의 위치 반환


# 신호 맵 생성 함수
def generate_signal_map(ap_positions, coverage_range, room_size, wall_penalty):
    """주어진 AP 위치에 대한 신호 맵을 생성합니다."""
    signal_map = np.zeros(room_size)  # 신호 맵 초기화
    for pos in ap_positions:
        for x in range(room_size[1]):
            for y in range(room_size[0]):
                if np.linalg.norm(pos - np.array([x, y])) <= coverage_range:
                    signal_map[y, x] += signal_strength(pos, coverage_range)

    return signal_map  # 최종 신호 맵 반환


# 이미지 인코딩 함수
def encode_image(signal_map):
    """신호 맵을 PNG 이미지로 인코딩하여 base64 문자열로 반환합니다."""
    plt.imshow(signal_map, cmap='hot', interpolation='nearest')  # 신호 맵 시각화
    plt.axis('off')  # 축 숨기기
    buf = io.BytesIO()  # 메모리 버퍼 생성
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)  # 이미지 저장
    plt.close()  # 플롯 닫기
    buf.seek(0)  # 버퍼 위치 초기화
    img_str = base64.b64encode(buf.read()).decode('utf-8')  # base64로 인코딩
    return f"data:image/png;base64,{img_str}"  # 데이터 URI 형식으로 반환


@app.route('/optimize', methods=['POST'])
def optimize():
    """최적화 요청을 처리합니다."""
    try:
        num_routers = int(request.json['num_routers'])
        coverage_range = float(request.json['coverage_range'])
        bandwidth = float(request.json['bandwidth'])
        num_users = int(request.json['num_users'])
        wall_penalty = float(request.json['wall_penalty'])  # 벽 패널티 추가

        image_data = request.json['image']

        # base64 이미지를 디코딩하여 읽기
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image_path = 'room_layout.png'  # 이미지 저장 경로
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # 이미지를 흑백으로 읽고 이진화
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_img = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY_INV)  # 이진화

        room_size = binary_img.shape  # 방의 크기
        bounds = np.array([[0, 0], [room_size[1], room_size[0]]])  # 경계 설정

        best_positions = []  # 최적의 위치 저장 리스트
        for _ in range(num_routers):
            best_pos = genetic_algorithm(num_generations=100,
                                         population_size=100,
                                         bounds=bounds,
                                         num_routers=num_routers,
                                         coverage_range=coverage_range,
                                         binary_img=binary_img,
                                         bandwidth=bandwidth,
                                         num_users=num_users,
                                         wall_penalty=wall_penalty)
            best_positions.append(best_pos.tolist())  # 최적 위치 리스트에 추가

        signal_map = generate_signal_map(best_positions, coverage_range,
                                         room_size, wall_penalty)  # 신호 맵 생성
        signal_image = encode_image(signal_map)  # 신호 맵 인코딩

        return jsonify({
            'positions': best_positions,
            'signal_image': signal_image
        })

    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")  # 에러 로그
        return jsonify({'error': str(e)}), 400  # 에러 메시지 반환


if __name__ == '__main__':
    app.run(debug=True)  # 디버그 모드로 Flask 앱 실행
