import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from image import extract_wall_and_internal_coordinates

def signal_strength(distance, frequency):
    if distance == 0:
        return -float('inf')
    return 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55

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

def pso_optimization(internal_coordinates, num_routers, frequency, coverage_radius, max_iter=100, swarm_size=30):
    internal_coordinates = np.array(internal_coordinates)

    swarm = [internal_coordinates[np.random.choice(len(internal_coordinates), num_routers, replace=False)] for _ in range(swarm_size)]
    velocities = [np.random.uniform(-1, 1, (num_routers, 2)) for _ in range(swarm_size)]
    personal_best_positions = swarm.copy()
    personal_best_scores = [objective_function(pos, internal_coordinates, frequency) for pos in swarm]
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    fig, ax = plt.subplots()
    x, y = zip(*internal_coordinates)
    ax.scatter(x, y, s=1, label="Internal Area", color="gray")

    for iteration in range(max_iter):
        for i in range(swarm_size):
            inertia = 0.5
            cognitive = 1.5
            social = 1.5
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (personal_best_positions[i] - swarm[i])
                + social * r2 * (global_best_position - swarm[i])
            )
            swarm[i] = swarm[i] + velocities[i]

            valid_positions = [pos for pos in swarm[i] if tuple(pos) in internal_coordinates]
            if len(valid_positions) < num_routers:
                valid_positions = internal_coordinates[np.random.choice(len(internal_coordinates), num_routers, replace=False)]
            swarm[i] = np.array(valid_positions)

            score = objective_function(swarm[i], internal_coordinates, frequency)
            if score < personal_best_scores[i]:
                personal_best_positions[i] = swarm[i]
                personal_best_scores[i] = score

        current_best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[current_best_idx] < global_best_score:
            global_best_position = personal_best_positions[current_best_idx]
            global_best_score = personal_best_scores[current_best_idx]

        router_x, router_y = zip(*global_best_position)
        ax.clear()
        ax.scatter(x, y, s=1, label="Internal Area", color="gray")
        ax.scatter(router_x, router_y, color="red", label="Router Positions")
        for router in global_best_position:
            circle = plt.Circle(router, coverage_radius, color='blue', alpha=0.3)
            ax.add_artist(circle)
        ax.legend()
        ax.set_title(f"Iteration {iteration + 1}, Best Score: {global_best_score}")
        plt.pause(0.1)

        print(f"Iteration {iteration + 1}, Best Score: {global_best_score}")

    plt.show()
    return global_best_position

if __name__ == "__main__":
    image_path = 'shame.png'
    wall_coords, internal_coords = extract_wall_and_internal_coordinates(image_path)

    num_routers = 5
    frequency = 2400  # MHz
    coverage_radius = 50  # arbitrary units

    best_positions = pso_optimization(internal_coords, num_routers, frequency, coverage_radius)
