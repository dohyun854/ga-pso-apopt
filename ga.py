import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from image import extract_wall_and_internal_coordinates

def signal_strength(distance, frequency):
    if distance == 0:
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

def ga_optimization(internal_coordinates, num_routers, frequency, coverage_radius, 
                    population_size=50, generations=100, mutation_rate=0.1):
    population = [
        np.random.choice(internal_coordinates, num_routers, replace=False) 
        for _ in range(population_size)
    ]
    
    def fitness(individual):
        return -objective_function(individual, internal_coordinates, frequency)

    def crossover(parent1, parent2):
        split = np.random.randint(1, num_routers)
        child = np.vstack((parent1[:split], parent2[split:]))
        return child

    def mutate(individual):
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(0, num_routers)
            individual[idx] = internal_coordinates[np.random.randint(0, len(internal_coordinates))]
        return individual

    for generation in range(generations):
        fitness_scores = np.array([fitness(individual) for individual in population])
        
        probabilities = np.exp(fitness_scores) / np.sum(np.exp(fitness_scores))
        selected_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
        selected_population = [population[i] for i in selected_indices]

        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[(i + 1) % population_size]
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.append(child1)
            new_population.append(child2)

        population = new_population

        best_individual = population[np.argmax(fitness_scores)]
        best_score = -min(fitness_scores)
        print(f"Generation {generation + 1}, Best Score: {best_score}")

    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

if __name__ == "__main__":
    image_path = 'path_to_your_image.png'
    wall_coords, internal_coords = extract_wall_and_internal_coordinates(image_path)

    num_routers = 5
    frequency = 2400  # MHz
    coverage_radius = 50  # arbitrary units

    best_positions = ga_optimization(internal_coords, num_routers, frequency, coverage_radius)

    x, y = zip(*internal_coords)
    plt.scatter(x, y, s=1, label="Internal Area", color="gray")

    router_x, router_y = zip(*best_positions)
    plt.scatter(router_x, router_y, color="red", label="Router Positions")

    for router in best_positions:
        circle = plt.Circle(router, coverage_radius, color='blue', alpha=0.3)
        plt.gca().add_artist(circle)

    plt.legend()
    plt.show()
