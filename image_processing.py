import cv2
import numpy as np
import os
import random

# 유전 알고리즘의 주요 설정
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    processed_image = np.zeros((height, width, 3), dtype=np.uint8)
    valid_positions = []

    for y in range(height):
        for x in range(width):
            intensity = image[y, x]
            if intensity < 50:
                attenuation_24ghz = 80
                attenuation_5ghz = 100
            else:
                attenuation_24ghz = 30
                attenuation_5ghz = 50
                valid_positions.append((x, y))  # 유효한 위치 저장
            processed_image[y, x] = [attenuation_24ghz, 0, attenuation_5ghz]

    return processed_image, valid_positions

def calculate_fitness(positions, image):
    fitness = 0
    for x, y in positions:
        signal_strength = 100 - image[y, x, 0]  # 2.4GHz 감쇄 반영 예시
        fitness += signal_strength
    return fitness

def select_mating_pool(population, fitness_scores):
    selected = random.choices(population, weights=fitness_scores, k=len(population))
    return selected

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(child, valid_positions):
    if random.random() < MUTATION_RATE:
        ap_to_mutate = random.randint(0, len(child) - 1)
        child[ap_to_mutate] = random.choice(valid_positions)  # 유효한 위치에서 선택
    return child

def genetic_algorithm(image, width, height, ap_count, valid_positions):
    population = [[random.choice(valid_positions) for _ in range(ap_count)]
                  for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(individual, image) for individual in population]
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parents = select_mating_pool(population, fitness_scores)
            child1 = crossover(parents[0], parents[1])
            child2 = crossover(parents[1], parents[0])
            new_population.extend([mutate(child1, valid_positions), mutate(child2, valid_positions)])

        population = new_population

    best_solution = max(population, key=lambda ind: calculate_fitness(ind, image))
    return best_solution

def process_image_for_ap_placement(image_path, output_folder, ap_count):
    image, valid_positions = load_and_preprocess_image(image_path)
    height, width, _ = image.shape

    best_positions = genetic_algorithm(image, width, height, ap_count, valid_positions)
    
    # 최적의 AP 배치를 이미지에 표시
    for (x, y) in best_positions:
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    
    output_path = os.path.join(output_folder, f'processed_{os.path.basename(image_path)}')
    cv2.imwrite(output_path, image)
    
    return f'outputs/processed_{os.path.basename(image_path)}'
