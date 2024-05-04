import random
from collections import namedtuple
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.backends import default_backend

# Starting time logging
start_time = datetime.now()

# Define namedtuple for storing chromosome (x, y) coordinates
Chromosome = namedtuple('Chromosome', ['x', 'y'])

def initialize_population(population_size, curve_parameters):
    population = []
    p = curve_parameters['p']  # Prime modulus
    a = curve_parameters['a']  # Curve coefficient
    b = curve_parameters['b']  # Curve coefficient

    for _ in range(population_size):
        # Generate random x and y coordinates within the curve's field
        x = random.randint(0, p - 1)
        y_squared = (x**3 + a * x + b) % p
        
        # Check if y_squared is a quadratic residue modulo p
        if pow(y_squared, (p - 1) // 2, p) == 1:
            y = pow(y_squared, (p + 1) // 4, p)  # Calculate y using modular square root
            population.append(Chromosome(x, y))
            population.append(Chromosome(x, p - y))  # Add the point's reflection

    return population

def calculate_fitness(chromosome, target_point, curve_parameters):
    p = curve_parameters['p']  # Prime modulus
    a = curve_parameters['a']  # Curve parameter a
    b = curve_parameters['b']  # Curve parameter b

    x_target, y_target = target_point

    # Check if the target_point is on the curve
    left_side = (y_target ** 2) % p
    right_side = (x_target ** 3 + a * x_target + b) % p
    if left_side != right_side:
        raise ValueError("Target point is not on the elliptic curve")

    # Calculate the distance between the chromosome and the target point
    distance_squared = (pow(chromosome.x - x_target, 2, p) + pow(chromosome.y - y_target, 2, p)) % p

    # Fitness value is inversely proportional to the distance
    fitness = 1 / (1 + distance_squared)
    return fitness

def tournament_selection(population, tournament_size, curve_parameters, target_point):
    # Create a tournament by randomly selecting chromosomes from the population
    tournament = random.sample(population, tournament_size)

    # Find the chromosome with the best fitness (closest to the target point)
    best_chromosome = None
    best_fitness = float('-inf')
    for chromosome in tournament:
        fitness = calculate_fitness(chromosome, target_point, curve_parameters)
        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome

    return best_chromosome

def select_parents(population, num_parents, tournament_size, curve_parameters, target_point):
    parents = []
    for _ in range(num_parents):
        parent = tournament_selection(population, tournament_size, curve_parameters, target_point)
        parents.append(parent)

    return parents

def uniform_crossover(parent1, parent2):
    offspring1_x = random.choice([parent1.x, parent2.x])
    offspring1_y = random.choice([parent1.y, parent2.y])
    offspring1 = Chromosome(offspring1_x, offspring1_y)

    offspring2_x = random.choice([parent1.x, parent2.x])
    offspring2_y = random.choice([parent1.y, parent2.y])
    offspring2 = Chromosome(offspring2_x, offspring2_y)

    return offspring1, offspring2

def mutate(chromosome, mutation_rate, curve_parameters):
    p = curve_parameters['p']  # Prime modulus

    mutated_x = chromosome.x
    if random.random() < mutation_rate:
        mutated_x = random.randint(0, p - 1)

    mutated_y = chromosome.y
    if random.random() < mutation_rate:
        y_squared = (mutated_x**3 + curve_parameters['a'] * mutated_x + curve_parameters['b']) % p
        if pow(y_squared, (p - 1) // 2, p) == 1:
            mutated_y = pow(y_squared, (p + 1) // 4, p)
        else:
            mutated_y = p - pow(y_squared, (p + 1) // 4, p)

    return Chromosome(mutated_x, mutated_y)

def genetic_algorithm(curve_parameters, target_point, population_size, mutation_rate, tournament_size, num_generations):
    # Initialize the population
    population = initialize_population(population_size, curve_parameters)

    # Run the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # Select parents for the next generation
        parents = select_parents(population, population_size, tournament_size, curve_parameters, target_point)

        # Create a new population by applying crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]

            # Perform uniform crossover
            offspring1, offspring2 = uniform_crossover(parent1, parent2)

            # Perform mutation
            mutated_offspring1 = mutate(offspring1, mutation_rate, curve_parameters)
            mutated_offspring2 = mutate(offspring2, mutation_rate, curve_parameters)

            new_population.append(mutated_offspring1)
            new_population.append(mutated_offspring2)

        # Update the population for the next generation
        population = new_population

    # Find the best chromosome (key pair) in the final population
    best_chromosome = None
    best_fitness = float('-inf')
    for chromosome in population:
        fitness = calculate_fitness(chromosome, target_point, curve_parameters)
        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome

    return best_chromosome

def convert_to_binary(text):
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def convert_to_text(binary):
    text = ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))
    return text

def perform_crossover(parents, crossover_points):
    children = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            if i // 2 < len(crossover_points):
                crossover_point = crossover_points[i // 2]
                child1 = parents[i][:crossover_point] + parents[i+1][crossover_point:]
                child2 = parents[i+1][:crossover_point] + parents[i][crossover_point:]
                children.append(child1)
                children.append(child2)
            else:
                children.append(parents[i])
                children.append(parents[i+1])
        else:
            children.append(parents[i])
    return children

def perform_mutation(crossed_data):
    mutated_data = []
    for chromosome in crossed_data:
        # Flip the first bit
        first_bit = '1' if chromosome[0] == '0' else '0'
        # Flip the last bit
        last_bit = '1' if chromosome[-1] == '0' else '0'
        # Keep the bits in between unchanged
        middle_bits = chromosome[1:-1]
        # Create the mutated chromosome
        mutated_chromosome = first_bit + middle_bits + last_bit
        mutated_data.append(mutated_chromosome)
    return mutated_data

def encrypt(plaintext, key, crossover_points):
    binary_plaintext = convert_to_binary(plaintext)
    print("Plaintext in binary: ", binary_plaintext)
    
    padding_length = 8 - (len(binary_plaintext) % 8)
    binary_plaintext += '0' * padding_length
    print("Padded length", binary_plaintext)
    
    binary_key = ''.join(format(num, '02b') for num in key)
    print("key", binary_key)
    binary_key = '0' * (len(binary_plaintext) - len(binary_key)) + binary_key
    print("padkey", binary_key)
    
    chromosomes = [binary_plaintext[i:i+8] for i in range(0, len(binary_plaintext), 8)]
    print("Segmented Data: ", chromosomes)

    chromosomes = perform_crossover(chromosomes, crossover_points)
    print("Crossover Data: ", chromosomes)

    chromosomes = perform_mutation(chromosomes)
    print("Mutated Data: ", chromosomes)

    diffused_data = ''.join(chromosomes)
    encrypted_data = ''.join(str(int(a) ^ int(b)) for a, b in zip(diffused_data, binary_key))
    print("Encrypted Data: ", encrypted_data)

    return encrypted_data

def decrypt(encrypted_data, key, crossover_points):
    binary_key = ''.join(format(num, '02b') for num in key)
    binary_key = '0' * (len(encrypted_data) - len(binary_key)) + binary_key
    
    decrypted_data = ''.join(str(int(a) ^ int(b)) for a, b in zip(encrypted_data, binary_key))
    print("Decrypted Data: ", decrypted_data)

    chromosomes = [decrypted_data[i:i+8] for i in range(0, len(decrypted_data), 8)]
    chromosomes = perform_mutation(chromosomes)
    print("Mutated Decrypted Data: ", chromosomes)

    chromosomes = perform_crossover(chromosomes, crossover_points)
    print("Crossover Decrypted Data: ", chromosomes)

    binary_plaintext = ''.join(chromosomes)
    
    # Remove the padding from the binary plaintext
    padding_length = binary_plaintext.count('0', -8)
    binary_plaintext = binary_plaintext[:-padding_length]
    print("Binary Decrypted Data: ", binary_plaintext)
    
    plaintext = convert_to_text(binary_plaintext)
    print("Decrypted Text: ", plaintext)

    return plaintext

# Example inputs
curve_parameters = {
    'p': 3,  # Prime modulus
    'a': 1,  # Curve coefficient a
    'b': 1    # Curve coefficient b
}

target_point = (
    1,
    0
)

population_size = 2
mutation_rate = 0.1
tournament_size = 2
num_generations = 2

# Print the best key pair found
print("----- KEYGEN -----")
best_key_pair = genetic_algorithm(curve_parameters, target_point, population_size, mutation_rate, tournament_size, num_generations)
key = (best_key_pair[0], best_key_pair[1])
print(f"Best key pair found: {best_key_pair}")
end_time = datetime.now()
print('Keygen Duration: {}'.format(end_time - start_time))

print("----- USAGE ----")
plaintext = input("Please enter a string: ")

# Crossover_points
user_input_crossover_points = input("Enter crossover points (comma-separated): ")
default_crossover_points = [2, 4, 6]
if user_input_crossover_points.strip() == "":
    crossover_points = default_crossover_points
else:
    crossover_points = [int(point.strip()) for point in user_input_crossover_points.split(",")]

print("Crossover Points: ",crossover_points)

print("----- ENCRYPTION -----")
encrypted_data = encrypt(plaintext, key, crossover_points)
print("----- DECRYPTION -----")
decrypted_text = decrypt(encrypted_data, key, crossover_points)