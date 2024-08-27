import numpy as np
import pyspiel
import random

class GeneticAlgorithm:
    def __init__(self, game, population_size, generations, mutation_rate, crossover_rate):
        self.game = game
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()
        self.current_individual = None

    def initialize_population(self):
        """Initializes the population with random strategies."""
        population = []
        for _ in range(self.population_size):
            individual = self.random_strategy()
            population.append(individual)
        return population

    def random_strategy(self):
        """Creates a random strategy for the game."""
        state = self.game.new_initial_state()
        actions = []
        while not state.is_terminal():
            legal_actions = state.legal_actions()  # Get all legal actions
            if not legal_actions:
                break  # Break if no legal actions are available

            random_action = random.choice(legal_actions)  # Choose a random action from the legal ones
            actions.append(random_action)
            state.apply_action(random_action)  # Apply the selected action

        return actions

    def fitness(self, individual):
        """Evaluates the fitness of an individual."""
        state = self.game.new_initial_state()
        for action in individual:
            if state.is_terminal():
                break
            state.apply_action(action)
        return state.returns()

    def selection(self, population, fitness_scores):
        """Selects individuals based on their fitness."""
        selected = random.choices(population, weights=fitness_scores, k=2)
        return selected

    def crossover(self, parent1, parent2):
        """Performs crossover between two parents to produce offspring."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2
        return child1, child2

    def mutate(self, individual):
        """Mutates an individual by randomly changing one of its actions."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # Get a new random action from the legal actions of the current state
                state = self.game.new_initial_state()
                legal_actions = state.legal_actions()
                if legal_actions:
                    individual[i] = random.choice(legal_actions)
        return individual

    def evolve(self):
        """Runs the genetic algorithm to evolve the population."""
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in self.population]
            new_population = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(self.population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population
            print(f"Generation {generation + 1}, Best Fitness: {max(fitness_scores)}")

    def step(self, state):
        """Selects an action based on the current state of the game."""
        if self.current_individual is None:
            # Choose the best individual from the initial population
            fitness_scores = [self.fitness(ind) for ind in self.population]
            self.current_individual = self.population[fitness_scores.index(max(fitness_scores))]
        
        action = self.current_individual[state.current_player()]
        return action

    import numpy as np
import pyspiel
import random

class GeneticAlgorithm:
    def __init__(self, game, population_size, generations, mutation_rate, crossover_rate):
        self.game = game
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()
        self.current_individual = None

    def initialize_population(self):
        """Initializes the population with random strategies."""
        population = []
        for _ in range(self.population_size):
            individual = self.random_strategy()
            population.append(individual)
        return population

    def random_strategy(self):
        """Creates a random strategy for the game."""
        state = self.game.new_initial_state()
        actions = []
        while not state.is_terminal():
            legal_actions = state.legal_actions()  # Get all legal actions
            if not legal_actions:
                break  # Break if no legal actions are available

            random_action = random.choice(legal_actions)  # Choose a random action from the legal ones
            actions.append(random_action)
            state.apply_action(random_action)  # Apply the selected action

        return actions

    def fitness(self, individual):
        """Evaluates the fitness of an individual."""
        state = self.game.new_initial_state()
        for action in individual:
            if state.is_terminal():
                break
            state.apply_action(action)
        return state.returns()

    def selection(self, population, fitness_scores):
        """Selects individuals based on their fitness."""
        selected = random.choices(population, weights=fitness_scores, k=2)
        return selected

    def crossover(self, parent1, parent2):
        """Performs crossover between two parents to produce offspring."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2
        return child1, child2

    def mutate(self, individual):
        """Mutates an individual by randomly changing one of its actions."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # Get a new random action from the legal actions of the current state
                state = self.game.new_initial_state()
                legal_actions = state.legal_actions()
                if legal_actions:
                    individual[i] = random.choice(legal_actions)
        return individual

    def evolve(self):
        """Runs the genetic algorithm to evolve the population."""
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in self.population]
            new_population = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(self.population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population
            print(f"Generation {generation + 1}, Best Fitness: {max(fitness_scores)}")

    def step(self, state):
        """Selects an action based on the current state of the game."""
        best_individual = None
        best_fitness = -float('inf')

        for individual in self.population:
            # Apply the individual's strategy up to the current state
            current_state = self.game.new_initial_state()
            for action in individual:
                if current_state.is_terminal():
                    break
                current_state.apply_action(action)

            # Evaluate fitness only up to the current state
            fitness = current_state.returns()[state.current_player()]
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual

        action_index = len(state.history())
        if action_index < len(best_individual):
            action = best_individual[action_index]
            if action in state.legal_actions():
                return action
    
        # If the selected action is not legal, choose a random legal action
        return random.choice(state.legal_actions())
        # Pick the action corresponding to the current game state
        # action_index = len(state.history())
        # return best_individual[action_index] if action_index < len(best_individual) else random.choice(state.legal_actions())
    

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions."""
        action_string = state.action_to_string(player_id, action)
        print(f"Player {player_id} took action: {action_string}")
        
    def restart(self):
        """Resets the board to the initial state."""
        self.state = self.game.new_initial_state()
        self.current_individual = None